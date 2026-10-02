from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .geotrax_pixel import GeoTraxPixelPoint, load_geotrax_pixel_tracks
from .pixel_benchmark import (
    FluidPixelTruth,
    PixelBenchmark,
    _SUPPORTED_GEOTRAX_CLASSES,
)
from .recall_loss_diagnosis import load_fluid_truth_all_classes


PREFERRED_LONG_GAP_IDS = (
    "3014",
    "2339",
    "3051",
    "3189",
    "2705",
    "1433",
    "537",
    "2453",
    "230",
    "1880",
)

ALLOWED_REVIEW_LABELS = (
    "VISIBLE_CLEAR",
    "PARTIAL_OCCLUSION",
    "HEAVY_OCCLUSION",
    "SMALL_FAR_OBJECT",
    "EDGE_OF_FRAME",
    "DENSE_TRAFFIC",
    "CLASS_AMBIGUOUS",
    "PREDICTED_NEARBY_WRONG_ASSOCIATION",
    "PREDICTED_NEARBY_WRONG_CLASS",
    "NO_NEARBY_GEOTRAX",
    "OTHER",
)


@dataclass(frozen=True)
class TrackDiagnosticRecord:
    truth_id: str
    truth_class: str
    truth_frames: int
    matched_frames: int
    matched_fraction: float
    number_of_predicted_ids: int
    maximum_miss_gap_frames: int


@dataclass(frozen=True)
class OfficialMatch:
    truth_frame: int
    predicted_track_id: str
    predicted_class: str
    pixel_distance: float


@dataclass(frozen=True)
class NearbyPrediction:
    predicted_track_id: str
    predicted_class: str
    predicted_frame: int
    x_px: float
    y_px: float
    pixel_distance: float


@dataclass(frozen=True)
class AuditTrack:
    diagnostic: TrackDiagnosticRecord
    truth_points: tuple[FluidPixelTruth, ...]
    official_matches: tuple[OfficialMatch, ...]
    miss_runs: tuple[tuple[int, ...], ...]
    raw_truth_labels: tuple[tuple[int, str], ...] = ()

    @property
    def truth_id(self) -> str:
        return self.diagnostic.truth_id

    @property
    def truth_class(self) -> str:
        return self.diagnostic.truth_class

    @property
    def actual_truth_frames(self) -> tuple[int, ...]:
        return tuple(point.frame for point in self.truth_points)

    @property
    def matched_frames(self) -> int:
        return self.diagnostic.matched_frames

    @property
    def matched_fraction(self) -> float:
        return self.diagnostic.matched_fraction

    @property
    def predicted_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {match.predicted_track_id for match in self.official_matches},
                key=_track_id_key,
            )
        )

    @property
    def maximum_miss_gap_frames(self) -> int:
        return max((len(run) for run in self.miss_runs), default=0)

    @property
    def raw_fluid_classes(self) -> tuple[str, ...]:
        labels = {label for _frame, label in self.raw_truth_labels}
        if not labels:
            labels = {self.truth_class}
        return tuple(sorted(labels))

    def raw_fluid_class_at(self, frame: int) -> str:
        for label_frame, label in self.raw_truth_labels:
            if label_frame == frame:
                return label
        return self.truth_class


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    sample_group: str
    track: AuditTrack
    target_gap: tuple[int, ...] | None
    sampled_frames: tuple[int, ...]

    @property
    def truth_id(self) -> str:
        return self.track.truth_id

    @property
    def matched_fraction(self) -> float:
        return self.track.matched_fraction


def _track_id_key(track_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)


def consecutive_frame_runs(frames: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    runs: list[list[int]] = []
    for frame in sorted(set(frames)):
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return tuple(tuple(run) for run in runs)


def _quantile_frames(frames: Sequence[int], fractions: Sequence[float]) -> tuple[int, ...]:
    if not frames:
        return ()
    last_index = len(frames) - 1
    selected = {
        frames[min(last_index, int(last_index * fraction + 0.5))]
        for fraction in fractions
    }
    return tuple(sorted(selected))


def representative_lifetime_frames(
    truth_frames: Sequence[int],
) -> tuple[int, ...]:
    return _quantile_frames(
        tuple(sorted(set(truth_frames))),
        (0.0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1.0),
    )


def representative_gap_frames(
    truth_frames: Sequence[int],
    gap_frames: Sequence[int],
) -> tuple[int, ...]:
    actual = set(truth_frames)
    gap = tuple(sorted(set(gap_frames)))
    if not gap:
        return ()
    selected = set(_quantile_frames(gap, (0.0, 0.25, 0.5, 0.75, 1.0)))
    if gap[0] - 1 in actual:
        selected.add(gap[0] - 1)
    if gap[-1] + 1 in actual:
        selected.add(gap[-1] + 1)
    return tuple(sorted(selected))


def nearby_predictions(
    truth_point: FluidPixelTruth,
    predicted: Sequence[GeoTraxPixelPoint],
    *,
    frame_offset: int = 1,
    radius_px: float = 100.0,
) -> tuple[NearbyPrediction, ...]:
    nearby: list[NearbyPrediction] = []
    for point in predicted:
        if point.frame + frame_offset != truth_point.frame:
            continue
        distance = math.hypot(
            point.x_px - truth_point.x_px,
            point.y_px - truth_point.y_px,
        )
        if distance <= radius_px:
            nearby.append(
                NearbyPrediction(
                    predicted_track_id=point.track_id,
                    predicted_class=point.vehicle_class,
                    predicted_frame=point.frame,
                    x_px=point.x_px,
                    y_px=point.y_px,
                    pixel_distance=float(distance),
                )
            )
    return tuple(
        sorted(
            nearby,
            key=lambda row: (row.pixel_distance, _track_id_key(row.predicted_track_id)),
        )
    )


def load_track_diagnostics(path: str | Path) -> list[TrackDiagnosticRecord]:
    result: list[TrackDiagnosticRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            result.append(
                TrackDiagnosticRecord(
                    truth_id=str(row["truth_id"]),
                    truth_class=str(row["truth_class"]),
                    truth_frames=int(row["truth_frames"]),
                    matched_frames=int(row["matched_frames"]),
                    matched_fraction=float(row["matched_fraction"]),
                    number_of_predicted_ids=int(row["number_of_predicted_ids"]),
                    maximum_miss_gap_frames=int(row["maximum_miss_gap_frames"]),
                )
            )
    return result


def load_raw_fluid_labels(
    path: str | Path,
) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("cx") in (None, "") or row.get("cy") in (None, ""):
                continue
            key = (str(row["id"]), int(float(row["frame"])))
            raw_label = str(row.get("type", "unknown"))
            existing = labels.get(key)
            if existing is not None and existing != raw_label:
                raise ValueError(
                    "Conflicting raw FLUID labels for truth track "
                    f"{key[0]} at frame {key[1]}"
                )
            labels[key] = raw_label
    return labels


def build_audit_tracks(
    predicted: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth],
    diagnostics: Sequence[TrackDiagnosticRecord],
    *,
    raw_truth_labels: Mapping[tuple[str, int], str] | None = None,
    frame_offset: int = 1,
    max_distance_px: float = 50.0,
) -> tuple[AuditTrack, ...]:
    motorcycle_diagnostic_ids: set[str] = set()
    for diagnostic in diagnostics:
        if diagnostic.truth_class != "MOTORCYCLE":
            continue
        if diagnostic.truth_id in motorcycle_diagnostic_ids:
            raise ValueError(
                "Duplicate motorcycle diagnostic truth_id: "
                f"{diagnostic.truth_id}"
            )
        motorcycle_diagnostic_ids.add(diagnostic.truth_id)

    predicted_supported = [
        point
        for point in predicted
        if point.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
    ]
    if not predicted_supported:
        raise ValueError("No supported Geo-trax observations are available")
    frame_start = min(point.frame + frame_offset for point in predicted_supported)
    frame_end = max(point.frame + frame_offset for point in predicted_supported)
    truth_window = [
        point for point in truth if frame_start <= point.frame <= frame_end
    ]
    accepted = PixelBenchmark(max_distance_px=max_distance_px).match_points(
        predicted,
        truth_window,
        frame_offset=frame_offset,
    )

    truth_by_id: dict[str, dict[int, FluidPixelTruth]] = defaultdict(dict)
    for point in truth_window:
        if point.vehicle_class != "MOTORCYCLE":
            continue
        if point.frame in truth_by_id[point.track_id]:
            raise ValueError(
                f"Duplicate FLUID observation for truth track {point.track_id} "
                f"at frame {point.frame}"
            )
        truth_by_id[point.track_id][point.frame] = point

    matches_by_id: dict[str, list[OfficialMatch]] = defaultdict(list)
    for predicted_point, truth_point, distance in accepted:
        if truth_point.vehicle_class != "MOTORCYCLE":
            continue
        matches_by_id[truth_point.track_id].append(
            OfficialMatch(
                truth_frame=truth_point.frame,
                predicted_track_id=predicted_point.track_id,
                predicted_class=predicted_point.vehicle_class,
                pixel_distance=distance,
            )
        )

    tracks: list[AuditTrack] = []
    for diagnostic in diagnostics:
        if diagnostic.truth_class != "MOTORCYCLE":
            continue
        points_by_frame = truth_by_id.get(diagnostic.truth_id)
        if not points_by_frame:
            raise ValueError(
                f"Motorcycle truth track {diagnostic.truth_id} is missing from "
                "the evaluation window"
            )
        points = tuple(points_by_frame[frame] for frame in sorted(points_by_frame))
        official_matches = tuple(
            sorted(
                matches_by_id.get(diagnostic.truth_id, []),
                key=lambda match: match.truth_frame,
            )
        )
        matched_frames = {match.truth_frame for match in official_matches}
        miss_runs = consecutive_frame_runs(set(points_by_frame) - matched_frames)
        predicted_ids = {match.predicted_track_id for match in official_matches}
        actual_max_gap = max((len(run) for run in miss_runs), default=0)
        actual_matched_fraction = len(matched_frames) / len(points)
        discrepancies = []
        if len(points) != diagnostic.truth_frames:
            discrepancies.append(
                f"truth_frames {len(points)} != {diagnostic.truth_frames}"
            )
        if len(matched_frames) != diagnostic.matched_frames:
            discrepancies.append(
                f"matched_frames {len(matched_frames)} != {diagnostic.matched_frames}"
            )
        if not math.isclose(
            actual_matched_fraction,
            diagnostic.matched_fraction,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            discrepancies.append(
                f"matched_fraction {actual_matched_fraction} != "
                f"{diagnostic.matched_fraction}"
            )
        if len(predicted_ids) != diagnostic.number_of_predicted_ids:
            discrepancies.append(
                "number_of_predicted_ids "
                f"{len(predicted_ids)} != {diagnostic.number_of_predicted_ids}"
            )
        if actual_max_gap != diagnostic.maximum_miss_gap_frames:
            discrepancies.append(
                f"maximum_miss_gap_frames {actual_max_gap} != "
                f"{diagnostic.maximum_miss_gap_frames}"
            )
        if discrepancies:
            raise ValueError(
                f"Track diagnostic mismatch for {diagnostic.truth_id}: "
                + "; ".join(discrepancies)
            )
        tracks.append(
            AuditTrack(
                diagnostic=diagnostic,
                truth_points=points,
                official_matches=official_matches,
                miss_runs=miss_runs,
                raw_truth_labels=tuple(
                    (
                        point.frame,
                        (
                            raw_truth_labels.get(
                                (point.track_id, point.frame),
                                point.vehicle_class,
                            )
                            if raw_truth_labels is not None
                            else point.vehicle_class
                        ),
                    )
                    for point in points
                ),
            )
        )
    return tuple(sorted(tracks, key=lambda track: _track_id_key(track.truth_id)))


def _longest_run(track: AuditTrack) -> tuple[int, ...]:
    if not track.miss_runs:
        return ()
    return min(track.miss_runs, key=lambda run: (-len(run), run[0]))


def _require_count(
    candidates: Sequence[AuditTrack],
    count: int,
    group: str,
) -> tuple[AuditTrack, ...]:
    if len(candidates) < count:
        raise ValueError(
            f"Sample group {group} requires {count} tracks; found {len(candidates)}"
        )
    return tuple(candidates[:count])


def select_audit_cases(
    tracks: Sequence[AuditTrack],
    *,
    cases_per_group: int = 5,
    preferred_long_gap_ids: Sequence[str] = PREFERRED_LONG_GAP_IDS,
) -> tuple[AuditCase, ...]:
    if cases_per_group <= 0:
        raise ValueError("cases_per_group must be positive")
    motorcycle = [track for track in tracks if track.truth_class == "MOTORCYCLE"]
    selected_ids: set[str] = set()

    complete_candidates = sorted(
        (track for track in motorcycle if track.matched_frames == 0),
        key=lambda track: (-len(track.truth_points), _track_id_key(track.truth_id)),
    )
    complete = _require_count(
        complete_candidates,
        cases_per_group,
        "completely_missed",
    )
    selected_ids.update(track.truth_id for track in complete)

    long_candidates = [
        track
        for track in motorcycle
        if track.truth_id not in selected_ids
        and track.matched_frames > 0
        and track.maximum_miss_gap_frames > 30
    ]
    preferred_order = {
        str(track_id): index for index, track_id in enumerate(preferred_long_gap_ids)
    }
    preferred = sorted(
        (track for track in long_candidates if track.truth_id in preferred_order),
        key=lambda track: preferred_order[track.truth_id],
    )
    remaining_long = sorted(
        (track for track in long_candidates if track.truth_id not in preferred_order),
        key=lambda track: (
            -track.maximum_miss_gap_frames,
            _track_id_key(track.truth_id),
        ),
    )
    long_gap = _require_count(
        (*preferred, *remaining_long),
        cases_per_group,
        "long_gap_gt30",
    )
    selected_ids.update(track.truth_id for track in long_gap)

    medium_candidates = sorted(
        (
            track
            for track in motorcycle
            if track.truth_id not in selected_ids
            and track.matched_frames > 0
            and 11 <= track.maximum_miss_gap_frames <= 30
        ),
        key=lambda track: (
            -track.maximum_miss_gap_frames,
            _track_id_key(track.truth_id),
        ),
    )
    medium = _require_count(
        medium_candidates,
        cases_per_group,
        "medium_gap_11_30",
    )
    selected_ids.update(track.truth_id for track in medium)

    control_candidates = sorted(
        (
            track
            for track in motorcycle
            if track.truth_id not in selected_ids and track.matched_frames > 0
        ),
        key=lambda track: (
            -track.matched_fraction,
            -len(track.truth_points),
            _track_id_key(track.truth_id),
        ),
    )
    controls = _require_count(
        control_candidates,
        cases_per_group,
        "high_recall_control",
    )

    groups = (
        ("A", "completely_missed", complete),
        ("B", "long_gap_gt30", long_gap),
        ("C", "medium_gap_11_30", medium),
        ("D", "high_recall_control", controls),
    )
    cases: list[AuditCase] = []
    for prefix, group_name, group_tracks in groups:
        for index, track in enumerate(group_tracks, start=1):
            if group_name == "completely_missed":
                target_gap = _longest_run(track)
                sampled = representative_lifetime_frames(track.actual_truth_frames)
            elif group_name == "high_recall_control":
                target_gap = None
                sampled = representative_lifetime_frames(track.actual_truth_frames)
            else:
                target_gap = _longest_run(track)
                sampled = representative_gap_frames(
                    track.actual_truth_frames,
                    target_gap,
                )
            cases.append(
                AuditCase(
                    case_id=f"{prefix}{index:02d}",
                    sample_group=group_name,
                    track=track,
                    target_gap=target_gap,
                    sampled_frames=sampled,
                )
            )
    return tuple(cases)


def _case_directory_name(case: AuditCase) -> str:
    safe_truth_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case.truth_id)
    return f"{case.case_id}_{safe_truth_id}"


def manifest_row(case: AuditCase, output_dir: str | Path) -> dict[str, object]:
    del output_dir
    if case.target_gap:
        gap_start: int | None = case.target_gap[0]
        gap_end: int | None = case.target_gap[-1]
        gap_length: int | None = len(case.target_gap)
    else:
        gap_start = None
        gap_end = None
        gap_length = None
    return {
        "case_id": case.case_id,
        "sample_group": case.sample_group,
        "truth_id": case.truth_id,
        "raw_fluid_class": ";".join(case.track.raw_fluid_classes),
        "canonical_truth_class": case.track.truth_class,
        "benchmark_supported": (
            case.track.truth_class in _SUPPORTED_GEOTRAX_CLASSES
        ),
        "truth_frames": case.track.diagnostic.truth_frames,
        "matched_frames": case.track.diagnostic.matched_frames,
        "matched_fraction": case.track.diagnostic.matched_fraction,
        "gap_start_frame": gap_start,
        "gap_end_frame": gap_end,
        "gap_length_frames": gap_length,
        "number_of_predicted_ids": case.track.diagnostic.number_of_predicted_ids,
        "maximum_miss_gap_frames": case.track.diagnostic.maximum_miss_gap_frames,
        "contact_sheet_path": (
            f"cases/{_case_directory_name(case)}/contact_sheet.jpg"
        ),
        "review_label": "",
        "review_notes": "",
    }


_MANIFEST_FIELDS = (
    "case_id",
    "sample_group",
    "truth_id",
    "raw_fluid_class",
    "canonical_truth_class",
    "benchmark_supported",
    "truth_frames",
    "matched_frames",
    "matched_fraction",
    "gap_start_frame",
    "gap_end_frame",
    "gap_length_frames",
    "number_of_predicted_ids",
    "maximum_miss_gap_frames",
    "contact_sheet_path",
    "review_label",
    "review_notes",
)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for visual audit export; run this script with "
            ".venv-geotrax\\Scripts\\python.exe"
        ) from exc
    return cv2


def _frame_metadata(
    case: AuditCase,
    truth_point: FluidPixelTruth,
    predicted_at_frame: Sequence[GeoTraxPixelPoint],
    *,
    frame_offset: int,
) -> dict[str, object]:
    official_by_frame = {
        match.truth_frame: match for match in case.track.official_matches
    }
    official = official_by_frame.get(truth_point.frame)
    nearby = nearby_predictions(
        truth_point,
        predicted_at_frame,
        frame_offset=frame_offset,
        radius_px=100.0,
    )
    if case.target_gap is None:
        gap_position = "lifetime control"
    elif truth_point.frame < case.target_gap[0]:
        gap_position = "before gap"
    elif truth_point.frame > case.target_gap[-1]:
        gap_position = "after gap"
    elif truth_point.frame == case.target_gap[0]:
        gap_position = "gap start"
    elif truth_point.frame == case.target_gap[-1]:
        gap_position = "gap end"
    else:
        gap_index = case.target_gap.index(truth_point.frame)
        gap_percent = round(100 * gap_index / max(1, len(case.target_gap) - 1))
        gap_position = f"gap {gap_percent}%"
    raw_fluid_class = case.track.raw_fluid_class_at(truth_point.frame)
    return {
        "source_video_frame": truth_point.frame,
        "video_zero_based_index": truth_point.frame - 1,
        "gap_category": case.sample_group,
        "gap_position": gap_position,
        "truth_point": {
            "truth_id": truth_point.track_id,
            "raw_fluid_class": raw_fluid_class,
            "truth_class": truth_point.vehicle_class,
            "canonical_class": truth_point.vehicle_class,
            "benchmark_supported": (
                truth_point.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
            ),
            "x_px": truth_point.x_px,
            "y_px": truth_point.y_px,
        },
        "matched": official is not None,
        "official_benchmark_match": (
            {
                "predicted_track_id": official.predicted_track_id,
                "predicted_class": official.predicted_class,
                "pixel_distance": official.pixel_distance,
            }
            if official is not None
            else None
        ),
        "nearby_geotrax_observations": [
            {
                "predicted_track_id": row.predicted_track_id,
                "predicted_class": row.predicted_class,
                "predicted_frame": row.predicted_frame,
                "x_px": row.x_px,
                "y_px": row.y_px,
                "pixel_distance": row.pixel_distance,
            }
            for row in nearby
        ],
        "image_path": f"frames/frame_{truth_point.frame:06d}.jpg",
        "raw_crop_path": f"crops/frame_{truth_point.frame:06d}_raw.png",
        "overlay_crop_path": (
            f"crops/frame_{truth_point.frame:06d}_overlay.png"
        ),
    }

def truth_crop_bounds(
    image_shape: Sequence[int],
    truth_x_px: float,
    truth_y_px: float,
    *,
    crop_size: int = 160,
) -> tuple[int, int, int, int]:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    crop_width = min(crop_size, image_width)
    crop_height = min(crop_size, image_height)
    center_x = int(round(truth_x_px))
    center_y = int(round(truth_y_px))
    left = min(max(0, center_x - crop_width // 2), image_width - crop_width)
    top = min(max(0, center_y - crop_height // 2), image_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def create_truth_crops(
    image,
    frame_metadata: dict[str, object],
    *,
    crop_size: int = 160,
    zoom_scale: int = 3,
):
    if zoom_scale <= 0:
        raise ValueError("zoom_scale must be positive")
    cv2 = _require_cv2()
    truth = frame_metadata["truth_point"]
    bounds = truth_crop_bounds(
        image.shape,
        truth["x_px"],
        truth["y_px"],
        crop_size=crop_size,
    )
    left, top, right, bottom = bounds
    raw_source = image[top:bottom, left:right].copy()
    overlay_source = raw_source.copy()
    official = frame_metadata["official_benchmark_match"]
    official_id = official["predicted_track_id"] if official else None

    for index, nearby in enumerate(
        frame_metadata["nearby_geotrax_observations"], start=1
    ):
        local_x = int(round(nearby["x_px"])) - left
        local_y = int(round(nearby["y_px"])) - top
        marker_x = min(max(local_x, 5), overlay_source.shape[1] - 6)
        marker_y = min(max(local_y, 5), overlay_source.shape[0] - 6)
        color = (
            (0, 255, 0)
            if nearby["predicted_track_id"] == official_id
            else (255, 255, 0)
        )
        cv2.circle(
            overlay_source,
            (marker_x, marker_y),
            6,
            color,
            2,
            cv2.LINE_8,
        )
        label_x = min(marker_x + 8, max(0, overlay_source.shape[1] - 12))
        label_y = min(
            max(11, marker_y - 8),
            max(11, overlay_source.shape[0] - 3),
        )
        cv2.putText(
            overlay_source,
            str(index),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    truth_local = (
        int(round(truth["x_px"])) - left,
        int(round(truth["y_px"])) - top,
    )
    cv2.circle(
        overlay_source,
        truth_local,
        7,
        (255, 0, 255),
        2,
        cv2.LINE_8,
    )
    zoom_size = (
        raw_source.shape[1] * zoom_scale,
        raw_source.shape[0] * zoom_scale,
    )
    raw_zoom = cv2.resize(
        raw_source,
        zoom_size,
        interpolation=cv2.INTER_NEAREST,
    )
    overlay_zoom = cv2.resize(
        overlay_source,
        zoom_size,
        interpolation=cv2.INTER_NEAREST,
    )
    return raw_zoom, overlay_zoom, bounds


def _context_view(image, bounds: tuple[int, int, int, int], panel_size: int):
    cv2 = _require_cv2()
    import numpy as np

    context = image.copy()
    left, top, right, bottom = bounds
    cv2.rectangle(
        context,
        (left, top),
        (max(left, right - 1), max(top, bottom - 1)),
        (0, 165, 255),
        2,
    )
    scale = min(panel_size / context.shape[1], panel_size / context.shape[0])
    resized = cv2.resize(
        context,
        (
            max(1, int(round(context.shape[1] * scale))),
            max(1, int(round(context.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((panel_size, panel_size, 3), dtype=image.dtype)
    x = (panel_size - resized.shape[1]) // 2
    y = (panel_size - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _compose_diagnostic_frame(
    image,
    frame_metadata: dict[str, object],
    *,
    case_id: str,
):
    cv2 = _require_cv2()
    import numpy as np

    raw_zoom, overlay_zoom, bounds = create_truth_crops(image, frame_metadata)
    frame_metadata["crop_bounds"] = list(bounds)
    frame_metadata["raw_crop_bounds"] = list(bounds)
    frame_metadata["overlay_crop_bounds"] = list(bounds)
    panel_size = raw_zoom.shape[1]
    context = _context_view(image, bounds, panel_size)
    gap = 12
    header_height = 100
    title_height = 34
    nearby_rows = frame_metadata["nearby_geotrax_observations"]
    legend_height = max(70, 34 + 25 * len(nearby_rows))
    width = panel_size * 3 + gap * 4
    visual_y = header_height + title_height
    output = np.zeros(
        (visual_y + panel_size + legend_height, width, 3),
        dtype=image.dtype,
    )
    output[:] = (18, 18, 18)
    columns = (
        (gap, "FULL CONTEXT", context),
        (gap * 2 + panel_size, "RAW ZOOM (NO OVERLAY)", raw_zoom),
        (gap * 3 + panel_size * 2, "OVERLAY ZOOM", overlay_zoom),
    )
    for x, title, panel in columns:
        cv2.putText(
            output,
            title,
            (x, header_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output[visual_y : visual_y + panel_size, x : x + panel_size] = panel

    truth = frame_metadata["truth_point"]
    status = "MATCHED" if frame_metadata["matched"] else "UNMATCHED"
    header_lines = (
        (
            f"Case {case_id} | Truth {truth['truth_id']} | "
            f"source frame {frame_metadata['source_video_frame']} | {status}"
        ),
        (
            f"Raw FLUID: {truth['raw_fluid_class']} | Canonical: "
            f"{truth['canonical_class']} | benchmark_supported: "
            f"{truth['benchmark_supported']}"
        ),
        (
            f"Category: {frame_metadata['gap_category']} | "
            f"Position: {frame_metadata['gap_position']}"
        ),
    )
    for index, line in enumerate(header_lines):
        cv2.putText(
            output,
            line,
            (gap, 27 + index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    official = frame_metadata["official_benchmark_match"]
    if official:
        official_text = (
            f"Official match: P{official['predicted_track_id']} | "
            f"{official['predicted_class']} | "
            f"{official['pixel_distance']:.2f}px"
        )
    else:
        official_text = "Official match: none"
    legend_y = visual_y + panel_size + 25
    cv2.putText(
        output,
        official_text,
        (gap, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    official_id = official["predicted_track_id"] if official else None
    for index, nearby in enumerate(nearby_rows, start=1):
        color = (
            (0, 255, 0)
            if nearby["predicted_track_id"] == official_id
            else (255, 255, 0)
        )
        text = (
            f"[{index}] P{nearby['predicted_track_id']} | "
            f"{nearby['predicted_class']} | {nearby['pixel_distance']:.1f}px"
        )
        cv2.putText(
            output,
            text,
            (gap, legend_y + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return output, raw_zoom, overlay_zoom


def _thumbnail(image, max_width: int = 640):
    cv2 = _require_cv2()
    if image.shape[1] <= max_width:
        return image.copy()
    scale = max_width / image.shape[1]
    return cv2.resize(
        image,
        (max_width, int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _write_contact_sheet(images: Sequence, path: Path) -> None:
    cv2 = _require_cv2()
    import numpy as np

    if not images:
        raise ValueError("Cannot create a contact sheet without images")
    columns = 1
    rows = math.ceil(len(images) / columns)
    cell_height = max(image.shape[0] for image in images)
    cell_width = max(image.shape[1] for image in images)
    sheet = np.zeros(
        (rows * cell_height, columns * cell_width, 3),
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = row * cell_height
        x = column * cell_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {path}")


def _point_coordinates(point: FluidPixelTruth) -> dict[str, float | int]:
    return {
        "frame": point.frame,
        "x_px": point.x_px,
        "y_px": point.y_px,
    }


def _truth_coordinate_statistics(case: AuditCase) -> dict[str, object]:
    points_by_frame = {point.frame: point for point in case.track.truth_points}
    sampled_points = [points_by_frame[frame] for frame in case.sampled_frames]
    selected_frames = (
        set(case.target_gap)
        if case.target_gap is not None
        else set(case.track.actual_truth_frames)
    )
    selected_points = [
        point
        for point in case.track.truth_points
        if point.frame in selected_frames
    ]
    first = sampled_points[0]
    last = sampled_points[-1]
    return {
        "first_sampled_truth": _point_coordinates(first),
        "last_sampled_truth": _point_coordinates(last),
        "sampled_pixel_displacement": math.hypot(
            last.x_px - first.x_px,
            last.y_px - first.y_px,
        ),
        "selected_interval_bounding_box": {
            "min_x_px": min(point.x_px for point in selected_points),
            "min_y_px": min(point.y_px for point in selected_points),
            "max_x_px": max(point.x_px for point in selected_points),
            "max_y_px": max(point.y_px for point in selected_points),
        },
    }


def _render_case(
    capture,
    case: AuditCase,
    predicted_by_truth_frame: dict[int, list[GeoTraxPixelPoint]],
    output_dir: Path,
    *,
    frame_offset: int,
) -> None:
    cv2 = _require_cv2()
    case_dir = output_dir / "cases" / _case_directory_name(case)
    frames_dir = case_dir / "frames"
    crops_dir = case_dir / "crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    truth_by_frame = {point.frame: point for point in case.track.truth_points}
    metadata_frames: list[dict[str, object]] = []
    thumbnails = []

    for source_frame in case.sampled_frames:
        truth_point = truth_by_frame[source_frame]
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame - 1)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(
                f"Could not read source video frame {source_frame} for "
                f"case {case.case_id}"
            )
        frame_metadata = _frame_metadata(
            case,
            truth_point,
            predicted_by_truth_frame.get(source_frame, []),
            frame_offset=frame_offset,
        )
        annotated, raw_crop, overlay_crop = _compose_diagnostic_frame(
            image,
            frame_metadata,
            case_id=case.case_id,
        )
        image_path = frames_dir / f"frame_{source_frame:06d}.jpg"
        raw_crop_path = crops_dir / f"frame_{source_frame:06d}_raw.png"
        overlay_crop_path = crops_dir / f"frame_{source_frame:06d}_overlay.png"
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError(f"Failed to write annotated frame: {image_path}")
        if not cv2.imwrite(str(raw_crop_path), raw_crop):
            raise RuntimeError(f"Failed to write raw crop: {raw_crop_path}")
        if not cv2.imwrite(str(overlay_crop_path), overlay_crop):
            raise RuntimeError(f"Failed to write overlay crop: {overlay_crop_path}")
        metadata_frames.append(frame_metadata)
        thumbnails.append(_thumbnail(annotated, max_width=1440))

    _write_contact_sheet(thumbnails, case_dir / "contact_sheet.jpg")
    metadata = {
        "case_id": case.case_id,
        "sample_group": case.sample_group,
        "truth_id": case.truth_id,
        "raw_fluid_classes": list(case.track.raw_fluid_classes),
        "truth_class": case.track.truth_class,
        "canonical_truth_class": case.track.truth_class,
        "benchmark_supported": (
            case.track.truth_class in _SUPPORTED_GEOTRAX_CLASSES
        ),
        "frame_offset": frame_offset,
        "official_match_distance_px": 50.0,
        "nearby_lookup_distance_px": 100.0,
        "sampled_frames": list(case.sampled_frames),
        "target_unmatched_interval": (
            {
                "start_frame": case.target_gap[0],
                "end_frame": case.target_gap[-1],
                "truth_observation_count": len(case.target_gap),
            }
            if case.target_gap
            else None
        ),
        "allowed_review_labels": list(ALLOWED_REVIEW_LABELS),
        "review_label": "",
        "review_notes": "",
        "truth_coordinate_statistics": _truth_coordinate_statistics(case),
        "frames": metadata_frames,
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_motorcycle_visual_audit(
    *,
    video: str | Path,
    fluid_tracks: str | Path,
    geotrax_tracks: str | Path,
    track_diagnostics: str | Path,
    output_dir: str | Path,
    frame_offset: int = 1,
    max_distance_px: float = 50.0,
) -> dict[str, object]:
    if frame_offset != 1:
        raise ValueError("The motorcycle visual audit requires frame_offset=1")
    if float(max_distance_px) != 50.0:
        raise ValueError(
            "The motorcycle visual audit requires max_distance_px=50.0"
        )

    predicted = load_geotrax_pixel_tracks(geotrax_tracks)
    truth = load_fluid_truth_all_classes(fluid_tracks)
    raw_truth_labels = load_raw_fluid_labels(fluid_tracks)
    diagnostics = load_track_diagnostics(track_diagnostics)
    tracks = build_audit_tracks(
        predicted,
        truth,
        diagnostics,
        raw_truth_labels=raw_truth_labels,
        frame_offset=frame_offset,
        max_distance_px=max_distance_px,
    )
    cases = select_audit_cases(tracks)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    predicted_by_truth_frame: dict[int, list[GeoTraxPixelPoint]] = defaultdict(list)
    for point in predicted:
        predicted_by_truth_frame[point.frame + frame_offset].append(point)

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open source video: {video}")
    try:
        for case in cases:
            _render_case(
                capture,
                case,
                predicted_by_truth_frame,
                target,
                frame_offset=frame_offset,
            )
    finally:
        capture.release()

    with (target / "audit_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(manifest_row(case, target))

    group_counts = Counter(case.sample_group for case in cases)
    return {
        "case_count": len(cases),
        "groups": dict(sorted(group_counts.items())),
        "frame_offset": frame_offset,
        "max_distance_px": float(max_distance_px),
        "nearby_distance_px": 100.0,
        "manifest": str(target / "audit_manifest.csv"),
    }
