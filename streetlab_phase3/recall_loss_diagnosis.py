from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from .class_mapping import map_vehicle_class
from .geotrax_pixel import GeoTraxPixelPoint, load_geotrax_pixel_tracks
from .identity_benchmark import IdentityBenchmark, TruthTrackIdentity
from .pixel_benchmark import (
    FluidPixelTruth,
    PixelBenchmark,
    PixelBenchmarkReport,
    _SUPPORTED_GEOTRAX_CLASSES,
)
from .serialization import package_to_dict


_GAP_BUCKETS = (
    "1 frame",
    "2-3 frames",
    "4-10 frames",
    "11-30 frames",
    "> 30 frames",
    "whole track missed",
)


@dataclass(frozen=True)
class PerClassRecall:
    truth_class: str
    benchmark_supported: bool
    truth_points: int
    matched_points: int
    missed_points: int
    point_recall: float | None
    truth_tracks: int
    matched_truth_tracks: int
    track_coverage: float | None
    mean_matched_fraction: float | None
    median_matched_fraction: float | None
    fraction_fragmented: float | None
    mean_predicted_ids_per_truth_track: float | None
    mean_longest_run_fraction: float | None
    median_longest_run_fraction: float | None


@dataclass(frozen=True)
class TrackRecallDiagnostic:
    truth_id: str
    truth_class: str
    truth_frames: int
    matched_frames: int
    matched_fraction: float
    number_of_predicted_ids: int
    fragmented: bool
    longest_matched_run_frames: int
    longest_run_fraction: float
    miss_event_count: int
    maximum_miss_gap_frames: int
    mean_miss_gap_frames: float
    first_match_delay_frames: int | None
    trailing_miss_frames: int | None


@dataclass(frozen=True)
class MissGapDistribution:
    truth_class: str
    gap_bucket: str
    gap_event_count: int
    gap_event_fraction: float
    missed_frame_count: int
    missed_frame_fraction: float


@dataclass(frozen=True)
class RecallLossDiagnosisResult:
    benchmark_compatible_report: PixelBenchmarkReport
    diagnostic_report: PixelBenchmarkReport
    per_class_recall: tuple[PerClassRecall, ...]
    track_diagnostics: tuple[TrackRecallDiagnostic, ...]
    miss_gap_distribution: tuple[MissGapDistribution, ...]
    report: dict[str, object]

    def report_dict(self) -> dict[str, object]:
        return self.report


def _optional_mean(values: Sequence[float | int]) -> float | None:
    return float(mean(values)) if values else None


def _optional_median(values: Sequence[float | int]) -> float | None:
    return float(median(values)) if values else None


def _consecutive_runs(frames: set[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for frame in sorted(frames):
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


def _gap_bucket(frame_count: int) -> str:
    if frame_count == 1:
        return "1 frame"
    if frame_count <= 3:
        return "2-3 frames"
    if frame_count <= 10:
        return "4-10 frames"
    if frame_count <= 30:
        return "11-30 frames"
    return "> 30 frames"


def load_fluid_truth_all_classes(path: str | Path) -> list[FluidPixelTruth]:
    result: list[FluidPixelTruth] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("cx") in (None, "") or row.get("cy") in (None, ""):
                continue
            result.append(
                FluidPixelTruth(
                    frame=int(float(row["frame"])),
                    track_id=str(row["id"]),
                    x_px=float(row["cx"]),
                    y_px=float(row["cy"]),
                    vehicle_class=map_vehicle_class(
                        row.get("type", "unknown")
                    ).canonical.value,
                )
            )
    return result


class RecallLossDiagnosis:
    def __init__(self, max_distance_px: float = 50.0) -> None:
        self.max_distance_px = float(max_distance_px)
        self.pixel_benchmark = PixelBenchmark(max_distance_px=max_distance_px)

    def evaluate(
        self,
        predicted: Sequence[GeoTraxPixelPoint],
        truth: Sequence[FluidPixelTruth],
        *,
        frame_offset: int = 1,
    ) -> RecallLossDiagnosisResult:
        supported_truth = [
            point
            for point in truth
            if point.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
        ]
        benchmark_compatible_report = self.pixel_benchmark.evaluate(
            predicted,
            supported_truth,
            frame_offsets=(frame_offset,),
        )
        diagnostic_report = self.pixel_benchmark.evaluate(
            predicted,
            truth,
            frame_offsets=(frame_offset,),
        )
        matches = self.pixel_benchmark.match_points(
            predicted,
            truth,
            frame_offset=frame_offset,
        )
        identity = IdentityBenchmark(
            max_distance_px=self.max_distance_px,
        ).evaluate(
            predicted,
            truth,
            frame_offsets=(frame_offset,),
        )

        if (
            diagnostic_report.evaluation_frame_start is None
            or diagnostic_report.evaluation_frame_end is None
        ):
            truth_window: list[FluidPixelTruth] = []
        else:
            truth_window = [
                point
                for point in truth
                if diagnostic_report.evaluation_frame_start
                <= point.frame
                <= diagnostic_report.evaluation_frame_end
            ]

        truth_by_track: dict[str, list[FluidPixelTruth]] = defaultdict(list)
        for point in truth_window:
            truth_by_track[point.track_id].append(point)

        matched_frames_by_truth: dict[str, set[int]] = defaultdict(set)
        matched_points_by_class: dict[str, int] = defaultdict(int)
        for _, truth_point, _ in matches:
            matched_frames_by_truth[truth_point.track_id].add(truth_point.frame)
            matched_points_by_class[truth_point.vehicle_class] += 1

        identity_by_truth = {
            detail.truth_track_id: detail
            for detail in identity.truth_track_details
        }
        track_diagnostics: list[TrackRecallDiagnostic] = []
        miss_runs_by_track: dict[str, list[list[int]]] = {}
        for truth_id, truth_points in sorted(truth_by_track.items()):
            truth_frames = {point.frame for point in truth_points}
            matched_frames = matched_frames_by_truth.get(truth_id, set())
            missed_frames = truth_frames - matched_frames
            miss_runs = _consecutive_runs(missed_frames)
            miss_runs_by_track[truth_id] = miss_runs
            miss_lengths = [len(run) for run in miss_runs]
            detail = identity_by_truth[truth_id]

            if matched_frames:
                first_match = min(matched_frames)
                last_match = max(matched_frames)
                first_match_delay = sum(frame < first_match for frame in truth_frames)
                trailing_misses = sum(frame > last_match for frame in truth_frames)
            else:
                first_match_delay = None
                trailing_misses = None

            track_diagnostics.append(
                TrackRecallDiagnostic(
                    truth_id=truth_id,
                    truth_class=detail.truth_class,
                    truth_frames=detail.truth_frame_count,
                    matched_frames=detail.matched_frame_count,
                    matched_fraction=detail.matched_fraction,
                    number_of_predicted_ids=detail.associated_predicted_id_count,
                    fragmented=detail.associated_predicted_id_count > 1,
                    longest_matched_run_frames=detail.longest_consecutive_match_run,
                    longest_run_fraction=detail.longest_run_fraction,
                    miss_event_count=len(miss_runs),
                    maximum_miss_gap_frames=max(miss_lengths, default=0),
                    mean_miss_gap_frames=(
                        float(mean(miss_lengths)) if miss_lengths else 0.0
                    ),
                    first_match_delay_frames=first_match_delay,
                    trailing_miss_frames=trailing_misses,
                )
            )

        details = tuple(track_diagnostics)
        detail_by_id = {detail.truth_id: detail for detail in details}
        identity_details_by_class: dict[str, list[TruthTrackIdentity]] = defaultdict(list)
        for detail in identity.truth_track_details:
            identity_details_by_class[detail.truth_class].append(detail)
        truth_points_by_class: dict[str, int] = defaultdict(int)
        for point in truth_window:
            truth_points_by_class[point.vehicle_class] += 1

        per_class_recall: list[PerClassRecall] = []
        for truth_class in sorted(truth_points_by_class):
            class_details = identity_details_by_class[truth_class]
            matched_details = [
                detail for detail in class_details if detail.matched_frame_count > 0
            ]
            truth_points = truth_points_by_class[truth_class]
            matched_points = matched_points_by_class[truth_class]
            per_class_recall.append(
                PerClassRecall(
                    truth_class=truth_class,
                    benchmark_supported=(
                        truth_class in _SUPPORTED_GEOTRAX_CLASSES
                    ),
                    truth_points=truth_points,
                    matched_points=matched_points,
                    missed_points=truth_points - matched_points,
                    point_recall=matched_points / truth_points,
                    truth_tracks=len(class_details),
                    matched_truth_tracks=len(matched_details),
                    track_coverage=(
                        len(matched_details) / len(class_details)
                        if class_details
                        else None
                    ),
                    mean_matched_fraction=_optional_mean(
                        [detail.matched_fraction for detail in class_details]
                    ),
                    median_matched_fraction=_optional_median(
                        [detail.matched_fraction for detail in class_details]
                    ),
                    fraction_fragmented=(
                        sum(
                            detail.associated_predicted_id_count > 1
                            for detail in matched_details
                        )
                        / len(matched_details)
                        if matched_details
                        else None
                    ),
                    mean_predicted_ids_per_truth_track=_optional_mean(
                        [
                            detail.associated_predicted_id_count
                            for detail in matched_details
                        ]
                    ),
                    mean_longest_run_fraction=_optional_mean(
                        [detail.longest_run_fraction for detail in class_details]
                    ),
                    median_longest_run_fraction=_optional_median(
                        [detail.longest_run_fraction for detail in class_details]
                    ),
                )
            )

        gap_counts: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {bucket: [0, 0] for bucket in _GAP_BUCKETS}
        )
        complete_miss_frames = 0
        short_gap_frames = 0
        long_gap_frames = 0
        for truth_id, truth_points in truth_by_track.items():
            diagnostic = detail_by_id[truth_id]
            scopes = ("ALL", diagnostic.truth_class)
            if diagnostic.matched_frames == 0:
                complete_miss_frames += diagnostic.truth_frames
                for scope in scopes:
                    gap_counts[scope]["whole track missed"][0] += 1
                    gap_counts[scope]["whole track missed"][1] += diagnostic.truth_frames
                continue

            for run in miss_runs_by_track[truth_id]:
                run_length = len(run)
                bucket = _gap_bucket(run_length)
                if run_length <= 10:
                    short_gap_frames += run_length
                else:
                    long_gap_frames += run_length
                for scope in scopes:
                    gap_counts[scope][bucket][0] += 1
                    gap_counts[scope][bucket][1] += run_length

        class_names = sorted(truth_points_by_class)
        distribution: list[MissGapDistribution] = []
        for scope in ("ALL", *class_names):
            counts = gap_counts[scope]
            total_events = sum(value[0] for value in counts.values())
            total_frames = sum(value[1] for value in counts.values())
            for bucket in _GAP_BUCKETS:
                event_count, frame_count = counts[bucket]
                distribution.append(
                    MissGapDistribution(
                        truth_class=scope,
                        gap_bucket=bucket,
                        gap_event_count=event_count,
                        gap_event_fraction=(
                            event_count / total_events if total_events else 0.0
                        ),
                        missed_frame_count=frame_count,
                        missed_frame_fraction=(
                            frame_count / total_frames if total_frames else 0.0
                        ),
                    )
                )

        per_class_tuple = tuple(per_class_recall)
        distribution_tuple = tuple(distribution)
        matched_identity_details = [
            detail
            for detail in identity.truth_track_details
            if detail.matched_frame_count > 0
        ]
        fragmented_tracks = sum(
            detail.associated_predicted_id_count > 1
            for detail in matched_identity_details
        )
        total_missed_frames = (
            complete_miss_frames + short_gap_frames + long_gap_frames
        )
        shares = {
            "complete_miss_share": (
                complete_miss_frames / total_missed_frames
                if total_missed_frames
                else 0.0
            ),
            "short_intermittent_gap_share": (
                short_gap_frames / total_missed_frames
                if total_missed_frames
                else 0.0
            ),
            "long_gap_share": (
                long_gap_frames / total_missed_frames
                if total_missed_frames
                else 0.0
            ),
        }
        if diagnostic_report.truth_points == 0:
            dominant_mode = "not evaluable"
        elif total_missed_frames:
            dominant_mode = max(
                (
                    (shares["complete_miss_share"], "complete misses"),
                    (
                        shares["short_intermittent_gap_share"],
                        "short intermittent gaps",
                    ),
                    (shares["long_gap_share"], "long gaps"),
                ),
                key=lambda item: item[0],
            )[1]
        else:
            dominant_mode = "no recall loss"
        lowest_recall = min(
            per_class_tuple,
            key=lambda row: (
                row.point_recall if row.point_recall is not None else float("inf"),
                row.truth_class,
            ),
            default=None,
        )
        completely_missed_tracks = sum(
            detail.matched_frames == 0 for detail in details
        )

        report: dict[str, object] = {
            "frame_offset": frame_offset,
            "max_distance_px": self.max_distance_px,
            "overall_held_out_statistics": {
                "all_class_diagnostic": package_to_dict(diagnostic_report),
                "benchmark_compatible": package_to_dict(
                    benchmark_compatible_report
                ),
            },
            "per_class_recall": [
                package_to_dict(row) for row in per_class_tuple
            ],
            "per_class_track_coverage": {
                row.truth_class: row.track_coverage for row in per_class_tuple
            },
            "fragmentation": {
                "matched_truth_tracks": len(matched_identity_details),
                "fragmented_truth_tracks": fragmented_tracks,
                "fraction_fragmented": (
                    fragmented_tracks / len(matched_identity_details)
                    if matched_identity_details
                    else None
                ),
            },
            "miss_gap_distribution": [
                package_to_dict(row)
                for row in distribution_tuple
                if row.truth_class == "ALL"
            ],
            "completely_missed_tracks": {
                "count": completely_missed_tracks,
                "fraction_of_truth_tracks": (
                    completely_missed_tracks / len(details) if details else None
                ),
                "missed_frames": complete_miss_frames,
            },
            "short_gap_vs_long_gap_shares": shares,
            "recall_loss_mode": {
                **shares,
                "dominant_mode": dominant_mode,
            },
            "lowest_recall_class": (
                {
                    "truth_class": lowest_recall.truth_class,
                    "point_recall": lowest_recall.point_recall,
                }
                if lowest_recall is not None
                else None
            ),
        }
        return RecallLossDiagnosisResult(
            benchmark_compatible_report=benchmark_compatible_report,
            diagnostic_report=diagnostic_report,
            per_class_recall=per_class_tuple,
            track_diagnostics=details,
            miss_gap_distribution=distribution_tuple,
            report=report,
        )


_PER_CLASS_FIELDS = tuple(PerClassRecall.__dataclass_fields__)
_TRACK_FIELDS = tuple(TrackRecallDiagnostic.__dataclass_fields__)
_GAP_FIELDS = tuple(MissGapDistribution.__dataclass_fields__)


def _write_dataclass_csv(path: Path, fields: tuple[str, ...], rows: Sequence) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(package_to_dict(row))


def run_recall_loss_diagnosis(
    *,
    geotrax_tracks: str | Path,
    fluid_tracks: str | Path,
    output_dir: str | Path,
    frame_offset: int = 1,
    max_distance_px: float = 50.0,
) -> RecallLossDiagnosisResult:
    predicted = load_geotrax_pixel_tracks(geotrax_tracks)
    truth = load_fluid_truth_all_classes(fluid_tracks)
    result = RecallLossDiagnosis(max_distance_px=max_distance_px).evaluate(
        predicted,
        truth,
        frame_offset=frame_offset,
    )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(
        target / "per_class_recall.csv",
        _PER_CLASS_FIELDS,
        result.per_class_recall,
    )
    _write_dataclass_csv(
        target / "track_recall_diagnostics.csv",
        _TRACK_FIELDS,
        result.track_diagnostics,
    )
    _write_dataclass_csv(
        target / "miss_gap_distribution.csv",
        _GAP_FIELDS,
        result.miss_gap_distribution,
    )
    (target / "recall_loss_report.json").write_text(
        json.dumps(result.report_dict(), indent=2),
        encoding="utf-8",
    )
    return result
