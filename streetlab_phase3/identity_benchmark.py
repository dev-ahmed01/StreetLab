from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from .geotrax_pixel import GeoTraxPixelPoint, load_geotrax_pixel_tracks
from .pixel_benchmark import (
    FluidPixelTruth,
    PixelBenchmark,
    _SUPPORTED_GEOTRAX_CLASSES,
    normalize_fluid_pixel_truth,
)
from .serialization import package_to_dict


@dataclass(frozen=True)
class IdentityScorecard:
    frame_offset: int
    max_distance_px: float
    max_switch_gap_frames: int
    truth_tracks: int
    predicted_tracks: int
    matched_truth_tracks: int
    matched_predicted_tracks: int
    unmatched_truth_tracks: int
    unmatched_predicted_tracks: int
    unmatched_predicted_track_fraction: float | None
    mean_predicted_ids_per_truth_track: float | None
    median_predicted_ids_per_truth_track: float | None
    max_predicted_ids_per_truth_track: int
    truth_tracks_with_exactly_1_predicted_id: int
    truth_tracks_with_2_predicted_ids: int
    truth_tracks_with_3plus_predicted_ids: int
    fraction_truth_tracks_unfragmented: float | None
    fraction_truth_tracks_fragmented: float | None
    total_truth_fragmentation_excess: int
    total_raw_id_switches: int
    total_contiguous_id_switches: int
    mean_contiguous_id_switches_per_matched_truth_track: float | None
    truth_tracks_with_zero_contiguous_switches: int
    fraction_truth_tracks_with_zero_contiguous_switches: float | None
    mean_truth_track_matched_fraction: float | None
    median_truth_track_matched_fraction: float | None
    mean_longest_run_fraction: float | None
    median_longest_run_fraction: float | None
    mean_dominant_pair_fraction: float | None
    median_dominant_pair_fraction: float | None


@dataclass(frozen=True)
class TruthTrackIdentity:
    truth_track_id: str
    truth_class: str
    truth_frame_count: int
    matched_frame_count: int
    matched_fraction: float
    associated_predicted_ids: tuple[str, ...]
    associated_predicted_id_count: int
    fragmentation_excess: int
    raw_id_switches: int
    contiguous_id_switches: int
    dominant_predicted_id: str | None
    dominant_pair_matched_frames: int
    dominant_pair_fraction: float | None
    longest_consecutive_match_run: int
    longest_run_fraction: float


@dataclass(frozen=True)
class IdentityPair:
    truth_track_id: str
    predicted_track_id: str
    matched_frames: int
    first_matched_frame: int
    last_matched_frame: int
    mean_pixel_error: float


@dataclass(frozen=True)
class IdentityBenchmarkResult:
    scorecard: IdentityScorecard
    truth_track_details: tuple[TruthTrackIdentity, ...]
    pair_details: tuple[IdentityPair, ...]

    def scorecard_dict(self) -> dict:
        return package_to_dict(self.scorecard)


def _optional_mean(values: Sequence[float | int]) -> float | None:
    return float(mean(values)) if values else None


def _optional_median(values: Sequence[float | int]) -> float | None:
    return float(median(values)) if values else None


def _longest_consecutive_run(frames: set[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for frame in sorted(frames):
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


class IdentityBenchmark:
    def __init__(
        self,
        max_distance_px: float = 50.0,
        max_switch_gap_frames: int = 5,
    ) -> None:
        if max_switch_gap_frames <= 0:
            raise ValueError("max_switch_gap_frames must be positive")
        self.pixel_benchmark = PixelBenchmark(max_distance_px=max_distance_px)
        self.max_switch_gap_frames = int(max_switch_gap_frames)

    def evaluate(
        self,
        predicted: Sequence[GeoTraxPixelPoint],
        truth: Sequence[FluidPixelTruth],
        *,
        frame_offsets: Sequence[int] = (-2, -1, 0, 1, 2),
    ) -> IdentityBenchmarkResult:
        pixel_report = self.pixel_benchmark.evaluate(
            predicted,
            truth,
            frame_offsets=frame_offsets,
        )
        matches = self.pixel_benchmark.match_points(
            predicted,
            truth,
            frame_offset=pixel_report.frame_offset,
        )
        predicted_supported = [
            point
            for point in predicted
            if point.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
        ]

        if (
            pixel_report.evaluation_frame_start is None
            or pixel_report.evaluation_frame_end is None
        ):
            truth_window: list[FluidPixelTruth] = []
        else:
            truth_window = [
                point
                for point in truth
                if pixel_report.evaluation_frame_start
                <= point.frame
                <= pixel_report.evaluation_frame_end
            ]

        truth_by_track: dict[str, list[FluidPixelTruth]] = defaultdict(list)
        for point in truth_window:
            truth_by_track[point.track_id].append(point)

        matches_by_truth: dict[
            str, list[tuple[GeoTraxPixelPoint, FluidPixelTruth, float]]
        ] = defaultdict(list)
        pair_matches: dict[
            tuple[str, str], list[tuple[GeoTraxPixelPoint, FluidPixelTruth, float]]
        ] = defaultdict(list)
        for match in matches:
            predicted_point, truth_point, _ = match
            matches_by_truth[truth_point.track_id].append(match)
            pair_matches[(truth_point.track_id, predicted_point.track_id)].append(match)

        pair_details = tuple(
            IdentityPair(
                truth_track_id=truth_track_id,
                predicted_track_id=predicted_track_id,
                matched_frames=len(pair),
                first_matched_frame=min(match[1].frame for match in pair),
                last_matched_frame=max(match[1].frame for match in pair),
                mean_pixel_error=float(mean(match[2] for match in pair)),
            )
            for (truth_track_id, predicted_track_id), pair in sorted(
                pair_matches.items()
            )
        )
        pair_by_ids = {
            (pair.truth_track_id, pair.predicted_track_id): pair
            for pair in pair_details
        }

        truth_track_details: list[TruthTrackIdentity] = []
        for truth_track_id, truth_points in sorted(truth_by_track.items()):
            truth_frames = {point.frame for point in truth_points}
            track_matches = sorted(
                matches_by_truth.get(truth_track_id, []),
                key=lambda match: (match[1].frame, match[0].track_id),
            )
            matched_frames = {match[1].frame for match in track_matches}
            associated_ids = tuple(sorted({match[0].track_id for match in track_matches}))

            raw_switches = 0
            contiguous_switches = 0
            for previous, current in zip(track_matches, track_matches[1:]):
                if previous[0].track_id == current[0].track_id:
                    continue
                raw_switches += 1
                if current[1].frame - previous[1].frame <= self.max_switch_gap_frames:
                    contiguous_switches += 1

            dominant_predicted_id: str | None = None
            dominant_pair_matched_frames = 0
            dominant_pair_fraction: float | None = None
            if associated_ids:
                candidate_pairs = [
                    pair_by_ids[(truth_track_id, predicted_track_id)]
                    for predicted_track_id in associated_ids
                ]
                dominant_pair = min(
                    candidate_pairs,
                    key=lambda pair: (
                        -pair.matched_frames,
                        pair.first_matched_frame,
                        pair.predicted_track_id,
                    ),
                )
                dominant_predicted_id = dominant_pair.predicted_track_id
                dominant_pair_matched_frames = dominant_pair.matched_frames
                dominant_pair_fraction = (
                    dominant_pair_matched_frames / len(matched_frames)
                )

            truth_frame_count = len(truth_frames)
            matched_frame_count = len(matched_frames)
            longest_run = _longest_consecutive_run(matched_frames)
            truth_track_details.append(
                TruthTrackIdentity(
                    truth_track_id=truth_track_id,
                    truth_class=min(truth_points, key=lambda point: point.frame).vehicle_class,
                    truth_frame_count=truth_frame_count,
                    matched_frame_count=matched_frame_count,
                    matched_fraction=matched_frame_count / truth_frame_count,
                    associated_predicted_ids=associated_ids,
                    associated_predicted_id_count=len(associated_ids),
                    fragmentation_excess=max(0, len(associated_ids) - 1),
                    raw_id_switches=raw_switches,
                    contiguous_id_switches=contiguous_switches,
                    dominant_predicted_id=dominant_predicted_id,
                    dominant_pair_matched_frames=dominant_pair_matched_frames,
                    dominant_pair_fraction=dominant_pair_fraction,
                    longest_consecutive_match_run=longest_run,
                    longest_run_fraction=longest_run / truth_frame_count,
                )
            )

        details = tuple(truth_track_details)
        matched_details = tuple(
            detail for detail in details if detail.matched_frame_count > 0
        )
        associated_counts = [
            detail.associated_predicted_id_count for detail in matched_details
        ]
        matched_truth_count = len(matched_details)
        matched_predicted_ids = {match[0].track_id for match in matches}
        predicted_ids = {point.track_id for point in predicted_supported}
        fragmented_count = sum(count >= 2 for count in associated_counts)
        unfragmented_count = sum(count == 1 for count in associated_counts)
        zero_switch_count = sum(
            detail.contiguous_id_switches == 0 for detail in matched_details
        )
        dominant_fractions = [
            detail.dominant_pair_fraction
            for detail in matched_details
            if detail.dominant_pair_fraction is not None
        ]

        scorecard = IdentityScorecard(
            frame_offset=pixel_report.frame_offset,
            max_distance_px=pixel_report.max_distance_px,
            max_switch_gap_frames=self.max_switch_gap_frames,
            truth_tracks=len(details),
            predicted_tracks=len(predicted_ids),
            matched_truth_tracks=matched_truth_count,
            matched_predicted_tracks=len(matched_predicted_ids),
            unmatched_truth_tracks=len(details) - matched_truth_count,
            unmatched_predicted_tracks=len(predicted_ids) - len(matched_predicted_ids),
            unmatched_predicted_track_fraction=(
                (len(predicted_ids) - len(matched_predicted_ids)) / len(predicted_ids)
                if predicted_ids
                else None
            ),
            mean_predicted_ids_per_truth_track=_optional_mean(associated_counts),
            median_predicted_ids_per_truth_track=_optional_median(associated_counts),
            max_predicted_ids_per_truth_track=max(associated_counts, default=0),
            truth_tracks_with_exactly_1_predicted_id=unfragmented_count,
            truth_tracks_with_2_predicted_ids=sum(
                count == 2 for count in associated_counts
            ),
            truth_tracks_with_3plus_predicted_ids=sum(
                count >= 3 for count in associated_counts
            ),
            fraction_truth_tracks_unfragmented=(
                unfragmented_count / matched_truth_count
                if matched_truth_count
                else None
            ),
            fraction_truth_tracks_fragmented=(
                fragmented_count / matched_truth_count if matched_truth_count else None
            ),
            total_truth_fragmentation_excess=sum(
                detail.fragmentation_excess for detail in matched_details
            ),
            total_raw_id_switches=sum(
                detail.raw_id_switches for detail in matched_details
            ),
            total_contiguous_id_switches=sum(
                detail.contiguous_id_switches for detail in matched_details
            ),
            mean_contiguous_id_switches_per_matched_truth_track=_optional_mean(
                [detail.contiguous_id_switches for detail in matched_details]
            ),
            truth_tracks_with_zero_contiguous_switches=zero_switch_count,
            fraction_truth_tracks_with_zero_contiguous_switches=(
                zero_switch_count / matched_truth_count if matched_truth_count else None
            ),
            mean_truth_track_matched_fraction=_optional_mean(
                [detail.matched_fraction for detail in details]
            ),
            median_truth_track_matched_fraction=_optional_median(
                [detail.matched_fraction for detail in details]
            ),
            mean_longest_run_fraction=_optional_mean(
                [detail.longest_run_fraction for detail in details]
            ),
            median_longest_run_fraction=_optional_median(
                [detail.longest_run_fraction for detail in details]
            ),
            mean_dominant_pair_fraction=_optional_mean(dominant_fractions),
            median_dominant_pair_fraction=_optional_median(dominant_fractions),
        )
        return IdentityBenchmarkResult(
            scorecard=scorecard,
            truth_track_details=details,
            pair_details=pair_details,
        )


_TRUTH_TRACK_FIELDS = (
    "truth_track_id",
    "truth_class",
    "truth_frame_count",
    "matched_frame_count",
    "matched_fraction",
    "associated_predicted_ids",
    "associated_predicted_id_count",
    "fragmentation_excess",
    "raw_id_switches",
    "contiguous_id_switches",
    "dominant_predicted_id",
    "dominant_pair_matched_frames",
    "dominant_pair_fraction",
    "longest_consecutive_match_run",
    "longest_run_fraction",
)

_PAIR_FIELDS = (
    "truth_track_id",
    "predicted_track_id",
    "matched_frames",
    "first_matched_frame",
    "last_matched_frame",
    "mean_pixel_error",
)


def _read_fluid_tracks(path: str | Path) -> list[FluidPixelTruth]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return normalize_fluid_pixel_truth(csv.DictReader(fh))


def run_identity_benchmark(
    *,
    geotrax_tracks: str | Path,
    fluid_tracks: str | Path,
    output_dir: str | Path,
    max_distance_px: float = 50.0,
    max_switch_gap_frames: int = 5,
) -> IdentityBenchmarkResult:
    predicted = load_geotrax_pixel_tracks(geotrax_tracks)
    truth = _read_fluid_tracks(fluid_tracks)
    result = IdentityBenchmark(
        max_distance_px=max_distance_px,
        max_switch_gap_frames=max_switch_gap_frames,
    ).evaluate(predicted, truth)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "identity_scorecard.json").write_text(
        json.dumps(result.scorecard_dict(), indent=2),
        encoding="utf-8",
    )

    with (target / "identity_truth_tracks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRUTH_TRACK_FIELDS)
        writer.writeheader()
        for detail in result.truth_track_details:
            row = package_to_dict(detail)
            row["associated_predicted_ids"] = ";".join(
                detail.associated_predicted_ids
            )
            writer.writerow(row)

    with (target / "identity_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=_PAIR_FIELDS)
        writer.writeheader()
        for pair in result.pair_details:
            writer.writerow(package_to_dict(pair))

    return result
