from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .class_mapping import map_vehicle_class
from .geotrax_pixel import GeoTraxPixelPoint
from .models import VehicleClass


_SUPPORTED_GEOTRAX_CLASSES = {
    VehicleClass.CAR.value,
    VehicleClass.BUS.value,
    VehicleClass.HEAVY_VEHICLE.value,
    VehicleClass.MOTORCYCLE.value,
}


@dataclass(frozen=True, slots=True)
class FluidPixelTruth:
    frame: int
    track_id: str
    x_px: float
    y_px: float
    vehicle_class: str


@dataclass(frozen=True)
class PixelBenchmarkReport:
    frame_offset: int
    max_distance_px: float
    dataset_truth_points: int
    truth_points: int
    predicted_points: int
    matched_points: int
    dataset_truth_tracks: int
    truth_tracks: int
    predicted_tracks: int
    matched_truth_tracks: int
    evaluation_frame_start: int | None
    evaluation_frame_end: int | None
    point_recall: float | None
    point_precision: float | None
    track_coverage: float | None
    class_agreement: float | None
    pixel_mae: float | None
    pixel_rmse: float | None


def normalize_fluid_pixel_truth(rows: Iterable[dict[str, object]]) -> list[FluidPixelTruth]:
    result: list[FluidPixelTruth] = []
    for row in rows:
        if row.get("cx") in (None, "") or row.get("cy") in (None, ""):
            continue
        mapping = map_vehicle_class(row.get("type", "unknown"))
        canonical = mapping.canonical.value
        if canonical not in _SUPPORTED_GEOTRAX_CLASSES:
            continue
        result.append(
            FluidPixelTruth(
                frame=int(float(row["frame"])),
                track_id=str(row["id"]),
                x_px=float(row["cx"]),
                y_px=float(row["cy"]),
                vehicle_class=canonical,
            )
        )
    return result


class PixelBenchmark:
    def __init__(self, max_distance_px: float = 50.0) -> None:
        if max_distance_px <= 0:
            raise ValueError("max_distance_px must be positive")
        self.max_distance_px = float(max_distance_px)

    def evaluate(
        self,
        predicted: Sequence[GeoTraxPixelPoint],
        truth: Sequence[FluidPixelTruth],
        *,
        frame_offsets: Sequence[int] = (-2, -1, 0, 1, 2),
    ) -> PixelBenchmarkReport:
        if not frame_offsets:
            raise ValueError("At least one frame offset is required")

        pred_supported = [
            p for p in predicted if p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
        ]

        scored = [
            (self._match(pred_supported, truth, offset), offset)
            for offset in frame_offsets
        ]
        best_matches, best_offset = max(
            scored,
            key=lambda item: (
                len(item[0]),
                -abs(item[1]),
                -item[1],
            ),
        )

        errors = [m[2] for m in best_matches]
        class_matches = [m[0].vehicle_class == m[1].vehicle_class for m in best_matches]
        matched_truth_tracks = {m[1].track_id for m in best_matches}

        dataset_truth_tracks = {p.track_id for p in truth}
        pred_tracks = {p.track_id for p in pred_supported}

        if pred_supported:
            evaluation_frame_start = min(p.frame + best_offset for p in pred_supported)
            evaluation_frame_end = max(p.frame + best_offset for p in pred_supported)
            truth_window = [
                p
                for p in truth
                if evaluation_frame_start <= p.frame <= evaluation_frame_end
            ]
        else:
            evaluation_frame_start = None
            evaluation_frame_end = None
            truth_window = []

        truth_tracks = {p.track_id for p in truth_window}
        truth_count = len(truth_window)
        pred_count = len(pred_supported)
        matched_count = len(best_matches)

        return PixelBenchmarkReport(
            frame_offset=best_offset,
            max_distance_px=self.max_distance_px,
            dataset_truth_points=len(truth),
            truth_points=truth_count,
            predicted_points=pred_count,
            matched_points=matched_count,
            dataset_truth_tracks=len(dataset_truth_tracks),
            truth_tracks=len(truth_tracks),
            predicted_tracks=len(pred_tracks),
            matched_truth_tracks=len(matched_truth_tracks),
            evaluation_frame_start=evaluation_frame_start,
            evaluation_frame_end=evaluation_frame_end,
            point_recall=(matched_count / truth_count) if truth_count else None,
            point_precision=(matched_count / pred_count) if pred_count else None,
            track_coverage=(
                len(matched_truth_tracks) / len(truth_tracks)
                if truth_tracks
                else None
            ),
            class_agreement=(
                sum(class_matches) / matched_count if matched_count else None
            ),
            pixel_mae=(sum(errors) / matched_count if matched_count else None),
            pixel_rmse=(
                sqrt(sum(error * error for error in errors) / matched_count)
                if matched_count
                else None
            ),
        )

    def _match(
        self,
        predicted: Sequence[GeoTraxPixelPoint],
        truth: Sequence[FluidPixelTruth],
        frame_offset: int,
    ) -> list[tuple[GeoTraxPixelPoint, FluidPixelTruth, float]]:
        pred_by_frame: dict[int, list[GeoTraxPixelPoint]] = defaultdict(list)
        truth_by_frame: dict[int, list[FluidPixelTruth]] = defaultdict(list)

        for point in predicted:
            pred_by_frame[point.frame + frame_offset].append(point)
        for point in truth:
            truth_by_frame[point.frame].append(point)

        matches: list[tuple[GeoTraxPixelPoint, FluidPixelTruth, float]] = []

        for frame in pred_by_frame.keys() & truth_by_frame.keys():
            pred_frame = pred_by_frame[frame]
            truth_frame = truth_by_frame[frame]
            pred_xy = np.array([[p.x_px, p.y_px] for p in pred_frame], dtype=float)
            truth_xy = np.array([[p.x_px, p.y_px] for p in truth_frame], dtype=float)

            diff = pred_xy[:, None, :] - truth_xy[None, :, :]
            distances = np.sqrt(np.sum(diff * diff, axis=2))
            pred_idx, truth_idx = linear_sum_assignment(distances)

            for i, j in zip(pred_idx, truth_idx):
                distance = float(distances[i, j])
                if distance <= self.max_distance_px:
                    matches.append((pred_frame[int(i)], truth_frame[int(j)], distance))

        return matches
