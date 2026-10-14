"""Shadow maximum-cardinality gate audit; frozen P3B benchmark is untouched.

P3B minimizes total Hungarian distance before dropping >50px pairs.
When objects are crowded, a minimum-distance assignment can return fewer
in-gate matches than a maximum-cardinality assignment. This *shadow* audit
quantifies that possibility without retroactively changing baseline scores.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import hypot, isfinite
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import (
    FluidPixelTruth, PixelBenchmark, _SUPPORTED_GEOTRAX_CLASSES,
)

MAX_DIST = 50.0
CLASSES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
Match = tuple[GeoTraxPixelPoint, FluidPixelTruth, float]


def maximum_cardinality_gate_matches(
    predicted: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth],
    *, offset: int = 1, radius: float = MAX_DIST,
) -> list[Match]:
    """First maximize in-radius matches, THEN minimize pixel distance."""
    if offset != 1 or radius != MAX_DIST:
        raise ValueError("Shadow audit uses the same frozen +1 frame offset / 50px gate")
    pred_by_frame: dict[int, list[GeoTraxPixelPoint]] = defaultdict(list)
    truth_by_frame: dict[int, list[FluidPixelTruth]] = defaultdict(list)
    for p in predicted:
        if p.vehicle_class not in _SUPPORTED_GEOTRAX_CLASSES:
            continue
        if not isfinite(p.x_px) or not isfinite(p.y_px):
            raise ValueError("Non-finite predicted coordinates")
        pred_by_frame[p.frame + offset].append(p)
    for t in truth:
        if t.vehicle_class not in _SUPPORTED_GEOTRAX_CLASSES:
            continue
        if not isfinite(t.x_px) or not isfinite(t.y_px):
            raise ValueError("Non-finite FLUID truth coordinates")
        truth_by_frame[t.frame].append(t)
    matches: list[Match] = []
    for frame in sorted(pred_by_frame.keys() & truth_by_frame.keys()):
        ps, ts = pred_by_frame[frame], truth_by_frame[frame]
        pxy = np.asarray([(p.x_px, p.y_px) for p in ps], dtype=float)
        txy = np.asarray([(t.x_px, t.y_px) for t in ts], dtype=float)
        dist = np.sqrt(((pxy[:, None, :] - txy[None, :, :]) ** 2).sum(axis=2))
        n, m = dist.shape
        # A single lost in-gate match must cost more than the maximum sum of
        # changes to all other real-real distances, even with crowded frames.
        penalty = (n + m + 1) * (radius + 1.0)
        impossible = (n + m + 1) * penalty * 4
        costs = np.full((n + m, n + m), float(impossible))
        costs[:n, :m] = np.where(dist <= radius, dist, impossible)
        costs[:n, m:] = penalty
        costs[n:, :m] = penalty
        costs[n:, m:] = 0.
        ri, ci = linear_sum_assignment(costs)
        for i, j in zip(ri, ci):
            if i < n and j < m and dist[i, j] <= radius:
                matches.append((ps[int(i)], ts[int(j)], float(dist[i, j])))
    return matches


def _compare_engine(
    predicted: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth], *,
    start_frame: int, end_frame: int,
) -> dict[str, Any]:
    actual_pred = [p for p in predicted
                   if start_frame <= p.frame <= end_frame
                   and p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES]
    actual_truth = [t for t in truth
                    if start_frame + 1 <= t.frame <= end_frame + 1
                    and t.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES]
    for seq in (actual_pred, actual_truth):
        if any(not isfinite(p.x_px) or not isfinite(p.y_px) for p in seq):
            raise ValueError("Non-finite spatial coordinates")
    legacy = PixelBenchmark(MAX_DIST).match_points(actual_pred, actual_truth, frame_offset=1)
    cardinality = maximum_cardinality_gate_matches(actual_pred, actual_truth)
    if len(cardinality) < len(legacy):
        raise AssertionError("Maximum-cardinality shadow cannot produce fewer in-gate matches")

    def key_of(match: Match) -> tuple[int, str]:
        return match[1].frame, match[1].track_id
    legacy_by_truth = {key_of(m): m for m in legacy}
    optimal_by_truth = {key_of(m): m for m in cardinality}
    if len(legacy_by_truth) != len(legacy) or len(optimal_by_truth) != len(cardinality):
        raise ValueError("Duplicate FLUID truth identities within a frame")
    new_truth = [m for k, m in optimal_by_truth.items() if k not in legacy_by_truth]
    lost_truth = [m for k, m in legacy_by_truth.items() if k not in optimal_by_truth]
    original_matched_ids = {id(p) for p, _, _ in legacy}
    optimal_matched_ids = {id(p) for p, _, _ in cardinality}
    rescued_by_class = Counter(t.vehicle_class for _, t, _ in new_truth)
    lost_by_class = Counter(t.vehicle_class for _, t, _ in lost_truth)
    rescued_high_car = sum(
        p.vehicle_class == "CAR" and p.confidence is not None and p.confidence >= .5
        and id(p) not in original_matched_ids and id(p) in optimal_matched_ids
        for p in actual_pred
    )
    return {
        "truth_points": len(actual_truth),
        "predicted_points": len(actual_pred),
        "frozen_p3b_hungarian_matches": len(legacy),
        "shadow_max_cardinality_matches": len(cardinality),
        "additional_in_gate_matches": len(cardinality) - len(legacy),
        "frozen_p3b_recall": len(legacy) / len(actual_truth) if actual_truth else None,
        "shadow_recall_NOT_comparable_to_historical_p3b": (
            len(cardinality) / len(actual_truth) if actual_truth else None),
        "frozen_p3b_unmatched_predictions": len(actual_pred) - len(legacy),
        "shadow_unmatched_predictions": len(actual_pred) - len(cardinality),
        "newly_matched_truth_by_class": {cls: rescued_by_class[cls] for cls in CLASSES},
        "truth_that_lost_legacy_match_by_class": {cls: lost_by_class[cls] for cls in CLASSES},
        "previously_unmatched_high_confidence_car_predictions_now_assigned": rescued_high_car,
        "newly_matched_truth_examples": [
            {"video_frame": t.frame - 1, "fluid_track_id": t.track_id,
             "truth_class": t.vehicle_class, "predicted_class": p.vehicle_class,
             "pixel_distance": distance}
            for p, t, distance in sorted(new_truth, key=lambda m: (m[1].frame, m[1].track_id))
        ],
    }


def shadow_matching_audit(
    standard: Sequence[GeoTraxPixelPoint],
    sliced: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth], *,
    start_frame: int, end_frame: int,
) -> dict[str, Any]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid video frame interval")
    return {
        "status": "SHADOW_MATCHER_DIAGNOSTIC_FROZEN_BENCHMARK_UNCHANGED",
        "eligible_for_promotion": False,
        "video_frame_range": [start_frame, end_frame],
        "frozen_spatial_radius_px": MAX_DIST,
        "frozen_frame_offset": 1,
        "important": (
            "The original P3B minimum-distance Hungarian matcher remains the "
            "source of official scores. This shadow algorithm maximizes in-gate "
            "match count before reducing distance. New matches may involve "
            "different identities or classes and require visual review. "
            "Do not mix shadow metrics into original historical scorecards."
        ),
        "standard": _compare_engine(standard, truth,
                                    start_frame=start_frame, end_frame=end_frame),
        "sliced": _compare_engine(sliced, truth,
                                  start_frame=start_frame, end_frame=end_frame),
    }
