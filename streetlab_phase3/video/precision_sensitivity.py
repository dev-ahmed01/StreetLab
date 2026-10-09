"""Offline post-track confidence and proximity diagnostics (not threshold tuning).

All scores use the unchanged FLUID +1 video-label alignment, 50px Hungarian
matching and exact original truth window. Filtering tracks AFTER association
cannot reproduce changing detector confidence or ByteTrack association.
"""
from __future__ import annotations

from collections import defaultdict
from math import hypot, isfinite
from typing import Any, Sequence

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import (
    FluidPixelTruth, PixelBenchmark, _SUPPORTED_GEOTRAX_CLASSES,
)
from streetlab_phase3.video.class_diagnostics import class_diagnostics

# Preset beforehand, before seeing per-threshold scores; descriptive only.
THRESHOLDS: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, 0.50, 0.60, 0.75)
MOTORCYCLE = "MOTORCYCLE"


def _validate(
    predictions: Sequence[GeoTraxPixelPoint],
    start_frame: int, end_frame: int,
) -> list[GeoTraxPixelPoint]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid inclusive video frame interval")
    window = [
        p for p in predictions
        if start_frame <= p.frame <= end_frame
        and p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
    ]
    for p in window:
        if p.confidence is None or not isfinite(p.confidence) or not 0 <= p.confidence <= 1:
            raise ValueError("Post-track filtering requires a finite confidence in [0,1] on every export")
    return window


def confidence_sensitivity(
    sliced: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth], *,
    start_frame: int, end_frame: int,
    thresholds: Sequence[float] = THRESHOLDS,
    standard: Sequence[GeoTraxPixelPoint] | None = None,
) -> dict[str, Any]:
    """Descriptive *post-track* re-scoring; NEVER claims tracker would behave this way."""
    if not thresholds or any(
        not isfinite(v) or not 0 < v <= 1 for v in thresholds
    ):
        raise ValueError("Thresholds must be finite nonempty values in (0,1]")
    if tuple(sorted(set(thresholds))) != tuple(thresholds):
        raise ValueError("Thresholds must be unique and ascending")
    present = _validate(sliced, start_frame, end_frame)
    target_truth = [
        p for p in truth if start_frame + 1 <= p.frame <= end_frame + 1
        and p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
    ]
    if not target_truth:
        raise ValueError("No FLUID truth points in requested frame interval")
    mc_truth = sum(p.vehicle_class == MOTORCYCLE for p in target_truth)
    if mc_truth == 0:
        raise ValueError("Motorcycle truth observations required for this audit")
    if standard is not None:
        _validate(standard, start_frame, end_frame)

    results: list[dict[str, Any]] = []
    for policy in ("global_post_track_cutoff", "motorcycle_only_post_track_cutoff"):
        for threshold in thresholds:
            filtered = [
                p for p in present
                if p.confidence is not None and (
                    p.confidence >= threshold
                    if policy == "global_post_track_cutoff"
                    or p.vehicle_class == MOTORCYCLE
                    else True
                )
            ]
            metric = PixelBenchmark(50.).evaluate(
                filtered, target_truth, frame_offsets=(1,),
                evaluation_frame_range=(start_frame + 1, end_frame + 1))
            by_class = class_diagnostics(
                filtered, target_truth,
                start_frame=start_frame, end_frame=end_frame)["by_class"]
            motorcycle = by_class[MOTORCYCLE]
            results.append({
                "policy": policy,
                "confidence_cutoff_inclusive": threshold,
                "truth_points": metric.truth_points,
                "motorcycle_truth_points": mc_truth,
                "predictions_kept": metric.predicted_points,
                "predictions_removed": len(present) - metric.predicted_points,
                "matched_points": metric.matched_points,
                "unmatched_predictions": metric.predicted_points - metric.matched_points,
                "overall_point_recall": metric.point_recall,
                "overall_point_precision": metric.point_precision,
                "motorcycle_spatial_recall": motorcycle["spatial_recall"],
                "motorcycle_correct_class_recall": motorcycle["correct_class_recall"],
                "motorcycle_correct_class_matches": motorcycle["correct_class_matches"],
                "motorcycle_unmatched_predictions": motorcycle["unmatched_predictions"],
            })

    return {
        "status": "POST_TRACK_FILTER_COUNTERFACTUAL_NOT_PROMOTION",
        "eligible_for_promotion": False,
        "video_frame_start": start_frame,
        "video_frame_end": end_frame,
        "thresholds_predeclared": list(thresholds),
        "frozen_fluid_frame_offset": 1,
        "frozen_spatial_gate_px": 50.,
        "note": (
            "A confidence filter applied to previously confirmed tracks cannot "
            "recreate detection NMS, low-confidence ByteTrack association, "
            "track activation or recovered tracks. These are only diagnostic "
            "post-track scores on reused May-26 tuning footage; do not choose "
            "a production cutoff from three frames."
        ),
        "results": results,
    }


def unmatched_proximity_audit(
    sliced: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth], *,
    start_frame: int, end_frame: int,
    proximity_px: float = 25.,
) -> dict[str, Any]:
    """Examine unmatched detections near existing annotations/other matches.

    Candidate duplicate = close centroid to a separately matched prediction.
    This is NOT bounding-box IoU or proof of duplicate detections.
    """
    if not isfinite(proximity_px) or proximity_px <= 0 or proximity_px > 50:
        raise ValueError("Proximity radius must be in (0,50] px")
    present = _validate(sliced, start_frame, end_frame)
    truth_window = [
        p for p in truth if start_frame + 1 <= p.frame <= end_frame + 1
        and p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
    ]
    matches = PixelBenchmark(50.).match_points(
        present, truth_window, frame_offset=1)
    matched_pred_ids = {id(p) for p, _, _ in matches}
    matched_by_frame: dict[int, list[GeoTraxPixelPoint]] = defaultdict(list)
    truth_by_frame: dict[int, list[FluidPixelTruth]] = defaultdict(list)
    for p, _, _ in matches:
        matched_by_frame[p.frame].append(p)
    for t in truth_window:
        truth_by_frame[t.frame - 1].append(t)
    candidates: list[dict[str, Any]] = []
    for p in present:
        if id(p) in matched_pred_ids:
            continue
        near_pred = (
            min((hypot(p.x_px - q.x_px, p.y_px - q.y_px)
                 for q in matched_by_frame[p.frame]), default=None)
        )
        near_truth = (
            min((hypot(p.x_px - q.x_px, p.y_px - q.y_px)
                 for q in truth_by_frame[p.frame]), default=None)
        )
        candidates.append({
            "video_frame": p.frame,
            "predicted_track_id": p.track_id,
            "predicted_class": p.vehicle_class,
            "confidence": p.confidence,
            "pixel_center": [p.x_px, p.y_px],
            "nearest_matched_prediction_px": near_pred,
            "nearest_truth_point_px": near_truth,
            "near_matched_prediction": (
                near_pred is not None and near_pred <= proximity_px),
            "near_truth_point": (
                near_truth is not None and near_truth <= 50.),
        })
    by_class = {}
    for cls in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE"):
        group = [x for x in candidates if x["predicted_class"] == cls]
        by_class[cls] = {
            "unmatched": len(group),
            "near_matched_prediction_count": sum(
                x["near_matched_prediction"] for x in group),
            "near_any_truth_within_50px": sum(
                x["near_truth_point"] for x in group),
            "high_confidence_at_least_0_5": sum(
                x["confidence"] >= .5 for x in group),
            "high_confidence_near_matched_prediction": sum(
                x["confidence"] >= .5 and x["near_matched_prediction"] for x in group),
        }
    return {
        "status": "UNMATCHED_PROXIMITY_HEURISTIC_NOT_PROOF_OF_DUPLICATES",
        "eligible_for_promotion": False,
        "video_frame_start": start_frame,
        "video_frame_end": end_frame,
        "near_other_match_centroid_radius_px": proximity_px,
        "truth_search_radius_px": 50.,
        "unmatched_total": len(candidates),
        "by_class": by_class,
        "examples": candidates,
        "note": (
            "Nearby unmatched centers can be overlapping vehicle boxes, separate "
            "vehicles, class ambiguity or incomplete FLUID annotations. There "
            "is no raw detection box IoU or independent visual verification."
        ),
    }
