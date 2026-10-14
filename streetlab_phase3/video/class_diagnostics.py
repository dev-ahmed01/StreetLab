"""Fixed-cohort per-class diagnosis for P3B tracking, no inference required.

Uses the SAME spatial Hungarian association as PixelBenchmark and preserves
class disagreement as a separate error instead of inflating false positives.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence, Any

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint, load_geotrax_pixel_tracks
from streetlab_phase3.pixel_benchmark import (
    FluidPixelTruth, PixelBenchmark, normalize_fluid_pixel_truth
)

SUPPORTED = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")


def class_diagnostics(
    predicted: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth], *,
    start_frame: int, end_frame: int,
    frame_offset: int = 1, max_pixel_distance: float = 50.0,
) -> dict[str, Any]:
    """Predeclare inclusive VIDEO frames; label frames use frozen +1 offset."""
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid evaluation frame interval")
    if frame_offset != 1 or max_pixel_distance != 50.0:
        raise ValueError("Frozen FLUID +1 offset and 50px spatial gate required")

    truth_window = [
        p for p in truth if start_frame + frame_offset <= p.frame
        <= end_frame + frame_offset and p.vehicle_class in SUPPORTED
    ]
    pred_window = [
        p for p in predicted if start_frame <= p.frame <= end_frame
        and p.vehicle_class in SUPPORTED
    ]
    matches = PixelBenchmark(max_pixel_distance).match_points(
        pred_window, truth_window, frame_offset=frame_offset)

    truth_counts = Counter(x.vehicle_class for x in truth_window)
    predicted_counts = Counter(x.vehicle_class for x in pred_window)
    spatial_truth_matches = Counter(t.vehicle_class for _, t, _ in matches)
    spatial_prediction_matches = Counter(p.vehicle_class for p, _, _ in matches)
    correct_truth_matches = Counter(t.vehicle_class for p, t, _ in matches
                                    if p.vehicle_class == t.vehicle_class)
    confusions = Counter((t.vehicle_class, p.vehicle_class)
                         for p, t, _ in matches if p.vehicle_class != t.vehicle_class)

    by_class: dict[str, dict[str, int | float | None]] = {}
    for category in SUPPORTED:
        gt = truth_counts[category]
        pred = predicted_counts[category]
        aligned = spatial_truth_matches[category]
        correct = correct_truth_matches[category]
        # A spatially matched prediction of the wrong class is NOT a spatial FP.
        matched_predictions = spatial_prediction_matches[category]
        by_class[category] = {
            "truth_points": gt,
            "predicted_points": pred,
            "spatial_matches": aligned,
            "correct_class_matches": correct,
            "spatial_misses": gt - aligned,
            "class_errors_on_spatial_matches": aligned - correct,
            "unmatched_predictions": pred - matched_predictions,
            "spatial_recall": aligned / gt if gt else None,
            "correct_class_recall": correct / gt if gt else None,
            "spatial_precision": matched_predictions / pred if pred else None,
            "correct_class_precision": correct / pred if pred else None,
        }
    return {
        "status": "FIXED_WINDOW_DIAGNOSTIC_NOT_PROMOTION",
        "evaluation_video_frame_start": start_frame,
        "evaluation_video_frame_end": end_frame,
        "evaluation_fluid_frame_start": start_frame + frame_offset,
        "evaluation_fluid_frame_end": end_frame + frame_offset,
        "frame_offset": frame_offset,
        "max_pixel_distance": max_pixel_distance,
        "truth_points": len(truth_window),
        "predicted_points": len(pred_window),
        "matched_points": len(matches),
        "correct_class_matches": sum(correct_truth_matches.values()),
        "by_class": by_class,
        "confusion_truth_to_predicted": [
            {"truth": t, "predicted": p, "points": count}
            for (t, p), count in sorted(confusions.items())
        ],
    }


def diagnose_files(tracks: Path, fluid_tracks: Path, *,
                   start_frame: int, end_frame: int) -> dict[str, Any]:
    with fluid_tracks.open(newline="", encoding="utf-8-sig") as handle:
        truth = normalize_fluid_pixel_truth(csv.DictReader(handle))
    predicted = load_geotrax_pixel_tracks(tracks)
    payload = class_diagnostics(predicted, truth,
                                start_frame=start_frame, end_frame=end_frame)
    payload["track_file"] = str(tracks)
    payload["fluid_tracks"] = str(fluid_tracks)
    return payload
