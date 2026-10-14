"""Paired error forensics over two EXISTING track exports, not an inference benchmark.

A spatial match follows the frozen PixelBenchmark Hungarian+50px rule.
Same-frame +1 FLUID alignment is fixed, and a wrong-class match is separately
counted as a classification error. No automatic threshold optimization or
promotion is performed on tuning footage.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Sequence

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import (
    FluidPixelTruth, PixelBenchmark, _SUPPORTED_GEOTRAX_CLASSES,
)

CLASSES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
BINS = ("[0,0.25)", "[0.25,0.50)", "[0.50,1.00]", "unknown")


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if not math.isfinite(value) or not (0 <= value <= 1):
        raise ValueError("Confidence must be finite and between 0 and 1")
    return BINS[0] if value < .25 else BINS[1] if value < .5 else BINS[2]


def _one_engine(
    predictions: Sequence[GeoTraxPixelPoint],
    truth_window: Sequence[FluidPixelTruth],
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    observed = [
        p for p in predictions
        if start_frame <= p.frame <= end_frame
        and p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
    ]
    if any(p.confidence is not None
           and (not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1)
           for p in observed):
        raise ValueError("Invalid prediction confidence")
    matches = PixelBenchmark(50.).match_points(
        observed, truth_window, frame_offset=1)
    match_truth: dict[tuple[int, str], tuple[GeoTraxPixelPoint, bool]] = {}
    matched_pred_refs = set()
    for prediction, ground_truth, _ in matches:
        key = (ground_truth.frame, ground_truth.track_id)
        if key in match_truth:
            raise ValueError("One FLUID observation matched multiple predictions")
        match_truth[key] = (prediction, prediction.vehicle_class == ground_truth.vehicle_class)
        matched_pred_refs.add(id(prediction))

    # Track duration within the SHARED evaluation interval; not lifetime.
    track_frames: dict[str, set[int]] = defaultdict(set)
    for p in observed:
        track_frames[p.track_id].add(p.frame)

    unmatched_by_class: dict[str, dict[str, Any]] = {}
    for vehicle_class in CLASSES:
        unmatched = [p for p in observed if p.vehicle_class == vehicle_class
                     and id(p) not in matched_pred_refs]
        matched = [p for p in observed if p.vehicle_class == vehicle_class
                   and id(p) in matched_pred_refs]
        unmatched_by_class[vehicle_class] = {
            "unmatched_point_count": len(unmatched),
            "unmatched_confidence_bands": {
                b: sum(_confidence_bucket(p.confidence) == b for p in unmatched)
                for b in BINS
            },
            "unmatched_points_from_one_frame_ids": sum(
                len(track_frames[p.track_id]) == 1 for p in unmatched
            ),
            "unmatched_points_from_multi_frame_ids": sum(
                len(track_frames[p.track_id]) >= 2 for p in unmatched
            ),
            "unmatched_mean_confidence": (
                mean([p.confidence for p in unmatched if p.confidence is not None])
                if any(p.confidence is not None for p in unmatched) else None
            ),
            "matched_mean_confidence": (
                mean([p.confidence for p in matched if p.confidence is not None])
                if any(p.confidence is not None for p in matched) else None
            ),
        }
    return {"matches_by_truth": match_truth,
            "unmatched_predictions_by_class": unmatched_by_class}


def paired_error_audit(
    standard: Sequence[GeoTraxPixelPoint],
    sliced: Sequence[GeoTraxPixelPoint],
    truth: Sequence[FluidPixelTruth], *,
    start_frame: int, end_frame: int,
) -> dict[str, Any]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid inclusive video frame window")
    relevant_truth = [
        p for p in truth if start_frame + 1 <= p.frame <= end_frame + 1
        and p.vehicle_class in _SUPPORTED_GEOTRAX_CLASSES
    ]
    unique_keys: dict[tuple[int, str], FluidPixelTruth] = {}
    for p in relevant_truth:
        key = (p.frame, p.track_id)
        if key in unique_keys:
            raise ValueError(f"Duplicate FLUID ground-truth observation: {key}")
        unique_keys[key] = p
    a = _one_engine(standard, relevant_truth, start_frame, end_frame)
    b = _one_engine(sliced, relevant_truth, start_frame, end_frame)
    matches_a = a["matches_by_truth"]
    matches_b = b["matches_by_truth"]

    categories: dict[str, Any] = {}
    for cls in CLASSES:
        cases = [
            (key, p) for key, p in sorted(unique_keys.items())
            if p.vehicle_class == cls
        ]
        spatial = Counter()
        correct = Counter()
        rescued: list[dict[str, Any]] = []
        lost: list[dict[str, Any]] = []
        wrong_class_examples: list[dict[str, Any]] = []
        for key, _ in cases:
            in_a, in_b = key in matches_a, key in matches_b
            spatial["both" if in_a and in_b else
                    "standard_only" if in_a else
                    "sliced_only" if in_b else "neither"] += 1
            correct_a = in_a and matches_a[key][1]
            correct_b = in_b and matches_b[key][1]
            correct["both" if correct_a and correct_b else
                    "standard_only" if correct_a else
                    "sliced_only" if correct_b else "neither"] += 1
            example = {"video_frame": key[0] - 1, "fluid_track_id": key[1]}
            if not in_a and in_b:
                rescued.append(example)
            elif in_a and not in_b:
                lost.append(example)
            for engine_name, m in (("standard", matches_a), ("sliced", matches_b)):
                if key in m and not m[key][1]:
                    wrong_class_examples.append({
                        **example, "engine": engine_name,
                        "predicted_class": m[key][0].vehicle_class,
                    })
        categories[cls] = {
            "truth_points": len(cases),
            "spatial": {k: spatial[k] for k in
                        ("both", "standard_only", "sliced_only", "neither")},
            "correct_class": {k: correct[k] for k in
                             ("both", "standard_only", "sliced_only", "neither")},
            "spatially_rescued_by_sliced": rescued,
            "spatially_lost_by_sliced": lost,
            "wrong_class_spatial_matches": wrong_class_examples,
        }
        if sum(categories[cls]["spatial"].values()) != len(cases):
            raise AssertionError("Spatial match partition mismatch")

    return {
        "status": "PAIRED_ERROR_AUDIT_TUNING_ONLY_NOT_PRODUCTION_EVIDENCE",
        "eligible_for_promotion": False,
        "evaluation_start_frame": start_frame,
        "evaluation_end_frame": end_frame,
        "fluid_frame_offset": 1,
        "max_pixel_distance": 50.,
        "confidence_bands": list(BINS),
        "track_persistence_scope": "only the intersected evaluation frames",
        "note": ("Rescue/loss are paired SPATIAL outcomes per FLUID observation; "
                 "correct-class outcomes are separately reported. An unmatched "
                 "observation is not proof of physical absence. This small "
                 "May-26 tuning sample cannot select a deployed threshold."),
        "by_truth_class": categories,
        "standard_unmatched": a["unmatched_predictions_by_class"],
        "sliced_unmatched": b["unmatched_predictions_by_class"],
    }
