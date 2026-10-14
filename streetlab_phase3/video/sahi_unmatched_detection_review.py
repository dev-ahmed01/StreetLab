"""Ground-truth-aware READ-ONLY precision review of cached SAHI detections.

Class-aware max-cardinality association matches the frozen *detector-only*
score, NOT the class-agnostic P3B tracking matcher. Every flagged prediction
is 'unmatched against sampled FLUID labels', never a confirmed physical FP.
No suppression thresholds, model inference, tracker execution or source edits.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from streetlab_phase3.pixel_benchmark import FluidPixelTruth, normalize_fluid_pixel_truth
from streetlab_phase3.video.cached_detector_subset import _detections, _read_audit
from streetlab_phase3.video.sahi_detection_audit import (
    CLASSES, Detection, score_detections,
)

RADIUS = 50.0
OFFSET = 1
NEAR = 25.0


def accepted_prediction_indices(
    predictions: Sequence[Detection],
    truths: Sequence[FluidPixelTruth],
) -> set[int]:
    """Reproduce sahi_detection_audit._accepted_matches's padded assignment."""
    if not predictions or not truths:
        return set()
    p = np.asarray([(d.x_px, d.y_px) for d in predictions], dtype=float)
    t = np.asarray([(d.x_px, d.y_px) for d in truths], dtype=float)
    dist = np.sqrt(((p[:, None, :] - t[None, :, :]) ** 2).sum(axis=2))
    n, m = dist.shape
    penalty = RADIUS + 1.
    impossible = (n + m + 1) * penalty * 4.
    cost = np.full((n + m, n + m), impossible, dtype=float)
    cost[:n, :m] = np.where(dist <= RADIUS, dist, impossible)
    cost[:n, m:] = penalty
    cost[n:, :m] = penalty
    cost[n:, m:] = 0.
    ii, jj = linear_sum_assignment(cost)
    return {int(i) for i, j in zip(ii, jj)
            if i < n and j < m and dist[i, j] <= RADIUS}


def nearest_distance(point: Detection,
                     others: Sequence[Detection | FluidPixelTruth]) -> float | None:
    if not others:
        return None
    return min(math.hypot(point.x_px - p.x_px, point.y_px - p.y_px)
               for p in others)


def audit_unmatched_detections(
    directory: Path, *, parity_file: Path | None = None,
    review_limit: int = 12,
) -> dict[str, Any]:
    if review_limit < 1 or review_limit > 100:
        raise ValueError("Review limit must be 1..100")
    source = _read_audit(directory)
    if source.get("runtime_backend", "pytorch") != "openvino":
        raise ValueError("This W04 precision review expects SHA-verified OpenVINO")
    frames = source["sample_frames"]
    with Path(source["trial"]["fluid_tracks"]).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        labels = normalize_fluid_pixel_truth(csv.DictReader(stream))
    predictions = _detections(directory, "sliced", frames)
    scored = score_detections(predictions, labels, frames)
    for klass in CLASSES:
        for key in ("truth", "predicted", "matched"):
            if (scored["per_class"][klass][key]
                != source["sliced"]["per_class"][klass][key]):
                raise ValueError(f"Cached sliced report differs: {klass}.{key}")
    for key in ("truth_points", "predicted_points", "matched_points"):
        if scored[key] != source["sliced"][key]:
            raise ValueError(f"Cached overall score differs: {key}")

    truth_by_frame: dict[int, list[FluidPixelTruth]] = defaultdict(list)
    selected = {frame + OFFSET for frame in frames}
    for row in labels:
        if row.frame in selected and row.vehicle_class in CLASSES:
            truth_by_frame[row.frame].append(row)

    rows: list[dict[str, Any]] = []
    by_class: dict[str, Any] = {}
    for klass in CLASSES:
        relevant: list[dict[str, Any]] = []
        for frame in frames:
            all_predictions = predictions[frame]
            choices = [d for d in all_predictions if d.vehicle_class == klass]
            truths = [t for t in truth_by_frame[frame + OFFSET]
                      if t.vehicle_class == klass]
            used = accepted_prediction_indices(choices, truths)
            if len(used) != len(set(used)):
                raise AssertionError("Duplicate matched detection index")
            kept = [choices[i] for i in sorted(used)]
            other_truth = [t for t in truth_by_frame[frame + OFFSET]
                           if t.vehicle_class != klass]
            for i, p in enumerate(choices):
                if i in used:
                    continue
                same_gt = nearest_distance(p, truths)
                other_gt = nearest_distance(p, other_truth)
                matched_pred = nearest_distance(p, kept)
                another_pred = nearest_distance(
                    p, [x for j, x in enumerate(choices) if j != i])
                item = {
                    "video_frame": frame,
                    "class": klass,
                    "x_px": p.x_px,
                    "y_px": p.y_px,
                    "confidence": p.confidence,
                    "nearest_same_class_truth_px": same_gt,
                    "nearest_other_class_truth_px": other_gt,
                    "nearest_matched_same_class_prediction_px": matched_pred,
                    "nearest_any_same_class_prediction_px": another_pred,
                    "near_matched_same_class_prediction_25px":
                        matched_pred is not None and matched_pred <= NEAR,
                    "near_any_same_class_truth_50px":
                        same_gt is not None and same_gt <= RADIUS,
                    "near_other_class_truth_50px":
                        other_gt is not None and other_gt <= RADIUS,
                }
                relevant.append(item)
                rows.append(item)
        expected = scored["per_class"][klass]["predicted"] - scored["per_class"][klass]["matched"]
        if len(relevant) != expected:
            raise AssertionError(f"{klass}: unmatched rows differ from cached scoring")
        bands = {
            "[0,.25)": sum(p["confidence"] < .25 for p in relevant),
            "[.25,.5)": sum(.25 <= p["confidence"] < .5 for p in relevant),
            "[.5,1]": sum(.5 <= p["confidence"] <= 1 for p in relevant),
        }
        by_class[klass] = {
            "truth": scored["per_class"][klass]["truth"],
            "matched": scored["per_class"][klass]["matched"],
            "unmatched": len(relevant),
            "confidence_bands": bands,
            "near_matched_same_class_prediction_25px": sum(
                p["near_matched_same_class_prediction_25px"] for p in relevant),
            "near_any_same_class_truth_50px": sum(
                p["near_any_same_class_truth_50px"] for p in relevant),
            "near_other_class_truth_50px": sum(
                p["near_other_class_truth_50px"] for p in relevant),
        }

    discordances: list[dict[str, Any]] = []
    if parity_file is not None:
        report = json.loads(parity_file.read_text(encoding="utf-8"))
        if (report.get("status") != "PAIRED_DETECTOR_TRUTH_OBSERVATION_AUDIT_NOT_TRACKING"
            or report.get("samples") != frames
            or report.get("same_checkpoint_sha256") != source["model_sha256"]
            or report.get("candidate_backend") != "openvino"):
            raise ValueError("Paired FLUID truth parity provenance does not match OpenVINO audit")
        for cls in CLASSES:
            for obs in report["by_class"][cls]["annotated_vehicle_observation_disagreements"]:
                frame, tid = obs["video_frame"], obs["fluid_track_id"]
                matched_truth = [
                    t for t in truth_by_frame[frame + OFFSET]
                    if t.vehicle_class == cls and t.track_id == tid]
                if len(matched_truth) != 1:
                    raise ValueError(f"Parity truth ID not found uniquely: {frame}, {tid}")
                t = matched_truth[0]
                discordances.append({
                    "review_reason": "pytorch_openvino_truth_observation_disagreement",
                    "source_frame": frame, "class": cls,
                    "fluid_track_id": tid,
                    "annotation_x_px": t.x_px,
                    "annotation_y_px": t.y_px,
                    "found_by": obs["found_by"],
                })
        if len(discordances) != report["total_annotation_observation_disagreements"]:
            raise ValueError("Parity disagreement count does not match records")

    # Manual inspection prioritization, NOT an executable suppression rule.
    # Distinct frame/class/coords avoid repeat review of the same observation.
    suggestions: list[dict[str, Any]] = list(discordances)
    def add(class_name: str, count: int, reason: str,
            *, near: bool | None = None) -> None:
        chosen = [r for r in rows if r["class"] == class_name
                  and (near is None or
                       r["near_matched_same_class_prediction_25px"] is near)]
        chosen.sort(key=lambda p: (-p["confidence"], p["video_frame"],
                                   p["x_px"], p["y_px"]))
        for p in chosen[:count]:
            suggestions.append({"review_reason": reason, **p})
    add("MOTORCYCLE", 3, "unmatched_motorcycle_near_matched", near=True)
    add("MOTORCYCLE", 2, "high_confidence_unmatched_motorcycle")
    add("CAR", 3, "high_confidence_unmatched_car")
    add("HEAVY_VEHICLE", 2, "unmatched_heavy_vehicle")
    add("BUS", 2, "unmatched_bus")
    unique = []
    seen = set()
    for item in suggestions:
        identity = (item.get("review_reason") == "pytorch_openvino_truth_observation_disagreement",
                    item["source_frame"] if "source_frame" in item else item["video_frame"],
                    item["class"], item.get("fluid_track_id"),
                    item.get("x_px"), item.get("y_px"))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)

    total_unmatched = scored["predicted_points"] - scored["matched_points"]
    if len(rows) != total_unmatched:
        raise AssertionError("Unexpected unmatched predictions count")
    return {
        "status": "EXPLORATORY_DETECTOR_UNMATCHED_REVIEW_NOT_FALSE_POSITIVE_VERDICT",
        "eligible_for_promotion": False,
        "prediction_source": str(directory / "sliced_detections.csv"),
        "frames": frames,
        "annotated_truth_points": scored["truth_points"],
        "matched_predictions": scored["matched_points"],
        "unmatched_predictions": total_unmatched,
        "classes": by_class,
        "parity_disagreements": discordances,
        "review_queue": unique[:max(review_limit, len(discordances))],
        "unmatched_rows": rows,
        "limitations": (
            "All unmatched items are class-aware 50px detector-only evaluator "
            "unmatched *observations*, NOT verified physical false positives. "
            "FLUID may omit objects or classify overlapping objects differently. "
            "Nearest-center distances cannot prove box IoU or duplication; "
            "nearby real motorcycles must not be suppressed automatically. "
            "Review queue uses annotations ONLY to select manual inspection "
            "examples. It never changes YOLO thresholds, exported results, "
            "NMS, ByteTrack or the frozen P3B scoring algorithm. "
            "May-26 is tuned footage, not a holdout."
        ),
    }
