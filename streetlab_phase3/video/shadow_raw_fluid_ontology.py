"""Read-only W04 shadow annotation-ontology / spatial-presence audit.

Locks EVERY existing class-aware detector-only 50px match first; only then
tries max-cardinality, class-agnostic spatial pairing of leftover detections
against unused original FLUID rows (including PEDESTRIAN and other excluded
families). Those links are clues, NOT physical-presence verification,
relabels, a replacement benchmark or permission to suppress predictions.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from streetlab_phase3.class_mapping import map_vehicle_class
from streetlab_phase3.video.cached_detector_subset import _detections, _read_audit
from streetlab_phase3.video.sahi_detection_audit import CLASSES, Detection, score_detections

OFFSET = 1
RADIUS = 50.


def _pairs(a: Sequence[Any], b: Sequence[Any]) -> list[tuple[int, int, float]]:
    """Maximum in-radius cardinality, then minimum summed center distance."""
    if not a or not b:
        return []
    xy_a = np.asarray([(p.x_px, p.y_px) for p in a], dtype=float)
    xy_b = np.asarray([(p["x_px"], p["y_px"]) for p in b], dtype=float)
    dist = np.sqrt(np.sum((xy_a[:, None, :] - xy_b[None, :, :]) ** 2, axis=2))
    n, m = dist.shape
    penalty = RADIUS + 1.0
    impossible = (n + m + 1) * penalty * 4.
    cost = np.full((n + m, n + m), impossible)
    cost[:n, :m] = np.where(dist <= RADIUS, dist, impossible)
    cost[:n, m:] = penalty
    cost[n:, :m] = penalty
    cost[n:, m:] = 0.
    ii, jj = linear_sum_assignment(cost)
    return [(int(i), int(j), float(dist[i, j])) for i, j in zip(ii, jj)
            if i < n and j < m and dist[i, j] <= RADIUS]


def shadow_raw_fluid_ontology(audit_dir: Path) -> dict[str, Any]:
    report = _read_audit(audit_dir)
    if report.get("runtime_backend") != "openvino":
        raise ValueError("OpenVINO experimental report is required")
    frames = report["sample_frames"]
    prediction_by_frame = _detections(audit_dir, "sliced", frames)
    source_truth = Path(report["trial"]["fluid_tracks"])
    selected_truth_frames = {f + OFFSET for f in frames}
    fluid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unique_truth_keys = set()
    with source_truth.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"frame", "id", "cx", "cy", "type"}.issubset(reader.fieldnames or []):
            raise ValueError("Raw FLUID annotations must have frame,id,cx,cy,type")
        for raw in reader:
            frame = int(float(raw["frame"]))
            if frame not in selected_truth_frames or not raw["cx"] or not raw["cy"]:
                continue
            key = (frame, str(raw["id"]))
            if key in unique_truth_keys:
                raise ValueError(f"Duplicate FLUID truth identity in frame: {key}")
            unique_truth_keys.add(key)
            x,y = float(raw["cx"]), float(raw["cy"])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("Non-finite FLUID coordinate")
            fluid[frame].append({
                "fluid_frame": frame, "fluid_id": str(raw["id"]),
                "raw_fluid_type": raw["type"],
                "mapped_class": map_vehicle_class(raw["type"]).canonical.value,
                "x_px": x, "y_px": y,
            })

    # Re-evaluate with the exact original truth denominator. The official
    # evaluator only supports the four detector classes.
    from streetlab_phase3.pixel_benchmark import FluidPixelTruth
    truth_for_evaluator = [
        FluidPixelTruth(r["fluid_frame"], r["fluid_id"],
                        r["x_px"], r["y_px"], r["mapped_class"])
        for frame in selected_truth_frames for r in fluid[frame]
        if r["mapped_class"] in CLASSES
    ]
    scored = score_detections(prediction_by_frame, truth_for_evaluator, frames)
    for name in ("truth_points", "predicted_points", "matched_points"):
        if scored[name] != report["sliced"][name]:
            raise ValueError(f"Frozen OpenVINO scoring mismatch: {name}")
    for klass in CLASSES:
        for name in ("truth","predicted","matched"):
            if scored["per_class"][klass][name] != report["sliced"]["per_class"][klass][name]:
                raise ValueError(f"Frozen class scoring mismatch: {klass}.{name}")

    locked_matches = 0
    rescued_rows: list[dict[str, Any]] = []
    remaining_after_shadow = Counter()
    remaining_before_shadow = Counter()
    raw_class_counts = Counter()
    for source_frame in frames:
        truths = fluid[source_frame + OFFSET]
        preds = prediction_by_frame[source_frame]
        raw_class_counts.update(t["mapped_class"] for t in truths)
        used_pred: set[int] = set()
        used_truth: set[int] = set()
        for klass in CLASSES:
            pi = [i for i, d in enumerate(preds) if d.vehicle_class == klass]
            ti = [i for i, t in enumerate(truths) if t["mapped_class"] == klass]
            for p_idx, t_idx, _distance in _pairs(
                [preds[i] for i in pi], [truths[j] for j in ti]
            ):
                used_pred.add(pi[p_idx])
                used_truth.add(ti[t_idx])
                locked_matches += 1
        residual_p_idx = [i for i in range(len(preds)) if i not in used_pred]
        residual_t_idx = [i for i in range(len(truths)) if i not in used_truth]
        residual_p = [preds[i] for i in residual_p_idx]
        residual_t = [truths[i] for i in residual_t_idx]
        remaining_before_shadow.update(p.vehicle_class for p in residual_p)
        shadow_pairs = _pairs(residual_p, residual_t)
        shadow_used = set()
        for pi, ti, distance in shadow_pairs:
            p, t = residual_p[pi], residual_t[ti]
            if p.vehicle_class == t["mapped_class"]:
                raise AssertionError(
                    "Leftover same-class pairing indicates class-aware matching was not maximal")
            shadow_used.add(pi)
            rescued_rows.append({
                "video_frame": source_frame,
                "prediction_class": p.vehicle_class,
                "prediction_confidence": p.confidence,
                "prediction_x_px": p.x_px,
                "prediction_y_px": p.y_px,
                "original_fluid_track_id": t["fluid_id"],
                "original_fluid_type": t["raw_fluid_type"],
                "mapped_fluid_class": t["mapped_class"],
                "fluid_x_px": t["x_px"], "fluid_y_px": t["y_px"],
                "distance_px": distance,
                "fluid_class_in_frozen_four_class_denominator":
                    t["mapped_class"] in CLASSES,
            })
        remaining_after_shadow.update(
            residual_p[i].vehicle_class
            for i in range(len(residual_p)) if i not in shadow_used)
    if locked_matches != scored["matched_points"]:
        raise AssertionError("Frozen class-aware match count changed")
    total_remaining = scored["predicted_points"] - locked_matches
    if sum(remaining_before_shadow.values()) != total_remaining:
        raise AssertionError("Original unmatched count mismatch")
    if len(rescued_rows) + sum(remaining_after_shadow.values()) != total_remaining:
        raise AssertionError("Shadow annotation pair counts do not sum correctly")

    class_pairs = Counter((r["prediction_class"], r["mapped_fluid_class"],
                           r["original_fluid_type"]) for r in rescued_rows)
    return {
        "status": "SHADOW_RAW_FLUID_CLASS_AGNOSTIC_PRESENCE_NOT_PHYSICAL_FP_PROOF",
        "eligible_for_promotion": False,
        "source_dir": str(audit_dir),
        "source_model_sha256": report["model_sha256"],
        "source_runtime_model_sha256": report["runtime_model_sha256"],
        "frame_offset": OFFSET, "radius_px": RADIUS,
        "sample_frames": frames,
        "original_frozen_class_aware": {
            "truth": scored["truth_points"],
            "predicted": scored["predicted_points"],
            "matched": locked_matches,
            "unmatched": total_remaining,
            "precision": scored["precision"],
            "recall": scored["recall"],
        },
        "all_original_fluid_annotations_by_mapped_type":
            dict(sorted(raw_class_counts.items())),
        "additional_shadow_cross_class_spatial_pairs": len(rescued_rows),
        "pairs_to_labels_outside_original_four_classes": sum(
            not row["fluid_class_in_frozen_four_class_denominator"]
            for row in rescued_rows),
        "originally_unmatched_by_prediction_class":
            dict(sorted(remaining_before_shadow.items())),
        "remaining_without_unused_FLUID_center_within_50px_by_class":
            dict(sorted(remaining_after_shadow.items())),
        "shadow_class_pair_counts": [
            {"predicted_class": a, "mapped_fluid_class": b,
             "raw_fluid_type": c, "count": n}
            for (a, b, c), n in sorted(class_pairs.items())
        ],
        "shadow_pairs": sorted(rescued_rows, key=lambda r:
                               (r["video_frame"], r["prediction_class"],
                                r["original_fluid_track_id"])),
        "limitations": (
            "Preserves all frozen class-aware matches, then only links previously "
            "unmatched detections to unused original FLUID annotation CENTERS of "
            "another class, including unsupported PEDESTRIAN/LIGHT_COMMERCIAL. "
            "A shadow cross-class spatial pairing is not proof the two centers "
            "refer to the same physical vehicle. Remaining unpaired detections "
            "are NOT confirmed physical false positives: FLUID may omit objects. "
            "This is a label ontology / object-presence DIAGNOSTIC, not official "
            "precision, recall, class confusion ground truth, NMS, or tracking. "
            "This tuned W04 sample cannot validate generalization. "
            "No scores, thresholds, annotations or source files are changed."
        ),
    }
