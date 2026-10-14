"""Explain the 166 residual predictions WITHOUT equating 'unpaired' to false positive.

Inputs are previously frozen SHA-linked detector scores, the read-only
187-row unmatched review, and the 21-pair shadow annotation result.
Produces exclusive REVIEW TRIAGE TIERS and independent proximity flags.
No inference, relabeling, suppression or new benchmark metric.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from streetlab_phase3.class_mapping import map_vehicle_class
from streetlab_phase3.video.cached_detector_subset import _read_audit

RADIUS = 50.0
NEAR = 25.0


def _identity(frame: int, klass: str, x: float, y: float, confidence: float) -> tuple[Any, ...]:
    return (frame, klass, x, y, confidence)


def triage_unmatched_residual(
    *, audit_dir: Path, review_file: Path, shadow_file: Path,
) -> dict[str, Any]:
    src = _read_audit(audit_dir)
    if src.get("runtime_backend") != "openvino":
        raise ValueError("OpenVINO detector-only audit required")
    review = json.loads(review_file.read_text(encoding="utf-8"))
    shadow = json.loads(shadow_file.read_text(encoding="utf-8"))
    frames = src["sample_frames"]
    if (review.get("status")
            != "EXPLORATORY_DETECTOR_UNMATCHED_REVIEW_NOT_FALSE_POSITIVE_VERDICT"
        or review.get("eligible_for_promotion") is not False
        or review.get("frames") != frames
        or Path(review.get("prediction_source", "")).resolve()
            != (audit_dir / "sliced_detections.csv").resolve()
        or review.get("matched_predictions") != src["sliced"]["matched_points"]
        or review.get("unmatched_predictions") != (
            src["sliced"]["predicted_points"] - src["sliced"]["matched_points"])):
        raise ValueError("Precision-review evidence does not match detector-only audit")
    if (shadow.get("status")
            != "SHADOW_RAW_FLUID_CLASS_AGNOSTIC_PRESENCE_NOT_PHYSICAL_FP_PROOF"
        or shadow.get("eligible_for_promotion") is not False
        or shadow.get("sample_frames") != frames
        or shadow.get("radius_px") != RADIUS
        or shadow.get("frame_offset") != 1
        or shadow.get("source_model_sha256") != src["model_sha256"]
        or shadow.get("source_runtime_model_sha256") != src["runtime_model_sha256"]
        or Path(shadow.get("source_dir", "")).resolve() != audit_dir.resolve()):
        raise ValueError("Shadow raw FLUID evidence does not match detector-only audit")
    frozen = shadow["original_frozen_class_aware"]
    for name, expected in (
        ("truth", src["sliced"]["truth_points"]),
        ("predicted", src["sliced"]["predicted_points"]),
        ("matched", src["sliced"]["matched_points"]),
        ("unmatched", src["sliced"]["predicted_points"] -
         src["sliced"]["matched_points"]),
    ):
        if frozen.get(name) != expected:
            raise ValueError(f"Frozen score drift in shadow report: {name}")
    unmatched_rows = review["unmatched_rows"]
    if len(unmatched_rows) != frozen["unmatched"]:
        raise ValueError("Precision review contains incorrect unmatched row count")
    cross_pairs = shadow["shadow_pairs"]
    if len(cross_pairs) != shadow["additional_shadow_cross_class_spatial_pairs"]:
        raise ValueError("Shadow class-pairing count mismatch")

    # Multiset identities protect against accidentally treating identical rows
    # as a single observation. We subtract only explicitly shadow-paired rows.
    pending = Counter(_identity(
        int(r["video_frame"]), r["class"], float(r["x_px"]),
        float(r["y_px"]), float(r["confidence"])) for r in unmatched_rows)
    cross_by_frame = defaultdict(list)
    for r in cross_pairs:
        key = _identity(
            int(r["video_frame"]), r["prediction_class"],
            float(r["prediction_x_px"]), float(r["prediction_y_px"]),
            float(r["prediction_confidence"]))
        if pending[key] <= 0:
            raise ValueError("Shadow pairing is not from the unmatched detector rows")
        pending[key] -= 1
        cross_by_frame[r["video_frame"]].append(r)

    # Only frame+1 annotations are used, preserving original (possibly
    # unsupported) classes and raw type strings without changing scorer truth.
    wanted = {frame + 1 for frame in frames}
    truth = defaultdict(list)
    with Path(src["trial"]["fluid_tracks"]).open("r", encoding="utf-8-sig",
                                                 newline="") as f:
        reader = csv.DictReader(f)
        if not {"frame", "cx", "cy", "type", "id"}.issubset(reader.fieldnames or []):
            raise ValueError("Invalid original FLUID columns")
        for row in reader:
            frame = int(float(row["frame"]))
            if frame not in wanted or not row["cx"] or not row["cy"]:
                continue
            x, y = float(row["cx"]), float(row["cy"])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("Nonfinite raw annotation coordinates")
            truth[frame].append({
                "fluid_id": str(row["id"]),
                "raw_type": str(row["type"]),
                "mapped_type": map_vehicle_class(row["type"]).canonical.value,
                "x_px": x, "y_px": y,
            })

    flags = Counter()
    tiers = Counter()
    by_class = defaultdict(Counter)
    residual_rows = []
    for original in unmatched_rows:
        row = dict(original)
        key = _identity(int(row["video_frame"]), row["class"],
                        float(row["x_px"]), float(row["y_px"]),
                        float(row["confidence"]))
        if pending[key] <= 0:
            continue
        pending[key] -= 1
        frame = int(row["video_frame"])
        nearest = sorted(
            (
                {**t, "distance_px": math.hypot(row["x_px"] - t["x_px"],
                                                row["y_px"] - t["y_px"])}
                for t in truth[frame + 1]
            ), key=lambda t: (t["distance_px"], t["fluid_id"]))
        same = row.get("nearest_same_class_truth_px")
        matched = row.get("nearest_matched_same_class_prediction_px")
        raw = nearest[0] if nearest else None
        raw_dist = raw["distance_px"] if raw else None
        near_matched = matched is not None and matched <= NEAR
        near_same_truth = same is not None and same <= RADIUS
        near_any_raw = raw_dist is not None and raw_dist <= RADIUS
        near_other_raw = near_any_raw and raw["mapped_type"] != row["class"]
        for name, flag in (
            ("near_matched_same_class_center_25px", near_matched),
            ("near_same_class_FLUID_center_50px", near_same_truth),
            ("near_other_raw_FLUID_class_center_50px", near_other_raw),
            ("near_any_raw_FLUID_center_50px", near_any_raw),
            ("no_raw_FLUID_center_within_50px", not near_any_raw),
        ):
            if flag:
                flags[name] += 1
        if near_matched:
            tier = "review_center_cluster_near_matched_prediction"
        elif near_same_truth:
            tier = "review_near_same_class_FLUID_center"
        elif near_other_raw:
            tier = "review_near_other_raw_FLUID_class_center"
        elif near_any_raw:
            tier = "review_near_raw_FLUID_center"
        else:
            tier = "review_no_raw_FLUID_center_within_50px"
        tiers[tier] += 1
        by_class[row["class"]][tier] += 1
        residual_rows.append({
            "video_frame": frame, "prediction_class": row["class"],
            "x_px": row["x_px"], "y_px": row["y_px"],
            "confidence": row["confidence"],
            "review_tier": tier,
            "near_matched_same_class_prediction_25px": near_matched,
            "near_any_raw_FLUID_center_50px": near_any_raw,
            "nearest_original_FLUID": raw,
        })
    if any(pending.values()):
        raise AssertionError("Unmatched multiset subtraction is incomplete")
    if (len(residual_rows) + len(cross_pairs) != frozen["unmatched"]
        or dict(Counter(r["prediction_class"] for r in residual_rows))
        != shadow["remaining_without_unused_FLUID_center_within_50px_by_class"]):
        raise ValueError("Residual counts differ from shadow ontology report")

    # Sorted manual-review samples, not a random or independent cohort.
    examples = []
    for tier in (
        "review_center_cluster_near_matched_prediction",
        "review_near_other_raw_FLUID_class_center",
        "review_no_raw_FLUID_center_within_50px",
        "review_near_same_class_FLUID_center",
        "review_near_raw_FLUID_center",
    ):
        for klass in ("MOTORCYCLE", "CAR", "BUS", "HEAVY_VEHICLE"):
            candidates = sorted(
                (r for r in residual_rows
                 if r["review_tier"] == tier and r["prediction_class"] == klass),
                key=lambda r: (-r["confidence"], r["video_frame"],
                               r["x_px"], r["y_px"]))
            examples.extend(candidates[:2])

    return {
        "status": "W04_UNMATCHED_RESIDUAL_REVIEW_TIERS_NOT_FP_LABELS",
        "eligible_for_promotion": False,
        "source_audit": str(audit_dir),
        "source_model_sha256": src["model_sha256"],
        "source_runtime_model_sha256": src["runtime_model_sha256"],
        "original_frozen_class_aware": frozen,
        "shadow_cross_class_pairs": len(cross_pairs),
        "residual_unpaired_predictions": len(residual_rows),
        "exclusive_review_tiers": dict(sorted(tiers.items())),
        "overlapping_proximity_flags": dict(sorted(flags.items())),
        "exclusive_tiers_by_prediction_class": {
            k: dict(sorted(by_class[k].items())) for k in
            ("MOTORCYCLE", "CAR", "BUS", "HEAVY_VEHICLE")
        },
        "representative_review_queue": examples[:32],
        "residual_rows": residual_rows,
        "limitations": (
            "These are exclusive, *heuristically prioritized REVIEW tiers*, "
            "not confirmed detector duplicates, true physical false alarms, "
            "misclassified vehicles, or annotation errors. Near a matched "
            "prediction CENTER does not prove same-object bbox IoU. "
            "A raw FLUID center can be already consumed by another prediction. "
            "No nearby FLUID center does not imply absent physical vehicle. "
            "Overlapping proximity flags have intentionally nonexclusive sums. "
            "The 21-frame tuned W04 sample, original scorer, all FLUID labels, "
            "tracking identities and detection outputs are unchanged."
        ),
    }
