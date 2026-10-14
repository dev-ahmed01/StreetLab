"""Paired ground-truth observation check for two cached detector-only audits.

Equal aggregate recall can conceal substitutions: this module compares the
*same annotated (frame, vehicle ID)* recovered by PyTorch and OpenVINO.
NO tracker identity is measured. Max-cardinality/class-aware association
uses the existing detection-only audit's explicit +1/50px scoring rule.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from streetlab_phase3.pixel_benchmark import FluidPixelTruth, normalize_fluid_pixel_truth
from streetlab_phase3.video.cached_detector_subset import _detections, _read_audit
from streetlab_phase3.video.sahi_detection_audit import CLASSES, Detection, score_detections

RADIUS = 50.
OFFSET = 1


def _matched_truth_ids(
    predictions: Sequence[Detection],
    truth: Sequence[FluidPixelTruth],
) -> set[str]:
    """Use same padded Hungarian costs as existing score_detections()."""
    if not predictions or not truth:
        return set()
    pxy = np.asarray([(p.x_px, p.y_px) for p in predictions], dtype=float)
    txy = np.asarray([(t.x_px, t.y_px) for t in truth], dtype=float)
    distances = np.sqrt(((pxy[:, None, :] - txy[None, :, :]) ** 2).sum(axis=2))
    n, m = distances.shape
    penalty = RADIUS + 1.
    impossible = (n + m + 1) * penalty * 4.
    cost = np.full((n + m, n + m), impossible)
    cost[:n, :m] = np.where(distances <= RADIUS, distances, impossible)
    cost[:n, m:] = penalty
    cost[n:, :m] = penalty
    cost[n:, m:] = 0.
    ri, ci = linear_sum_assignment(cost)
    return {truth[int(j)].track_id for i, j in zip(ri, ci)
            if i < n and j < m and distances[i, j] <= RADIUS}


def compare_detector_truth_identity(
    reference_dir: Path, candidate_dir: Path,
) -> dict[str, Any]:
    """Compare identities in FLUID *annotations*, not ByteTrack identities."""
    if reference_dir.resolve() == candidate_dir.resolve():
        raise ValueError("Use two distinct cached audit directories")
    ref, cand = _read_audit(reference_dir), _read_audit(candidate_dir)
    frames = cand["sample_frames"]
    if len(frames) < 2 or not set(frames).issubset(ref["sample_frames"]):
        raise ValueError("Reference must contain each candidate frame")
    for key in ("video", "weights", "fluid_tracks", "confidence", "image_size",
                "device", "class_map_file", "slice_height", "slice_width",
                "overlap", "frame_offset", "max_pixel_distance"):
        if ref["trial"].get(key) != cand["trial"].get(key):
            raise ValueError(f"Detector parity settings differ: {key}")
    for key in ("model_sha256", "ground_truth_sha256", "class_map"):
        if ref[key] != cand[key]:
            raise ValueError(f"Detector parity provenance differs: {key}")
    if (ref.get("runtime_backend", "pytorch") != "pytorch"
        or cand.get("runtime_backend") != "openvino"):
        raise ValueError("Expected PyTorch reference versus OpenVINO candidate")

    with Path(ref["trial"]["fluid_tracks"]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        all_truth = normalize_fluid_pixel_truth(csv.DictReader(stream))
    truth_by_key: dict[tuple[int, str], list[FluidPixelTruth]] = defaultdict(list)
    for t in all_truth:
        if t.frame - OFFSET in frames and t.vehicle_class in CLASSES:
            truth_by_key[(t.frame, t.vehicle_class)].append(t)

    det = {}
    for label, directory, data in (
        ("pytorch", reference_dir, ref),
        ("openvino", candidate_dir, cand),
    ):
        values = _detections(directory, "sliced", data["sample_frames"])
        full = score_detections(values, all_truth, data["sample_frames"])
        for key in ("truth_points", "predicted_points", "matched_points"):
            if full[key] != data["sliced"][key]:
                raise ValueError(f"Cached {label} detector rows do not reproduce {key}")
        subset = {f: values[f] for f in frames}
        det[label] = subset

    by_class: dict[str, Any] = {}
    total_discordances = 0
    for klass in CLASSES:
        both = ref_only = cand_only = neither = 0
        disagreements: list[dict[str, Any]] = []
        for frame in frames:
            labels = truth_by_key[(frame + OFFSET, klass)]
            ids = [t.track_id for t in labels]
            if len(set(ids)) != len(ids):
                raise ValueError(f"FLUID duplicate track id for {klass} source frame {frame}")
            recovered = {}
            for name in ("pytorch", "openvino"):
                preds = [p for p in det[name][frame] if p.vehicle_class == klass]
                recovered[name] = _matched_truth_ids(preds, labels)
            for obj in labels:
                a, b = (obj.track_id in recovered[name]
                        for name in ("pytorch", "openvino"))
                if a and b:
                    both += 1
                elif a:
                    ref_only += 1
                    disagreements.append({
                        "video_frame": frame, "fluid_track_id": obj.track_id,
                        "found_by": "pytorch_only"})
                elif b:
                    cand_only += 1
                    disagreements.append({
                        "video_frame": frame, "fluid_track_id": obj.track_id,
                        "found_by": "openvino_only"})
                else:
                    neither += 1
        ref_matched = both + ref_only
        cand_matched = both + cand_only
        # Must reproduce source's strict class-aware scorer, not merely its totals.
        for name, count in (("pytorch", ref_matched), ("openvino", cand_matched)):
            expected = score_detections(det[name], all_truth, frames)[
                "per_class"][klass]["matched"]
            if count != expected:
                raise AssertionError(
                    f"{klass} {name} match identity mapping diverges from scored count")
        total_discordances += ref_only + cand_only
        by_class[klass] = {
            "truth_observations": both + ref_only + cand_only + neither,
            "matched_by_both": both,
            "pytorch_only": ref_only,
            "openvino_only": cand_only,
            "neither": neither,
            "annotated_vehicle_observation_disagreements": disagreements,
        }
    return {
        "status": "PAIRED_DETECTOR_TRUTH_OBSERVATION_AUDIT_NOT_TRACKING",
        "eligible_for_promotion": False,
        "samples": frames,
        "reference_backend": "pytorch",
        "candidate_backend": "openvino",
        "same_checkpoint_sha256": ref["model_sha256"],
        "source_annotated_frame_offset": OFFSET,
        "class_aware_gate_radius_px": RADIUS,
        "same_slice_config": {
            k: ref["trial"][k] for k in ("image_size", "slice_height",
                                         "slice_width", "overlap")
        },
        "by_class": by_class,
        "total_annotation_observation_disagreements": total_discordances,
        "limitations": (
            "Compares FLUID annotated truth-observation IDs under the existing "
            "class-aware 50px detector-only matching rule. Equal recall can hide "
            "different recovered annotated vehicles. This does not recover or "
            "evaluate ByteTrack predicted IDs, temporal switches or fragmentation. "
            "The same tuned May-26 frames cannot establish held-out performance. "
            "The detector-only association, and possible FLUID omissions, limit "
            "physical error interpretations. No production promotion."
        ),
    }
