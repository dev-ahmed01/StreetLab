"""Compare a wider cached detector audit with a smaller paired CPU tile probe.

Re-scores the *identical* candidate sample frames from prior cached CSVs;
never compares a 21-frame reference aggregate against a 5-frame candidate.
Class-aware, max-cardinality detection-only scorer; NO ByteTrack identities.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
from streetlab_phase3.video.sahi_detection_audit import (
    CLASSES, Detection, score_detections,
)


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_audit(directory: Path) -> dict[str, Any]:
    data = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if data.get("status") != "EXPERIMENTAL_DETECTION_ONLY":
        raise ValueError("Expected a complete and verified detection-only audit")
    frames = data.get("sample_frames")
    if (not isinstance(frames, list) or not frames
        or any(type(f) is not int or f < 0 for f in frames)
        or len(frames) != len(set(frames))
        or frames != sorted(frames)):
        raise ValueError("Invalid unique sampled frames in audit report")
    trial = data["trial"]
    if list(range(trial["start_frame"], trial["end_frame"] + 1,
                  trial["sample_step"])) != frames:
        raise ValueError("Reported samples do not match declared sample step")
    if trial.get("frame_offset") != 1 or trial.get("max_pixel_distance") != 50:
        raise ValueError("Frozen +1 and 50px scoring required")
    if _sha(Path(trial["weights"])) != data["model_sha256"]:
        raise ValueError("Checkpoint bytes differ from source manifest")
    if _sha(Path(trial["fluid_tracks"])) != data["ground_truth_sha256"]:
        raise ValueError("Truth annotations differ from source manifest")
    return data


def _detections(directory: Path, mode: str, frames: list[int]) -> dict[int, list[Detection]]:
    records: dict[int, list[Detection]] = {f: [] for f in frames}
    with (directory / f"{mode}_detections.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            cls = row["vehicle_class"]
            x, y, confidence = (float(row[field])
                                for field in ("x_px", "y_px", "confidence"))
            if frame not in records or cls not in CLASSES or not all(
                math.isfinite(z) for z in (x, y, confidence)
            ) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid {mode} detection or out-of-window sample")
            records[frame].append(Detection(frame, x, y, cls, confidence))
    return records


def compare_cached_detector_subsets(
    reference_dir: Path, candidate_dir: Path,
) -> dict[str, Any]:
    if reference_dir.resolve() == candidate_dir.resolve():
        raise ValueError("Use distinct reference and candidate directories")
    ref = _read_audit(reference_dir)
    candidate = _read_audit(candidate_dir)
    reference_frames, frames = ref["sample_frames"], candidate["sample_frames"]
    if len(frames) < 2 or not set(frames).issubset(reference_frames):
        raise ValueError("Candidate must select two or more frames from cached reference")
    for key in ("video", "fluid_tracks", "weights", "confidence",
                "image_size", "device", "frame_offset",
                "max_pixel_distance", "class_map_file"):
        if ref["trial"].get(key) != candidate["trial"].get(key):
            raise ValueError(f"Reference and candidate have different {key}")
    for key in ("model_sha256", "ground_truth_sha256", "class_map"):
        if ref[key] != candidate[key]:
            raise ValueError(f"Reference and candidate have different {key}")

    with Path(ref["trial"]["fluid_tracks"]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        truth = normalize_fluid_pixel_truth(csv.DictReader(handle))
    if not any(p.vehicle_class == "MOTORCYCLE"
               and p.frame in {f + 1 for f in frames} for p in truth):
        raise ValueError("Selected frames contain no motorcycle truth")

    scores: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for label, directory, data, audit_frames in (
        ("cached_full_640", reference_dir, ref, reference_frames),
        ("cached_sliced_640", reference_dir, ref, reference_frames),
        ("candidate_full_640", candidate_dir, candidate, frames),
        ("candidate_sliced", candidate_dir, candidate, frames),
    ):
        mode = "sliced" if "sliced" in label else "standard"
        all_det = _detections(directory, mode, audit_frames)
        entire = score_detections(all_det, truth, audit_frames)
        stored = data[mode]
        for key in ("truth_points", "predicted_points", "matched_points"):
            if entire[key] != stored[key]:
                raise ValueError(f"Export {label} differs from stored {key}")
        selected = {f: all_det[f] for f in frames}
        scores[label] = score_detections(selected, truth, frames)
        evidence[label] = {
            "source": str(directory / f"{mode}_detections.csv"),
            "sha256": _sha(directory / f"{mode}_detections.csv"),
            "original_evaluation_frame_count": len(audit_frames),
            "timing_median_across_original_sample_s": stored["latency_median_s"],
        }
    expected_truth = {scores[k]["truth_points"] for k in scores}
    expected_moto = {
        scores[k]["per_class"]["MOTORCYCLE"]["truth"] for k in scores}
    if len(expected_truth) != 1 or len(expected_moto) != 1:
        raise AssertionError("Inconsistent ground-truth denominator")
    ref_sahi = scores["cached_sliced_640"]
    proposed = scores["candidate_sliced"]
    m_ref = ref_sahi["per_class"]["MOTORCYCLE"]["recall"]
    m_new = proposed["per_class"]["MOTORCYCLE"]["recall"]
    return {
        "status": "FIXED_SUBSET_DETECTION_ONLY_CPU_PROBE_NOT_PRODUCTION",
        "eligible_for_promotion": False,
        "sample_frames": frames,
        "source_video_frame_span": [frames[0], frames[-1]],
        "same_truth_points": expected_truth.pop(),
        "same_motorcycle_truth_points": expected_moto.pop(),
        "reference_slice_config": {
            k: ref["trial"][k] for k in ("image_size", "slice_height",
                                         "slice_width", "overlap")
        },
        "candidate_slice_config": {
            k: candidate["trial"][k] for k in ("image_size", "slice_height",
                                               "slice_width", "overlap")
        },
        "scores": scores,
        "sliced_candidate_minus_reference": {
            "motorcycle_recall_percentage_points": (
                100 * (m_new - m_ref) if m_ref is not None and m_new is not None else None),
            "overall_recall_percentage_points": 100 * (
                proposed["recall"] - ref_sahi["recall"]),
            "overall_precision_percentage_points": (
                100 * (proposed["precision"] - ref_sahi["precision"])
                if ref_sahi["precision"] is not None
                and proposed["precision"] is not None else None),
        },
        "export_evidence": evidence,
        "limitations": (
            "All four detector-only scores use the identical candidate sample frames, "
            "the same checkpoint and a class-aware maximum-cardinality 50px matcher; "
            "not the frozen class-agnostic ByteTrack pixel score. The reference "
            "timing median covers its original 21-frame sample while candidate "
            "timing covers the smaller candidate sample, so it is not a paired "
            "per-frame latency comparison. No track identities, no holdout, "
            "no automatic parameter promotion."
        ),
    }
