"""Offline fixed-cohort W04 comparison; does NOT run detection or tracking.

Compare two existing raw Geo-trax pixel exports over identical source frames.
Preserve differing detector resolutions and warm-up settings as confounders.
Never issue a production promotion decision from a short pilot.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
from streetlab_phase3.identity_benchmark import IdentityBenchmark
from streetlab_phase3.pixel_benchmark import PixelBenchmark, normalize_fluid_pixel_truth
from streetlab_phase3.serialization import package_to_dict
from streetlab_phase3.video.class_diagnostics import class_diagnostics


def _manifest(path: Path) -> dict[str, Any]:
    filename = path.with_suffix(".manifest.json")
    payload = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest at {filename}")
    return payload


def _metric_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    av, bv = a.get(key), b.get(key)
    if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
        return None
    if not (math.isfinite(av) and math.isfinite(bv)):
        return None
    return float(bv - av)


def compare_existing_runs(
    *, standard_tracks: Path, sliced_tracks: Path, fluid_tracks: Path,
    start_frame: int, end_frame: int,
) -> dict[str, Any]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid inclusive evaluation window")
    if standard_tracks.resolve() == sliced_tracks.resolve():
        raise ValueError("Cannot compare a file with itself")
    standard_manifest = _manifest(standard_tracks)
    sliced_manifest = _manifest(sliced_tracks)
    if (standard_manifest.get("mode") != "standard" or
            sliced_manifest.get("mode") != "sliced"):
        raise ValueError("Expected verified standard and sliced tracking manifests")

    for field in ("video", "model_sha256", "confidence", "class_map",
                  "track_activation_threshold", "high_conf_det_threshold",
                  "lost_track_buffer", "minimum_consecutive_frames",
                  "evaluation_frame_offset"):
        if standard_manifest.get(field) != sliced_manifest.get(field):
            raise ValueError(f"Noncomparable inputs: {field} differs")

    if not standard_manifest.get("model_sha256"):
        raise ValueError("Checkpoint SHA-256 must be available for both runs")
    for name, manifest in (("standard", standard_manifest), ("sliced", sliced_manifest)):
        if (not isinstance(manifest.get("start_frame"), int) or
                not isinstance(manifest.get("end_frame"), int) or
                not manifest["start_frame"] <= start_frame <= end_frame <= manifest["end_frame"]):
            raise ValueError(f"{name} result does not cover requested fixed video window")
        if manifest.get("evaluation_frame_offset") != 1:
            raise ValueError("FLUID frame alignment must be frozen at +1")

    with fluid_tracks.open("r", encoding="utf-8-sig", newline="") as fh:
        truth = normalize_fluid_pixel_truth(csv.DictReader(fh))
    if not truth:
        raise ValueError("No supported FLUID annotations found")
    window = (start_frame + 1, end_frame + 1)
    results: dict[str, Any] = {}
    for name, filename, manifest in (
        ("standard", standard_tracks, standard_manifest),
        ("sliced", sliced_tracks, sliced_manifest),
    ):
        predictions = load_geotrax_pixel_tracks(filename)
        metric = package_to_dict(PixelBenchmark(50.).evaluate(
            predictions, truth, frame_offsets=(1,), evaluation_frame_range=window))
        identity = IdentityBenchmark(50.).evaluate(
            predictions, truth, frame_offsets=(1,),
            evaluation_frame_range=window).scorecard_dict()
        detail = class_diagnostics(predictions, truth, start_frame=start_frame,
                                   end_frame=end_frame)
        results[name] = {
            "tracks": str(filename),
            "manifest": {
                k: manifest.get(k) for k in (
                    "mode", "model_sha256", "image_size", "slice_height",
                    "slice_width", "overlap", "warmup_frames", "confidence",
                    "elapsed_seconds", "processed_frames",
                    "first_decoded_frame", "evaluation_frames",
                )
            },
            "pixel": metric,
            "identity": identity,
            "class_diagnostics": detail,
        }

    a, b = results["standard"], results["sliced"]
    truth_count = a["pixel"]["truth_points"]
    if truth_count != b["pixel"]["truth_points"]:
        raise ValueError("Truth denominator differs across runs")
    for klass in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE"):
        if (a["class_diagnostics"]["by_class"][klass]["truth_points"] !=
                b["class_diagnostics"]["by_class"][klass]["truth_points"]):
            raise ValueError(f"Different {klass} truth denominators")

    changes = {
        "point_recall_percentage_points":
            100 * _metric_delta(a["pixel"], b["pixel"], "point_recall")
            if _metric_delta(a["pixel"], b["pixel"], "point_recall") is not None else None,
        "point_precision_percentage_points":
            100 * _metric_delta(a["pixel"], b["pixel"], "point_precision")
            if _metric_delta(a["pixel"], b["pixel"], "point_precision") is not None else None,
        "correct_class_motorcycle_recall_percentage_points": None,
        "motorcycle_spatial_misses_delta": (
            b["class_diagnostics"]["by_class"]["MOTORCYCLE"]["spatial_misses"] -
            a["class_diagnostics"]["by_class"]["MOTORCYCLE"]["spatial_misses"]
        ),
        "unmatched_predictions_delta":
            (b["pixel"]["predicted_points"] - b["pixel"]["matched_points"]) -
            (a["pixel"]["predicted_points"] - a["pixel"]["matched_points"]),
        "pixel_mae_delta_px": _metric_delta(a["pixel"], b["pixel"], "pixel_mae"),
    }
    mc_key = "correct_class_recall"
    mca = a["class_diagnostics"]["by_class"]["MOTORCYCLE"]
    mcb = b["class_diagnostics"]["by_class"]["MOTORCYCLE"]
    if mca[mc_key] is not None and mcb[mc_key] is not None:
        changes["correct_class_motorcycle_recall_percentage_points"] = (
            100 * (mcb[mc_key] - mca[mc_key]))

    confounders = []
    for key, display in (
        ("image_size", "different detector input resolutions"),
        ("warmup_frames", "different tracker warm-up lengths"),
        ("first_decoded_frame", "different first processed source frames"),
    ):
        if standard_manifest.get(key) != sliced_manifest.get(key):
            confounders.append(display)
    if end_frame - start_frame + 1 < 100:
        confounders.append("fewer than 100 evaluation frames: identity metrics unstable")
    return {
        "status": "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE",
        "eligible_for_promotion": False,
        "evaluation_start_frame": start_frame,
        "evaluation_end_frame": end_frame,
        "evaluation_frame_count": end_frame - start_frame + 1,
        "frozen_fluid_offset": 1,
        "frozen_matching_radius_px": 50,
        "truth_points": truth_count,
        "confounders": confounders,
        "interpretation": (
            "Same FLUID truth and video frames; only a short, non-equivalent-"
            "configuration diagnostic. Compare class-level motorcycle recall and "
            "unmatched predictions; repeat longer with equal warm-up before any gate."
        ),
        "deltas_sliced_minus_standard": changes,
        "results": results,
    }
