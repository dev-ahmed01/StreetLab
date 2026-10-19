"""Fail-closed display adapter for the frozen May25 autonomous object audit.

Read-only; never recomputes evidence, runs inference or promotes the cohort.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

class AuditDisplayError(ValueError):
    pass

def read_audit(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise AuditDisplayError("Audit file is missing or linked")
    if path.stat().st_size > 10_000_000:
        raise AuditDisplayError("Audit exceeds display size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise AuditDisplayError("Unable to read audit") from exc
    if not isinstance(value, dict):
        raise AuditDisplayError("Invalid audit root")
    from streetlab_integration.w04_autonomous_diagnostics import STATUS
    if value.get("status") != STATUS:
        raise AuditDisplayError("Original automated audit not completed")
    if (value.get("source_video_frame_window") != [60, 179]
        or value.get("FLUID_frame_window") != [61, 180]
        or value.get("physical_object_labels_independently_adjudicated") is not False
        or value.get("independent_holdout_pass") is not False
        or value.get("production_release_approved") is not False
        or value.get("no_new_detector_inference") is not True
        or value.get("no_changes_to_model_or_tracking_policy") is not True
        or value.get("no_FLUID_relabeling") is not True):
        raise AuditDisplayError("Frozen method and release boundaries not satisfied")
    sources = value.get("original_video_and_FLUID_SHA")
    if not isinstance(sources, dict):
        raise AuditDisplayError("Source provenance absent")
    for key in ("original_video_sha256", "original_truth_sha256"):
        digest = sources.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AuditDisplayError("Original source fingerprint missing")
    stats = value.get("recomputed_class_diagnostics")
    if not isinstance(stats, dict) or not isinstance(stats.get("by_class"), dict):
        raise AuditDisplayError("Class scorecard absent")
    by_class = stats["by_class"]
    if any(k not in by_class for k in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")):
        raise AuditDisplayError("Incomplete class scorecard")
    heavy = by_class["HEAVY_VEHICLE"]
    bus = by_class["BUS"]
    for key in ("truth_points", "spatial_matches", "correct_class_matches", "class_errors_on_spatial_matches"):
        if type(heavy.get(key)) is not int or heavy[key] < 0:
            raise AuditDisplayError("Invalid HEAVY counts")
    if (heavy["spatial_matches"] != heavy["correct_class_matches"] + heavy["class_errors_on_spatial_matches"]
        or heavy["truth_points"] < heavy["spatial_matches"]
        or bus.get("truth_points") != 0
        or value.get("BUS_FLUID_truth_points") != 0):
        raise AuditDisplayError("Frozen cohort class accounting failed")
    if type(value.get("HEAVY_wrong_class_matched_points")) is not int or value["HEAVY_wrong_class_matched_points"] != heavy["class_errors_on_spatial_matches"]:
        raise AuditDisplayError("HEAVY object accounting does not conserve")
    if type(value.get("HEAVY_truth_tracks_with_class_disagreement")) is not int:
        raise AuditDisplayError("Distinct HEAVY track count absent")
    rows = value.get("truth_track_diagnostics")
    buses = value.get("BUS_predicted_track_diagnostics")
    predicted = value.get("predicted_track_class_stability")
    if not all(isinstance(item, list) for item in (rows, buses, predicted)):
        raise AuditDisplayError("Whole-cohort track rows absent")
    if len(rows) != value.get("unique_FLUID_truth_tracks") or len(predicted) != value.get("unique_predicted_tracks"):
        raise AuditDisplayError("Track row totals do not conserve")
    heavy_rows = [r for r in rows if isinstance(r, dict) and r.get("truth_class") == "HEAVY_VEHICLE"]
    heavy_errors = [r for r in heavy_rows if type(r.get("wrong_class_matches")) is int and r["wrong_class_matches"] > 0]
    if (sum(r["wrong_class_matches"] for r in heavy_errors) != value["HEAVY_wrong_class_matched_points"]
        or len(heavy_errors) != value["HEAVY_truth_tracks_with_class_disagreement"]
        or len(buses) != value.get("BUS_unique_predicted_track_ids")):
        raise AuditDisplayError("Unique object attribution does not conserve")
    if (value.get("truth_points") != stats.get("truth_points")
        or value.get("matched_points") != stats.get("matched_points")
        or value.get("predicted_points") != stats.get("predicted_points")):
        raise AuditDisplayError("Original matcher counts disagree")
    f1 = value.get("original_saved_FLUID_label_F1")
    if type(f1) not in (int, float) or not math.isfinite(f1) or not 0 <= f1 <= 1:
        raise AuditDisplayError("Invalid original F1")
    return {
        "status": "verified_saved_audit_display",
        "historical_regression": True,
        "source_video_frame_window": [60,179],
        "matched_points": value["matched_points"],
        "truth_points": value["truth_points"],
        "spatial_f1_pct": round(100*f1,2),
        "heavy_wrong_class_frame_points": value["HEAVY_wrong_class_matched_points"],
        "heavy_truth_track_ids_with_disagreement": len(heavy_errors),
        "bus_predicted_unique_track_ids": len(buses),
        "bus_label_accuracy_measurable": False,
        "original_video_sha256": sources["original_video_sha256"],
        "original_truth_sha256": sources["original_truth_sha256"],
        "production_release_approved": False,
        "independent_holdout_pass": False,
        "native_sumo_real_site_calibrated": False,
    }
