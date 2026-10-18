"""May-25 vs W04 forensic audit and source-bound error queue — NO inference.

Audit the original saved continuous-tracking batch against the original May-25
FLUID labels, then build a deterministic queue of candidate class errors.
The results are historical *regression* evidence, never blind validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .w04_holdout_intake import digest, read_json, validate_lock
from .w04_crossdate_regression import STATUS as REGRESSION_STATUS, WINDOW, WARMUP

AUDIT_STATUS = "W04_MAY25_REAL_REGRESSION_REPLAYED_WITH_REVIEW_QUEUE_NOT_BLIND"
EVENT_FIELDS = (
    "priority", "video_frame", "fluid_frame", "event", "truth_class",
    "predicted_class", "truth_track_id", "predicted_track_id",
    "truth_cx", "truth_cy", "predicted_cx", "predicted_cy",
    "matching_distance_px", "annotation_status",
)
WEIGHTS = {
    "HEAVY_WRONG_CLASS": 5,
    "HEAVY_TRUTH_UNMATCHED": 3,
    "BUS_PRED_MATCHED_NONBUS_TRUTH": 4,
    "BUS_PRED_UNMATCHED": 1,
    "MOTORCYCLE_TRUTH_UNMATCHED": 1,
}
TRUTH_CLASSES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")


class AuditError(ValueError):
    """Saved research records or source correspondences have changed."""


def _check_equal(actual: Any, expected: Any, context: str) -> None:
    """Compare nested frozen numerical evidence without string rounding."""
    if type(actual) in (float, int) and type(expected) in (float, int):
        if not (math.isfinite(actual) and math.isfinite(expected) and
                math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10)):
            raise AuditError(f"Recomputed value differs from frozen {context}")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AuditError(f"Field coverage differs from frozen {context}")
        for key in actual:
            _check_equal(actual[key], expected[key], context+"."+str(key))
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AuditError(f"Array length differs from frozen {context}")
        for i, (v, w) in enumerate(zip(actual, expected)):
            _check_equal(v, w, f"{context}[{i}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise AuditError(f"Original/recomputed frozen evidence mismatch at {context}")


def _event(event: str, *, video_frame: int,
           truth: Any = None, prediction: Any = None, distance: float | None = None) -> dict:
    return {
        "priority": WEIGHTS[event],
        "video_frame": video_frame,
        "fluid_frame": video_frame + 1,
        "event": event,
        "truth_class": truth.vehicle_class if truth is not None else None,
        "predicted_class": prediction.vehicle_class if prediction is not None else None,
        "truth_track_id": truth.track_id if truth is not None else None,
        "predicted_track_id": prediction.track_id if prediction is not None else None,
        "truth_cx": truth.x_px if truth is not None else None,
        "truth_cy": truth.y_px if truth is not None else None,
        "predicted_cx": prediction.x_px if prediction is not None else None,
        "predicted_cy": prediction.y_px if prediction is not None else None,
        "matching_distance_px": distance,
        "annotation_status": "FLUID_LABEL_ONLY_NOT_PHYSICALLY_ADJUDICATED",
    }


def review_candidates(predictions: Sequence[Any], truth: Sequence[Any],
                      *, start_frame: int = 60, end_frame: int = 179,
                      max_frames: int = 18, max_events: int = 220) -> dict:
    """Use unchanged +1/50px class-agnostic Hungarian correspondence."""
    from streetlab_phase3.pixel_benchmark import PixelBenchmark
    if max_frames <= 0 or max_events <= 0:
        raise AuditError("Review export caps must be positive")
    selected_truth = [p for p in truth
                      if start_frame+1 <= p.frame <= end_frame+1 and p.vehicle_class in TRUTH_CLASSES]
    selected_preds = [p for p in predictions
                      if start_frame <= p.frame <= end_frame and p.vehicle_class in TRUTH_CLASSES]
    matches = PixelBenchmark(50.).match_points(selected_preds, selected_truth, frame_offset=1)
    used_truth = {id(t) for _, t, _ in matches}
    used_pred = {id(p) for p, _, _ in matches}
    all_events = []
    for pred, actual, distance in matches:
        if actual.vehicle_class == "HEAVY_VEHICLE" and pred.vehicle_class != "HEAVY_VEHICLE":
            all_events.append(_event("HEAVY_WRONG_CLASS", video_frame=pred.frame,
                                     truth=actual, prediction=pred, distance=distance))
        if pred.vehicle_class == "BUS" and actual.vehicle_class != "BUS":
            all_events.append(_event("BUS_PRED_MATCHED_NONBUS_TRUTH", video_frame=pred.frame,
                                     truth=actual, prediction=pred, distance=distance))
    for actual in selected_truth:
        if id(actual) not in used_truth:
            if actual.vehicle_class == "HEAVY_VEHICLE":
                all_events.append(_event("HEAVY_TRUTH_UNMATCHED",
                                         video_frame=actual.frame-1, truth=actual))
            if actual.vehicle_class == "MOTORCYCLE":
                all_events.append(_event("MOTORCYCLE_TRUTH_UNMATCHED",
                                         video_frame=actual.frame-1, truth=actual))
    for pred in selected_preds:
        if id(pred) not in used_pred and pred.vehicle_class == "BUS":
            all_events.append(_event("BUS_PRED_UNMATCHED", video_frame=pred.frame, prediction=pred))

    per_frame: dict[int, Counter[str]] = defaultdict(Counter)
    for event in all_events:
        per_frame[event["video_frame"]][event["event"]] += 1
    ranked = sorted(
        per_frame, key=lambda frame: (
            -sum(WEIGHTS[typ] * count for typ, count in per_frame[frame].items()),
            -per_frame[frame]["HEAVY_WRONG_CLASS"], frame,
        ),
    )
    chosen = set(ranked[:max_frames])
    events = sorted(
        (item for item in all_events if item["video_frame"] in chosen),
        key=lambda e: (
            -WEIGHTS[e["event"]], e["video_frame"], e["event"],
            str(e["truth_track_id"]), str(e["predicted_track_id"]),
        ),
    )[:max_events]
    frames = [
        {"video_frame": frame, "fluid_frame": frame+1,
         "evidence_priority_score": sum(WEIGHTS[k] * n for k, n in per_frame[frame].items()),
         "candidate_error_counts": dict(sorted(per_frame[frame].items())),
         "review_is_independent_human_adjudication": False}
        for frame in ranked[:max_frames]
    ]
    return {
        "frames": frames,
        "events": events,
        "total_candidate_events_by_category": dict(sorted(Counter(
            e["event"] for e in all_events).items())),
        "total_candidate_events": len(all_events),
        "exported_review_frames": len(frames),
        "exported_candidate_events": len(events),
        "spatial_matches_in_original_fixed_cohort": len(matches),
        "review_is_not_independent_physical_truth": True,
        "unmatched_BUS_predictions_are_not_proven_false_objects": True,
    }


def audit(*, run_dir: Path, protocol_folder: Path,
          verifier=None, score_factory=None, track_reader=None,
          truth_reader=None) -> dict:
    summary_file = run_dir / "cross_date_summary.json"
    preflight_file = run_dir / "preflight.json"
    summary = read_json(summary_file)
    preflight = read_json(preflight_file)
    protocol_file = protocol_folder / "holdout_protocol.json"
    lock_file = protocol_folder / "policy_lock.json"
    protocol = read_json(protocol_file)
    lock = read_json(lock_file)
    validate_lock(protocol, lock, digest(lock_file))

    if (summary.get("status") != REGRESSION_STATUS or
        summary.get("cohort_role") != "PREVIOUSLY_USED_MAY25_REGRESSION_REFERENCE" or
        summary.get("source_window") != [60, 179] or
        summary.get("truth_frame_window") != [61, 180] or
        summary.get("decoded_frames") != 150 or summary.get("evaluation_frames") != 120 or
        summary.get("selected_policy") != lock["selected_policy"] or
        summary.get("frozen_w04_model_sha256") != lock["source_model_sha256"] or
        summary.get("not_a_blind_or_untouched_holdout") is not True or
        summary.get("independent_holdout_pass") is not False or
        summary.get("production_release_approved") is not False):
        raise AuditError("Uploaded/saved May25 report is not the original unpromoted 120-frame cohort")
    if (preflight.get("status") != "W04_MAY25_CROSSDATE_PREFLIGHT_VERIFIED_REGRESSION_ONLY" or
        preflight.get("policy_lock_sha256") != digest(lock_file) or
        preflight.get("protocol_sha256") != digest(protocol_file) or
        preflight.get("source_video_sha256") != summary["source_video_sha256"] or
        preflight.get("source_truth_sha256") != summary["original_fluid_sha256"] or
        preflight.get("model_tree_sha256") != lock["source_model_sha256"] or
        preflight.get("source_video_dimensions") != summary.get("source_video_dimensions") or
        preflight.get("source_frame_rate_reported") != summary.get("source_frame_rate_reported") or
        preflight.get("fixed_source_evaluation_frames") != [60, 179] or
        preflight.get("fixed_fluid_evaluation_frames") != [61, 180] or
        preflight.get("tracker_warmup_frames") != WARMUP):
        raise AuditError("Preflight, frozen policy lock or completed regression source disagrees")
    if verifier is None:
        from streetlab_phase3.video.holdout_validation import verify_batch
        verifier = verify_batch
    injected = any(x is not None for x in (score_factory, track_reader, truth_reader))
    verified = verifier(run_dir / "batch")
    if verified.get("original_source_sha256_verified") is not True:
        raise AuditError("Batch must SHA-verify real original May25 video and FLUID sources")
    batch = verified["report"]
    origin = verified["provenance"]
    if (verified["policy_count"] != 1 or
        len(batch.get("policies", [])) != 1 or
        batch.get("evaluated_first_frame") != WINDOW[0] or
        batch.get("evaluated_last_frame") != WINDOW[1] or
        batch.get("first_decoded_frame") != 30 or
        batch.get("warmup_frames_processed") != 30 or
        batch.get("processed_frames") != 150 or
        origin.get("source_is_independent_holdout") is not False or
        origin.get("video_sha256") != summary["source_video_sha256"] or
        origin.get("fluid_sha256") != summary["original_fluid_sha256"] or
        origin.get("openvino_model_sha256") != lock["source_model_sha256"] or
        summary.get("original_saved_batch_sha256") != verified["batch_sha256"]):
        raise AuditError("Original saved tracking batch does not match source-bound regression report")
    policy = batch["policies"][0]
    if policy["name"] != lock["selected_policy"] or policy["settings"] != lock["selected_settings"]:
        raise AuditError("Saved detector/tracker policy drifted from preregistration")
    tracks = run_dir / "batch" / policy["tracks"]
    truth_path = Path(origin["fluid_path"])
    if score_factory is None:
        from streetlab_phase3.video.continuous_box_tracking_lab import cached_window_scorer
        score_factory = cached_window_scorer
    scorer = score_factory(truth_path, 60, 179)
    pixel, identity = scorer(tracks=tracks, fluid_tracks=truth_path,
                             max_distance_px=50., start_frame=60, end_frame=179)
    class_report = scorer.class_report(tracks)
    _check_equal(pixel, policy["pixel"], "pixel_metrics")
    _check_equal(identity, policy["identity"], "identity_metrics")
    _check_equal(class_report, policy["class_diagnostics"], "class_diagnostics")
    _check_equal(summary["truth_label_precision"], pixel["point_precision"], "summary.precision")
    _check_equal(summary["truth_label_recall"], pixel["point_recall"], "summary.recall")
    f1 = 2 * pixel["point_precision"] * pixel["point_recall"] / (
        pixel["point_precision"] + pixel["point_recall"]
    )
    _check_equal(summary["truth_label_f1"], f1, "summary.f1")
    _check_equal(summary["contiguous_id_switches"],
                 identity["total_contiguous_id_switches"], "summary.identity_switches")
    _check_equal(summary["per_class_fluid_label_diagnostics"],
                 class_report["by_class"], "summary.per_class")
    _check_equal(summary["detector_cpu_median_seconds_per_frame"],
                 batch["detector_median_seconds"], "summary.detector_cpu")
    _check_equal(summary["tracker_cpu_median_seconds_per_frame"],
                 policy["tracking_median_seconds"], "summary.tracker_cpu")
    _check_equal(summary["postprocess_cpu_median_seconds_per_frame"],
                 policy["postprocess_median_seconds"], "summary.postprocess_cpu")

    if truth_reader is None:
        from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
        with truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
            truth = normalize_fluid_pixel_truth(csv.DictReader(handle))
    else:
        truth = truth_reader(truth_path)
    if track_reader is None:
        from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
        predicted = load_geotrax_pixel_tracks(tracks)
    else:
        predicted = track_reader(tracks)
    review = review_candidates(predicted, truth)
    if review["spatial_matches_in_original_fixed_cohort"] != class_report["matched_points"]:
        raise AuditError("Review queue Hungarian matches differ from frozen class report")

    selected_dev = protocol["selected_development_metrics"]
    dev_p = selected_dev["precision"]
    dev_r = selected_dev["recall"]
    dev_f1 = selected_dev["f1"]
    w04_frames = 201
    if (not all(type(v) in (int, float) and 0 <= v <= 1
                for v in (dev_p, dev_r, dev_f1)) or
        selected_dev["policy"] != lock["selected_policy"] or
        selected_dev["contiguous_id_switches"] < 0):
        raise AuditError("Cannot compare against authentic frozen development metrics")
    delta = {
        "precision_percentage_points": 100 * (pixel["point_precision"] - dev_p),
        "recall_percentage_points": 100 * (pixel["point_recall"] - dev_r),
        "f1_percentage_points": 100 * (f1 - dev_f1),
    }
    classes = {}
    for category in TRUTH_CLASSES:
        source = class_report["by_class"][category]
        baseline = selected_dev["class_diagnostics"][category]
        current = source["correct_class_recall"]
        original = baseline["correct_class_recall"]
        classes[category] = {
            "May25_correct_class_recall": current,
            "W04_development_correct_class_recall": original,
            "difference_percentage_points": (
                100 * (current-original) if current is not None and original is not None else None),
            "May25_FLUID_truth_points": source["truth_points"],
            "W04_FLUID_truth_points": baseline["truth_points"],
        }
    stage_medians = sum(summary[k] for k in (
        "detector_cpu_median_seconds_per_frame",
        "tracker_cpu_median_seconds_per_frame",
        "postprocess_cpu_median_seconds_per_frame",
    ))
    if injected:
        audit_status = "SYNTHETIC_FORensics_TEST_NOT_AUTHENTIC_VIDEO_AUDIT"
    else:
        audit_status = AUDIT_STATUS
    return {
        "schema_version": 1, "status": audit_status,
        "selected_policy": lock["selected_policy"],
        "saved_regression_summary_sha256": digest(summary_file),
        "saved_regression_batch_sha256": verified["batch_sha256"],
        "preflight_sha256": digest(preflight_file),
        "policy_lock_sha256": digest(lock_file),
        "original_video_and_FLUID_SHA_verified_from_saved_batch": True,
        "frozen_model_SHA_verified_by_regression_preflight": not injected,
        "original_saved_scores_recomputed_from_tracks_and_FLUID": not injected,
        "development_evaluation_frames": w04_frames,
        "historical_regression_evaluation_frames": 120,
        "development": {
            "precision": dev_p, "recall": dev_r, "f1": dev_f1,
            "contiguous_id_switches": selected_dev["contiguous_id_switches"],
        },
        "May25": {
            "precision": pixel["point_precision"], "recall": pixel["point_recall"],
            "f1": f1, "contiguous_id_switches": identity["total_contiguous_id_switches"],
            "source_fps": preflight["source_frame_rate_reported"],
            "source_video_dimensions": preflight["source_video_dimensions"],
            "median_stage_time_sum_seconds_per_frame": stage_medians,
            "real_end_to_end_CPU_p95_verified": False,
        },
        "difference_May25_minus_tuned_W04": delta,
        "by_class_comparison": classes,
        "confusion_truth_to_predicted": class_report["confusion_truth_to_predicted"],
        "BUS_spatial_matches_means_BUS_TRUTH_not_BUS_predictions": True,
        "BUS_unmatched_predictions_not_proven_false_objects": True,
        "review": review,
        "scientific_limitations": [
            "May25 is a previously used regression-reference recording, NOT an untouched holdout",
            "May25 120 frames and W04 May26 201 frames differ in dates, density and population",
            "F1 difference is descriptive only; no controlled same-cohort baseline or effect size",
            "Identity-switch counts cannot be compared as rates of improved identity integrity without controlling truth tracks and scene",
            "Zero BUS FLUID truth points cannot establish bus accuracy or exclude real unlabelled buses",
            "FLUID truth labels require independent two-reviewer physical-object adjudication",
            "Sum of stage medians is NOT full-pipeline end-to-end p95",
            "Native SUMO remains synthetic and Chennai held-out site transfer failed",
        ],
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reaudit saved May25 cross-date scores and export class-error candidates")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite original forensic review: {args.output_dir}")
    output = audit(run_dir=args.run_dir, protocol_folder=args.protocol_folder)
    output["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "regression_forensics.json").open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)
        file.write("\n")
    with (args.output_dir / "review_events.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(output["review"]["events"])
    print(json.dumps({
        "status": output["status"], "precision": output["May25"]["precision"],
        "recall": output["May25"]["recall"], "f1": output["May25"]["f1"],
        "review_frames": output["review"]["exported_review_frames"],
        "review_events": output["review"]["exported_candidate_events"],
        "independent_holdout_pass": False,
        "output": str(args.output_dir / "regression_forensics.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
