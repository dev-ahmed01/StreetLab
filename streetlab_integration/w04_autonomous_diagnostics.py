"""Source-verified automatic W04 May25 object/track error attribution.

Read frozen real-video outputs. Never rerun inference, change model/policy, relabel
FLUID truth, claim physical vehicle identities, or assert independent holdout.
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

STATUS = "W04_MAY25_AUTOMATED_LABEL_RELATIVE_OBJECT_AUDIT_COMPLETE"
CLASSES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
TRACK_COLUMNS = ("truth_track_id", "truth_class", "truth_points", "matched_points",
                 "spatial_misses", "correct_class_matches", "wrong_class_matches",
                 "dominant_predicted_class", "dominant_predicted_class_fraction",
                 "predicted_classes", "matched_predicted_track_ids",
                 "first_fluid_frame", "last_fluid_frame",
                 "mean_matching_distance_px", "close_competing_match_count")


class AuditError(ValueError):
    """Failed original source, frozen-metric, or object-level conservation gate."""


def _ratio(top: int, bottom: int) -> float | None:
    return top / bottom if bottom else None


def _top(counter: Counter[str]) -> str | None:
    return sorted(counter, key=lambda k: (-counter[k], str(k)))[0] if counter else None


def _distance(a: Any, b: Any) -> float:
    return math.hypot(a.x_px - b.x_px, a.y_px - b.y_px)


def analyze_points(predicted: Sequence[Any], truth: Sequence[Any], *,
                   width: int, height: int, start: int = 60, end: int = 179,
                   offset: int = 1, gate: float = 50.) -> dict:
    """Reuse the SAME 50px class-agnostic Hungarian matcher on the whole cohort."""
    if (start, end, offset, gate) != (60, 179, 1, 50.):
        raise AuditError("Frozen May25 120 frames, +1 alignment and 50px gate required")
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise AuditError("Original native dimensions required")

    from streetlab_phase3.pixel_benchmark import PixelBenchmark
    from streetlab_phase3.video.class_diagnostics import class_diagnostics
    predictions = [p for p in predicted
                   if start <= p.frame <= end and p.vehicle_class in CLASSES]
    truths = [t for t in truth
              if start+offset <= t.frame <= end+offset and t.vehicle_class in CLASSES]
    if not predictions or not truths:
        raise AuditError("Missing frozen May25 prediction tracks or matching FLUID points")
    for item in (*predictions, *truths):
        if not (math.isfinite(item.x_px) and math.isfinite(item.y_px)):
            raise AuditError("FLUID or prediction center is nonfinite")
    # Preserve the original scorer even if a predicted track center drifts
    # just outside the source image. Report it as an objective defect rather
    # than silently filtering it and changing the frozen class score.
    pred_out_of_bounds = sum(
        not (0 <= p.x_px < width and 0 <= p.y_px < height)
        for p in predictions
    )
    if any(not (0 <= t.x_px < width and 0 <= t.y_px < height)
           for t in truths):
        raise AuditError("Original FLUID center outside preflight-verified native image")

    matches = PixelBenchmark(gate).match_points(predictions, truths, frame_offset=offset)
    diagnostic = class_diagnostics(predictions, truths,
                                   start_frame=start, end_frame=end)
    if len(matches) != diagnostic["matched_points"]:
        raise AuditError("Match count differs from frozen class diagnostics")

    by_truth: dict[str, list[Any]] = defaultdict(list)
    by_prediction: dict[str, list[Any]] = defaultdict(list)
    truth_frame: dict[int, list[Any]] = defaultdict(list)
    pred_frame: dict[int, list[Any]] = defaultdict(list)
    for t in truths:
        by_truth[t.track_id].append(t)
        truth_frame[t.frame].append(t)
    for p in predictions:
        by_prediction[p.track_id].append(p)
        pred_frame[p.frame].append(p)

    matched_truth: dict[str, list[tuple[Any, Any, float]]] = defaultdict(list)
    matched_pred: dict[str, list[tuple[Any, Any, float]]] = defaultdict(list)
    confusing_pairs = Counter()
    competing_truth_matches = Counter()
    ambiguity_matches = 0
    ambiguity_wrong_matches = 0
    ambiguity_heavy_wrong = 0
    for p, t, distance in matches:
        matched_truth[t.track_id].append((p, t, distance))
        matched_pred[p.track_id].append((p, t, distance))
        alternate = (
            [_distance(p, t2) for t2 in truth_frame[t.frame] if t2 is not t] +
            [_distance(p2, t) for p2 in pred_frame[p.frame] if p2 is not p]
        )
        close_competitor = any(
            candidate <= gate and candidate - distance <= 10.
            for candidate in alternate
        )
        if close_competitor:
            ambiguity_matches += 1
            competing_truth_matches[t.track_id] += 1
        if p.vehicle_class != t.vehicle_class:
            confusing_pairs[(t.vehicle_class, p.vehicle_class)] += 1
            if close_competitor:
                ambiguity_wrong_matches += 1
                if t.vehicle_class == "HEAVY_VEHICLE":
                    ambiguity_heavy_wrong += 1

    truth_rows = []
    for track_id, points in by_truth.items():
        observations = matched_truth[track_id]
        classes = Counter(p.vehicle_class for p, _, _ in observations)
        pred_ids = Counter(p.track_id for p, _, _ in observations)
        dominant = _top(classes)
        correct = sum(p.vehicle_class == t.vehicle_class
                      for p, t, _ in observations)
        truth_rows.append({
            "truth_track_id": track_id,
            "truth_class": points[0].vehicle_class,
            "truth_points": len(points),
            "matched_points": len(observations),
            "spatial_misses": len(points) - len(observations),
            "correct_class_matches": correct,
            "wrong_class_matches": len(observations) - correct,
            "dominant_predicted_class": dominant,
            "dominant_predicted_class_fraction": (
                _ratio(classes[dominant], len(observations)) if dominant else None),
            "predicted_classes": dict(sorted(classes.items())),
            "matched_predicted_track_ids": dict(sorted(pred_ids.items())),
            "first_fluid_frame": min(t.frame for t in points),
            "last_fluid_frame": max(t.frame for t in points),
            "mean_matching_distance_px": (
                sum(d for _, _, d in observations)/len(observations)
                if observations else None),
            "close_competing_match_count": competing_truth_matches[track_id],
        })
    truth_rows.sort(key=lambda x: (-x["wrong_class_matches"],
                                    -x["spatial_misses"], str(x["truth_track_id"])))
    heavy = [r for r in truth_rows if r["truth_class"] == "HEAVY_VEHICLE"]
    heavy_errors = [r for r in heavy if r["wrong_class_matches"]]
    heavy_wrong = sum(r["wrong_class_matches"] for r in heavy_errors)
    if heavy_wrong != diagnostic["by_class"]["HEAVY_VEHICLE"]["class_errors_on_spatial_matches"]:
        raise AuditError("HEAVY class-count conservation failed")

    bus_predicted = []
    stability = []
    for track_id, points in by_prediction.items():
        counts = Counter(p.vehicle_class for p in points)
        dominant = _top(counts)
        matched_truth_ids = Counter(t.track_id for _, t, _ in matched_pred[track_id])
        stability.append({
            "predicted_track_id": track_id,
            "frames": len(points),
            "dominant_class": dominant,
            "dominant_fraction": _ratio(counts[dominant], len(points)),
            "predicted_classes": dict(sorted(counts.items())),
            "matched_FLUID_truth_track_ids": dict(sorted(matched_truth_ids.items())),
            "matches_to_multiple_FLUID_tracks": len(matched_truth_ids) > 1,
        })
        if counts["BUS"]:
            matched_bus = sum(
                p.vehicle_class == "BUS" for p, _, _ in matched_pred[track_id])
            bus_frames = [p.frame for p in points if p.vehicle_class == "BUS"]
            bus_predicted.append({
                "predicted_track_id": track_id,
                "BUS_prediction_frames": counts["BUS"],
                "BUS_matched_any_FLUID_class": matched_bus,
                "BUS_unmatched_FLUID_frames": counts["BUS"] - matched_bus,
                "first_source_frame": min(bus_frames),
                "last_source_frame": max(bus_frames),
            })
    stability.sort(key=lambda r: (-r["frames"], str(r["predicted_track_id"])))
    bus_predicted.sort(key=lambda r: (-r["BUS_unmatched_FLUID_frames"],
                                      str(r["predicted_track_id"])))

    expected_confusion = Counter({
        (row["truth"], row["predicted"]): row["points"]
        for row in diagnostic["confusion_truth_to_predicted"]})
    if (confusing_pairs != expected_confusion
        or sum(r["truth_points"] for r in truth_rows) != len(truths)
        or sum(r["matched_points"] for r in truth_rows) != len(matches)):
        raise AuditError("Recomputed object-level accounting does not reproduce original cohort")

    return {
        "status": STATUS,
        "source_video_frame_window": [start, end],
        "FLUID_frame_window": [start+1, end+1],
        "video_dimensions_original_pixels": [width, height],
        "truth_points": len(truths),
        "predicted_points": len(predictions),
        "predicted_points_outside_native_image_not_filtered": pred_out_of_bounds,
        "matched_points": len(matches),
        "unique_FLUID_truth_tracks": len(by_truth),
        "unique_predicted_tracks": len(by_prediction),
        "recomputed_class_diagnostics": diagnostic,
        "truth_track_diagnostics": truth_rows,
        "predicted_track_class_stability": stability,
        "HEAVY_truth_unique_tracks": len(heavy),
        "HEAVY_truth_tracks_with_class_disagreement": len(heavy_errors),
        "HEAVY_wrong_class_matched_points": heavy_wrong,
        "HEAVY_error_share_top_two_FLUID_tracks": (
            _ratio(sum(r["wrong_class_matches"] for r in heavy_errors[:2]), heavy_wrong)),
        "HEAVY_error_share_top_five_FLUID_tracks": (
            _ratio(sum(r["wrong_class_matches"] for r in heavy_errors[:5]), heavy_wrong)),
        "HEAVY_wrong_matches_with_close_competing_association": ambiguity_heavy_wrong,
        "total_matched_points_with_close_competing_association": ambiguity_matches,
        "wrong_class_matches_with_close_competing_association": ambiguity_wrong_matches,
        "BUS_predicted_track_diagnostics": bus_predicted,
        "BUS_unique_predicted_track_ids": len(bus_predicted),
        "BUS_FLUID_truth_points": diagnostic["by_class"]["BUS"]["truth_points"],
        "physical_object_labels_independently_adjudicated": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }


def _load_original(run_dir: Path, forensics_dir: Path,
                   original_paths: dict[str, Path] | None = None):
    from streetlab_integration.w04_visual_review_packet import check_sources
    from streetlab_integration.w04_holdout_intake import read_json, digest
    from streetlab_phase3.video.holdout_validation import verify_batch
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
    from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
    from streetlab_phase3.video.class_diagnostics import class_diagnostics
    from streetlab_integration.w04_crossdate_forensics import AUDIT_STATUS

    video, sources, events = check_sources(
        run_dir=run_dir, forensics_dir=forensics_dir,
        original_paths=original_paths,
    )
    report = read_json(forensics_dir/"regression_forensics.json")
    summary = read_json(run_dir/"cross_date_summary.json")
    if report["status"] != AUDIT_STATUS:
        raise AuditError("Source-based forensic replay must be complete")
    verified = verify_batch(run_dir/"batch", original_paths=original_paths)
    if (verified.get("original_source_sha256_verified") is not True
        or verified["batch_sha256"] != report["saved_regression_batch_sha256"]
        or verified["policy_count"] != 1):
        raise AuditError("Original full batch cannot be source-verified")
    batch = verified["report"]
    original = verified["provenance"]
    if (not original or original.get("source_is_independent_holdout") is not False
        or original.get("eligible_for_promotion") is not False
        or original.get("video_sha256") != sources["original_video_sha256"]
        or original.get("fluid_sha256") != sources["original_truth_sha256"]
        or summary.get("selected_policy") != "hard_nms_ios_0.30"
        or summary.get("independent_holdout_pass") is not False):
        raise AuditError("Frozen original sources have changed or were promoted")
    policy = batch["policies"][0]
    if policy["name"] != "hard_nms_ios_0.30":
        raise AuditError("May25 policy differs from frozen W04 selection")
    root = (run_dir/"batch").resolve()
    track_file = (root/policy["tracks"]).resolve()
    if (not track_file.is_relative_to(root)
        or track_file.is_symlink() or not track_file.is_file()):
        raise AuditError("Untrusted original track location")
    pred = load_geotrax_pixel_tracks(track_file)
    fluid_path = (Path(original_paths["fluid_path"])
                  if original_paths is not None else Path(original["fluid_path"]))
    with fluid_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        truth = normalize_fluid_pixel_truth(csv.DictReader(stream))
    original_scores = class_diagnostics(pred, truth,
                                        start_frame=60, end_frame=179)
    if (original_scores["by_class"] != summary["per_class_fluid_label_diagnostics"]
        or original_scores["confusion_truth_to_predicted"]
           != report["confusion_truth_to_predicted"]
        or original_scores["matched_points"]
           != report["review"]["spatial_matches_in_original_fixed_cohort"]):
        raise AuditError("Full-cohort class diagnostics disagree with original May25 records")
    return pred, truth, sources, summary, report, verified["batch_sha256"], digest


def findings_markdown(result: dict) -> str:
    byclass = result["recomputed_class_diagnostics"]["by_class"]
    heavy = byclass["HEAVY_VEHICLE"]
    lines = [
        "# StreetLab — autonomous W04 object-level research audit",
        "",
        "**Historical May25 regression, not a blind holdout or field validation.**",
        "",
        f"- Frozen FLUID-label spatial F1: **{result['original_saved_FLUID_label_F1']:.2%}**.",
        f"- **{result['matched_points']}** class-agnostic matches out of "
        f"**{result['truth_points']}** FLUID truth points, across "
        f"**{result['unique_FLUID_truth_tracks']}** annotated track identities.",
        f"- HEAVY_VEHICLE FLUID-label correct-class recall: **{heavy['correct_class_recall']:.2%}**.",
        f"- **{result['HEAVY_wrong_class_matched_points']}** repeated frame-level "
        f"HEAVY class disagreements arise from **{result['HEAVY_truth_tracks_with_class_disagreement']}** "
        "different FLUID track IDs.",
        f"- Top two HEAVY FLUID track identities account for "
        f"**{result['HEAVY_error_share_top_two_FLUID_tracks']:.2%}** of those disagreements.",
        f"- **{result['HEAVY_wrong_matches_with_close_competing_association']}** "
        "heavy disagreements have a nearby alternative Hungarian association "
        "(within the original 50px gate and 10px of chosen match distance).",
        f"- BUS has **{result['BUS_FLUID_truth_points']}** FLUID truth points but "
        f"**{result['BUS_unique_predicted_track_ids']}** distinct predicted BUS track IDs; "
        "physical BUS correctness is **not measurable**.",
        "",
        "## Top recurring class disagreements",
        "",
        "| FLUID track | FLUID class | Truth frames | Matched | Class errors | "
        "Dominant model class | Dominant share of matched labels |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in [r for r in result["truth_track_diagnostics"]
                if r["wrong_class_matches"]][:15]:
        share = row["dominant_predicted_class_fraction"]
        lines.append(
            f"| {row['truth_track_id']} | {row['truth_class']} | "
            f"{row['truth_points']} | {row['matched_points']} | "
            f"{row['wrong_class_matches']} | {row['dominant_predicted_class']} | "
            f"{share:.1%} |")
    lines.extend([
        "",
        "Dominant model-class share measures **track consistency**, not calibrated "
        "classifier confidence or correctness of either vehicle ontology.",
        "",
        "## Automated decision",
        "",
        "The frozen W04 model and its original FLUID labels remain unchanged. "
        "Repeated mismatch on the same physical target is **not multiple independent "
        "vehicle failures**, and model stability cannot by itself prove FLUID taxonomy "
        "is wrong. The scientific record is complete at the **historical, "
        "label-relative regression** level only.",
        "",
        "No manual review is required to run this programmatic audit. However, "
        "its results cannot supply missing independently adjudicated physical "
        "vehicle types, original same-window T000, true end-to-end CPU p95 or "
        "a genuinely untouched evaluation source.",
        "",
        "**Independent holdout: not passed. Production: not approved.** "
        "The prior Chennai field-generalization result remains failed; "
        "native SUMO experiments remain synthetic.",
        "",
    ])
    return "\n".join(lines)


def run(*, run_dir: Path, forensics_dir: Path, output_dir: Path,
        original_paths: dict[str, Path] | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite previous completed research")
    pred, truth, sources, summary, forensics, batch_sha, digest = _load_original(
        run_dir, forensics_dir, original_paths=original_paths)
    width, height = summary["source_video_dimensions"]
    result = analyze_points(pred, truth, width=width, height=height)
    if (result["HEAVY_wrong_class_matched_points"]
            != summary["per_class_fluid_label_diagnostics"]["HEAVY_VEHICLE"]["class_errors_on_spatial_matches"]):
        raise AuditError("HEAVY full-cohort source count mismatch")
    result.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_video_and_FLUID_SHA": sources,
        "original_tracking_batch_sha256": batch_sha,
        "original_saved_regression_summary_sha256": digest(run_dir/"cross_date_summary.json"),
        "original_source_forensics_sha256": digest(forensics_dir/"regression_forensics.json"),
        "original_saved_FLUID_label_F1": summary["truth_label_f1"],
        "no_new_detector_inference": True,
        "no_changes_to_model_or_tracking_policy": True,
        "no_FLUID_relabeling": True,
        "method": "Original frozen +1 frame/50px class-agnostic Hungarian, all 120 frames; object-identity aggregation",
    })
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir/"automated_object_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output_dir/"all_truth_track_diagnostics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACK_COLUMNS)
        writer.writeheader()
        for row in result["truth_track_diagnostics"]:
            writer.writerow({
                **row,
                "predicted_classes": json.dumps(row["predicted_classes"], sort_keys=True),
                "matched_predicted_track_ids": json.dumps(
                    row["matched_predicted_track_ids"], sort_keys=True),
            })
    with (output_dir/"predicted_track_class_stability.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        fields = ("predicted_track_id", "frames", "dominant_class",
                  "dominant_fraction", "predicted_classes",
                  "matched_FLUID_truth_track_ids",
                  "matches_to_multiple_FLUID_tracks")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["predicted_track_class_stability"]:
            writer.writerow({
                **row, "predicted_classes": json.dumps(row["predicted_classes"]),
                "matched_FLUID_truth_track_ids": json.dumps(
                    row["matched_FLUID_truth_track_ids"]),
            })
    with (output_dir/"research_findings.md").open("x", encoding="utf-8") as stream:
        stream.write(findings_markdown(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous May25 object-level source-verified scientific audit")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--forensics-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-video", type=Path,
                        help="Drive-staged original 20250525_video.mp4; bytes must match frozen SHA")
    parser.add_argument("--source-truth", type=Path,
                        help="Drive-staged original 20250525_video_Traj.csv; bytes must match frozen SHA")
    args = parser.parse_args(argv)
    if (args.source_video is None) != (args.source_truth is None):
        parser.error("--source-video and --source-truth must be provided together")
    source_paths = ({"video_path": args.source_video,
                     "fluid_path": args.source_truth}
                    if args.source_video is not None else None)
    result = run(run_dir=args.run_dir, forensics_dir=args.forensics_dir,
                 output_dir=args.output_dir, original_paths=source_paths)
    print(json.dumps({
        "status": result["status"],
        "heavy_error_FLUID_tracks": result["HEAVY_truth_tracks_with_class_disagreement"],
        "heavy_error_frame_matches": result["HEAVY_wrong_class_matched_points"],
        "top_2_track_error_share": result["HEAVY_error_share_top_two_FLUID_tracks"],
        "inference_repeated": False,
        "independent_holdout_pass": False,
        "research_report": str(args.output_dir/"research_findings.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
