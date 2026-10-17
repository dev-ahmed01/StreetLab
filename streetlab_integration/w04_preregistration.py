"""Bind the verified W04 development replay to ONE preregistered holdout policy.

No unseen footage is assessed here. This is a source-bound method decision and
research protocol, not an independent performance result or release approval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from streetlab_phase3.video.holdout_validation import (
    LOCK_STATUS, PASS_LIMITS, lock_policy, sha, verify_batch,
)

STATUS = "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED"
DEFAULT_POLICY = "hard_nms_ios_0.30"
ALLOWED_WINDOW = [10750, 10950]
FLUID_WINDOW = [10751, 10951]


class ProtocolError(ValueError):
    """The supplied historical evidence cannot justify a prospective lock."""


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"Expected original evidence file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeError) as exc:
        raise ProtocolError(f"Invalid evidence JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("Evidence must have a JSON object root")
    return data


def _finite(x: object, name: str, *, minimum=0., maximum=None) -> float:
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ProtocolError(f"Invalid finite numerical evidence: {name}")
    value = float(x)
    if value < minimum or maximum is not None and value > maximum:
        raise ProtocolError(f"Out-of-range numerical evidence: {name}")
    return value


def _expected_roles(studies: list[dict]) -> list[dict]:
    result = []
    for study in studies:
        result.append({
            "study_id": study.get("study_id"),
            "manifest_role": study.get("role"),
            "established_unseen_for_this_preregistration": False,
        })
    return result


def assess(replay: dict, verified: dict, *, replay_sha: str, policy: str,
           cpu_budget_s: float, manifest: dict) -> dict:
    """Validate every saved replay policy against original frozen batch data."""
    budget = _finite(cpu_budget_s, "prospective CPU budget", minimum=0.001, maximum=60.)
    if (replay.get("status") != "W04_DEVELOPMENT_SCORE_REPLAY_MATCHED_ORIGINAL_FLUID"
        or replay.get("scope") != "ORIGINAL_W04_TUNED_DEVELOPMENT_LABEL_REPLAY_ONLY"
        or replay.get("source_window") != ALLOWED_WINDOW
        or replay.get("evaluation_fluid_frame_window") != FLUID_WINDOW
        or replay.get("source_video_model_truth_verified_against_frozen_provenance") is not True
        or replay.get("reuses_previously_evaluated_development_window") is not True
        or replay.get("same_window_t000_available") is not False
        or replay.get("independent_heldout_accuracy_claimed") is not False
        or replay.get("new_detector_inference_performed") is not False
        or replay.get("physical_object_labels_independently_adjudicated") is not False
        or replay.get("production_release_approved") is not False):
        raise ProtocolError("W04 score replay is not an authenticated development-only result")
    if verified.get("original_source_sha256_verified") is not True:
        raise ProtocolError("Original source hashes were not independently checked")
    batch = verified.get("report", {})
    provenance = verified.get("provenance", {})
    if (not isinstance(provenance, dict)
        or provenance.get("eligible_for_promotion") is not False
        or batch.get("eligible_for_promotion") is not False
        or batch.get("evaluated_first_frame") != 10750
        or batch.get("evaluated_last_frame") != 10950
        or batch.get("warmup_frames_processed") != 90
        or batch.get("same_window_T000") is not None
        or replay.get("original_batch_sha256") != verified.get("batch_sha256")
        or replay.get("original_fluid_sha256") != provenance.get("fluid_sha256")):
        raise ProtocolError("W04 replay is inconsistent with frozen original batch/provenance")
    original = batch.get("policies", [])
    reported = replay.get("scorecards", [])
    if (type(verified.get("policy_count")) is not int
        or verified["policy_count"] != 25
        or replay.get("policy_count") != 25
        or len(original) != 25
        or not isinstance(reported, list) or len(reported) != 25):
        raise ProtocolError("Expected exactly the original 25 W04 development policies")
    indexed = {}
    for entry in reported:
        if not isinstance(entry, dict) or not isinstance(entry.get("policy"), str):
            raise ProtocolError("Malformed replay policy entry")
        if entry["policy"] in indexed:
            raise ProtocolError("Duplicate replay policy")
        indexed[entry["policy"]] = entry
    if set(indexed) != {p.get("name") for p in original}:
        raise ProtocolError("Missing or substituted W04 policy in replay")

    candidates = []
    hash_groups: dict[str, list[str]] = defaultdict(list)
    selected = None
    raw = None
    for item in original:
        name = item["name"]
        saved = indexed[name]
        p = _finite(item["pixel"].get("point_precision"), name+".precision", maximum=1)
        r = _finite(item["pixel"].get("point_recall"), name+".recall", maximum=1)
        switches = _finite(item["identity"].get("total_contiguous_id_switches"),
                           name+".switches")
        if (saved.get("track_sha256") != item.get("tracks_sha256")
            or saved.get("policy") != name
            or type(saved.get("verified_metric_fields")) is not int
            or saved["verified_metric_fields"] < 50
            or type(saved.get("previously_saved_precision")) not in (int, float)
            or type(saved.get("previously_saved_recall")) not in (int, float)
            or type(saved.get("previously_saved_identity_switches")) not in (int, float)
            or not math.isclose(saved["previously_saved_precision"], p, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(saved["previously_saved_recall"], r, rel_tol=1e-12, abs_tol=1e-12)
            or saved["previously_saved_identity_switches"] != switches):
            raise ProtocolError(f"Track SHA or replayed metric disagreement for {name}")
        f1 = 2*p*r/(p+r) if p+r else 0.
        diagnostics = item.get("class_diagnostics", {}).get("by_class", {})
        if not isinstance(diagnostics, dict):
            raise ProtocolError("Class diagnostics missing from frozen W04 source")
        classes = {}
        for label in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE"):
            detail = diagnostics.get(label)
            if not isinstance(detail, dict):
                raise ProtocolError(f"Missing {label} class diagnostics for {name}")
            n = detail.get("truth_points")
            if type(n) is not int or n < 0:
                raise ProtocolError(f"Invalid {label} FLUID truth count in {name}")
            val = detail.get("correct_class_recall")
            classes[label] = {"truth_points": n,
                              "correct_class_recall": (
                                _finite(val, name+"."+label, maximum=1.) if n else None)}
            if n and val is None:
                raise ProtocolError(f"Recall absent with {label} truth points in {name}")
        one = {"policy": name, "precision": p, "recall": r, "f1": f1,
               "contiguous_id_switches": int(switches), "class_diagnostics": classes,
               "track_sha256": saved["track_sha256"]}
        candidates.append(one)
        hash_groups[one["track_sha256"]].append(name)
        if name == policy:
            selected = one
        if name == "raw_unmerged":
            raw = one
    if selected is None or raw is None:
        raise ProtocolError("Selected policy or raw-unmerged comparison missing")
    ranked = sorted(candidates, key=lambda x: (-x["f1"], x["contiguous_id_switches"], x["policy"]))
    if ranked[0]["policy"] != DEFAULT_POLICY or policy != DEFAULT_POLICY:
        raise ProtocolError("This preregistration is pinned to documented W04 leader hard_nms_ios_0.30")
    class_tradeoffs = {}
    for category, item in selected["class_diagnostics"].items():
        base = raw["class_diagnostics"][category]
        a, b = item["correct_class_recall"], base["correct_class_recall"]
        class_tradeoffs[category] = {
            "truth_points_in_W04": item["truth_points"],
            "selected_correct_class_recall": a,
            "raw_unmerged_correct_class_recall": b,
            "selected_minus_raw_percentage_points": (100*(a-b) if a is not None and b is not None else None),
            "interpretation": "NOT_MEASURABLE_ZERO_TRUTH" if item["truth_points"] == 0
                               else "FLUID_LABEL_DEVELOPMENT_ONLY",
        }
    if not isinstance(manifest.get("studies"), list):
        raise ProtocolError("Frozen FLUID study manifest unavailable")
    expected_roles = {s.get("study_id"): s.get("role") for s in manifest["studies"]}
    if (expected_roles.get("fluid_fidrt_20250526") != "held_out_validation"
        or expected_roles.get("fluid_fidrt_20250120") != "development_calibration"
        or expected_roles.get("fluid_fidrt_20250525") != "regression_reference"):
        raise ProtocolError("FLUID study manifest differs from historical locked roles")

    return {
        "schema_version": 1,
        "status": STATUS,
        "research_scope": "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY",
        "replay_sha256": replay_sha,
        "development_batch_sha256": verified["batch_sha256"],
        "source_video_sha256": provenance["video_sha256"],
        "source_fluid_sha256": provenance["fluid_sha256"],
        "source_model_sha256": provenance["openvino_model_sha256"],
        "selected_policy": policy,
        "selection_rule": "MAX_DEVELOPMENT_SPATIAL_F1_WITH_ID_SWITCHES_AS_TIE_BREAKER",
        "selected_development_metrics": selected,
        "development_leaderboard_top_five": [
            {k: item[k] for k in ("policy", "precision", "recall", "f1", "contiguous_id_switches")}
            for item in ranked[:5]
        ],
        "raw_unmerged_reference": {k: raw[k] for k in ("policy", "precision", "recall", "f1", "contiguous_id_switches")},
        "correct_class_tradeoffs_vs_raw": class_tradeoffs,
        "duplicate_track_byte_groups": [
            {"policies": names, "shared_track_sha256": digest}
            for digest, names in sorted(hash_groups.items()) if len(names) > 1
        ],
        "prospective_cpu_stage_median_budget_seconds_per_frame": budget,
        "cpu_budget_is_predeclared_requirement_not_measured_p95": True,
        "predeclared_numeric_gates": {**PASS_LIMITS, "max_cpu_seconds_per_frame": budget},
        "original_source_study_roles": _expected_roles(manifest["studies"]),
        "new_source_established_untouched": False,
        "same_window_t000_baseline_available": False,
        "two_reviewer_physical_labels_verified": False,
        "end_to_end_cpu_p95_measured": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "remaining_requirements": [
            "Find evidence for genuinely untouched and different camera/video; historical manifest roles alone do not establish it",
            "Create frozen same-window reference tracking; label a replacement baseline honestly if original T000 is missing",
            "Run identical frozen policy on untouched continuous frames and independently replay all FLUID-label scores",
            "Check motorcycle/heavy/BUS physical-class accuracy through blinded two-reviewer adjudication",
            "Measure end-to-end CPU p95 and retain independent external review",
            "Do not change W04, Chennai, source video, FLUID labels or frozen model after this lock",
        ],
    }


def prepare(*, batch_dir: Path, replay_file: Path, output_dir: Path,
            project_root: Path, policy: str = DEFAULT_POLICY,
            max_cpu_seconds_per_frame: float = 2.0) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing protocol: {output_dir}")
    replay = _read(replay_file)
    manifest = _read(project_root / "data/benchmarks/fluid_fidrt/manifest.json")
    verified = verify_batch(batch_dir)
    protocol = assess(replay, verified, replay_sha=sha(replay_file),
                      policy=policy, cpu_budget_s=max_cpu_seconds_per_frame,
                      manifest=manifest)
    # Avoid generating a lock whose future holdout cannot be evaluated due to
    # a known dev stage-median that already exceeds the proposed CPU limit.
    selected_item = next(p for p in verified["report"]["policies"] if p["name"] == policy)
    stage_sum = sum(_finite(v, k) for k, v in (
        ("detector", verified["report"]["detector_median_seconds"]),
        ("postprocess", selected_item["postprocess_median_seconds"]),
        ("tracking", selected_item["tracking_median_seconds"]),
    ))
    protocol["development_sum_of_stage_medians_seconds_per_frame"] = stage_sum
    protocol["sum_of_stage_medians_is_not_end_to_end_p95"] = True
    if stage_sum > max_cpu_seconds_per_frame:
        raise ProtocolError("Proposed CPU budget is already below W04 sum of stage medians")
    output_dir.mkdir(parents=True, exist_ok=False)
    # Reuse the existing Build 4 authoritative lock contract unmodified.
    lock_path = output_dir / "policy_lock.json"
    lock = lock_policy(batch_dir, policy, lock_path,
                       max_cpu_seconds_per_frame=max_cpu_seconds_per_frame)
    if lock.get("status") != LOCK_STATUS:
        raise ProtocolError("Phase 3 policy lock failed")
    protocol["policy_lock_sha256"] = sha(lock_path)
    protocol["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (output_dir / "holdout_protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze W04 policy/prospective holdout protocol, not holdout results")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--max-cpu-seconds-per-frame", type=float, default=2.0)
    args = parser.parse_args(argv)
    result = prepare(batch_dir=args.batch_dir, replay_file=args.replay,
                     output_dir=args.output_dir, project_root=args.project_root,
                     policy=args.policy, max_cpu_seconds_per_frame=args.max_cpu_seconds_per_frame)
    print(json.dumps({
        "status": result["status"],
        "selected_policy": result["selected_policy"],
        "development_f1": result["selected_development_metrics"]["f1"],
        "new_source_established_untouched": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "policy_lock": str(args.output_dir / "policy_lock.json"),
        "protocol": str(args.output_dir / "holdout_protocol.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
