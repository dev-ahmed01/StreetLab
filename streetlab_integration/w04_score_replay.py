"""Replay W04 recorded track metrics against the original FLUID labels, without inference.

This is a development-only source-truth reproducibility check. It must NEVER
claim an independent held-out evaluation, real-object correctness or promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from streetlab_phase3.video.holdout_validation import sha, verify_batch


class ReplayError(ValueError):
    """Frozen evidence unavailable, altered, or inconsistent with score replay."""


def _load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ReplayError(f"Cannot parse evidence JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReplayError(f"Expected evidence object: {path}")
    return payload


def _same(expected: object, actual: object, where: str) -> None:
    if expected is None or actual is None:
        if expected is not actual:
            raise ReplayError(f"Mismatch: {where} expected={expected} actual={actual}")
        return
    if type(expected) in (float, int) and type(actual) in (float, int):
        if not (math.isfinite(expected) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)):
            raise ReplayError(f"Frozen metric differs from original-label replay: {where}")
        return
    if expected != actual:
        raise ReplayError(f"Frozen metric differs from original-label replay: {where}")


def _compare(saved: dict, replayed: dict, keys: tuple[str, ...], where: str) -> int:
    if not isinstance(saved, dict) or not isinstance(replayed, dict):
        raise ReplayError(f"Missing metric dictionary: {where}")
    count = 0
    for key in keys:
        if key in saved:
            if key not in replayed:
                raise ReplayError(f"Replayed metric unavailable: {where}.{key}")
            _same(saved[key], replayed[key], where+"."+key)
            count += 1
    return count


PIXEL = ("frame_offset", "max_distance_px", "evaluation_frame_start",
         "evaluation_frame_end", "truth_points", "predicted_points",
         "matched_points", "point_precision", "point_recall")
IDENTITY = ("frame_offset", "max_distance_px", "truth_tracks", "predicted_tracks",
            "matched_truth_tracks", "fraction_truth_tracks_fragmented",
            "total_contiguous_id_switches")
CLASS = ("truth_points", "predicted_points", "spatial_matches",
         "correct_class_matches", "spatial_misses", "class_errors_on_spatial_matches",
         "unmatched_predictions", "spatial_recall", "correct_class_recall",
         "spatial_precision", "correct_class_precision")


def replay(*, batch_dir: Path, readiness_file: Path, scorer_factory=None,
           verified_batch=None) -> dict:
    """Audit all saved W04 policy labels with an authentic or injected scorer.

    scorer_factory/verified_batch injections are for synthetic unit tests only.
    Such runs are unconditionally disallowed from claiming authentic replay.
    """
    receipt = _load(readiness_file)
    if (receipt.get("status") != "W04_LOCAL_DEVELOPMENT_ASSETS_VERIFIED"
        or receipt.get("source_video_and_model_verified") is not True
        or receipt.get("matching_fluid_truth_verified") is not True
        or receipt.get("independent_holdout_pass") is not False
        or receipt.get("eligible_for_production") is not False):
        raise ReplayError("Readiness receipt does not verify original W04 development assets")
    files = receipt.get("files", {})
    origin_file = batch_dir / "source_provenance.json"
    recorded = files.get("provenance", {}).get("sha256")
    if recorded is None or not origin_file.is_file() or sha(origin_file) != recorded:
        raise ReplayError("Readiness receipt/provenance content hash mismatch")
    verified = verified_batch if verified_batch is not None else verify_batch(batch_dir)
    if verified.get("original_source_sha256_verified") is not True:
        raise ReplayError("Build 3 saved batch original source hashes not verified")
    report = verified["report"]
    provenance = verified["provenance"]
    if (not isinstance(provenance, dict) or
        report.get("evaluated_first_frame") != 10750 or
        report.get("evaluated_last_frame") != 10950 or
        report.get("warmup_frames_processed") != 90 or
        provenance.get("video_frame_window") != [10750, 10950]):
        raise ReplayError("W04 development cohort is not the frozen 201-frame window")
    for kind, hash_field in (("original_video", "video_sha256"),
                             ("openvino_model", "openvino_model_sha256"),
                             ("fluid_truth", "fluid_sha256")):
        state = files.get(kind, {})
        if state.get("sha256_verified") is not True or state.get("sha256") != provenance.get(hash_field):
            raise ReplayError("Readiness source does not match Build 3 provenance: "+kind)
    fluid = Path(provenance["fluid_path"])
    if not fluid.is_file() or sha(fluid) != provenance.get("fluid_sha256"):
        raise ReplayError("FLUID original label source missing or modified")
    truth_injected = scorer_factory is not None or verified_batch is not None
    if scorer_factory is None:
        from streetlab_phase3.video.continuous_box_tracking_lab import cached_window_scorer
        scorer_factory = cached_window_scorer
    scorer = scorer_factory(fluid, 10750, 10950)
    if not callable(scorer) or not callable(getattr(scorer, "class_report", None)):
        raise ReplayError("Original label scorer does not support class diagnostics")
    results = []
    for policy in report.get("policies", []):
        name = policy["name"]
        tracks = batch_dir / policy["tracks"]
        pix, ident = scorer(tracks=tracks, fluid_tracks=fluid, max_distance_px=50.,
                            start_frame=10750, end_frame=10950)
        check_count = _compare(policy["pixel"], pix, PIXEL, name+".pixel")
        check_count += _compare(policy["identity"], ident, IDENTITY, name+".identity")
        if not {"point_precision", "point_recall", "truth_points"}.issubset(policy["pixel"]):
            raise ReplayError("Policy missing core pixel score: "+name)
        if "total_contiguous_id_switches" not in policy["identity"]:
            raise ReplayError("Policy missing identity switch count: "+name)
        cls = scorer.class_report(tracks)
        old = policy.get("class_diagnostics")
        if not isinstance(old, dict) or not isinstance(cls, dict):
            raise ReplayError("Class comparison unavailable: "+name)
        check_count += _compare(old, cls, ("frame_offset", "max_pixel_distance",
                                           "truth_points", "predicted_points",
                                           "matched_points", "correct_class_matches"), name+".classes")
        for family in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE"):
            old_family = old.get("by_class", {}).get(family)
            new_family = cls.get("by_class", {}).get(family)
            if not isinstance(old_family, dict) or not isinstance(new_family, dict):
                raise ReplayError("Missing per-class report: "+name+"."+family)
            check_count += _compare(old_family, new_family, CLASS, name+"."+family)
        results.append({
            "policy": name, "track_sha256": sha(tracks), "verified_metric_fields": check_count,
            "previously_saved_precision": policy["pixel"]["point_precision"],
            "previously_saved_recall": policy["pixel"]["point_recall"],
            "previously_saved_identity_switches": policy["identity"]["total_contiguous_id_switches"],
        })
    if len(results) != verified["policy_count"] or len(results) == 0:
        raise ReplayError("Some frozen tracking policies were not replayed")
    return {
        "schema_version": 1,
        "status": ("W04_SYNTHETIC_TEST_INJECTION_NOT_SOURCE_REPLAY" if truth_injected
                   else "W04_DEVELOPMENT_SCORE_REPLAY_MATCHED_ORIGINAL_FLUID"),
        "scope": "ORIGINAL_W04_TUNED_DEVELOPMENT_LABEL_REPLAY_ONLY",
        "readiness_file_sha256": sha(readiness_file),
        "original_batch_sha256": verified["batch_sha256"],
        "original_fluid_sha256": sha(fluid),
        "source_window": [10750, 10950],
        "evaluation_fluid_frame_window": [10751, 10951],
        "policy_count": len(results),
        "scorecards": results,
        "source_video_model_truth_verified_against_frozen_provenance": True,
        "same_window_t000_available": report.get("same_window_T000") is not None,
        "reuses_previously_evaluated_development_window": True,
        "independent_heldout_accuracy_claimed": False,
        "new_detector_inference_performed": False,
        "physical_object_labels_independently_adjudicated": False,
        "production_release_approved": False,
        "next_step": "Preregister one policy, then measure truly untouched different source video with same-cohort baseline, independent reviewer adjudication, and p95 runtime.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay saved W04 scores, not model inference")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file only")
    args = parser.parse_args(argv)
    outcome = replay(batch_dir=args.batch_dir, readiness_file=args.readiness)
    outcome["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(outcome, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": outcome["status"],
                      "policies": outcome["policy_count"],
                      "same_window_t000_available": outcome["same_window_t000_available"],
                      "independent_heldout_accuracy_claimed": False,
                      "production_release_approved": False,
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
