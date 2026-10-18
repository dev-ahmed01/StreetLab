"""Read-only intake for a prospective W04 video holdout, not a holdout score.

Even a different SHA-256 cannot prove a video was unseen. An independent
chronology/camera audit, same-cohort reference and human labels remain external.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED"
LOCK = "PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT"
KNOWN = {
    "20250526_video.mp4": "W04_TUNED_DEVELOPMENT",
    "20250525_video.mp4": "REGRESSION_REFERENCE",
    "20250120_video.mp4": "DEVELOPMENT_CALIBRATION",
}
BASELINES = ("none", "original_t000", "new_replacement")


class IntakeError(ValueError):
    """Fail closed for mismatched, forged or incomplete preregistration."""


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise IntakeError(f"Missing or linked preregistration evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise IntakeError("Unreadable preregistration JSON") from exc
    if not isinstance(value, dict):
        raise IntakeError("Preregistration root must be an object")
    return value


def validate_lock(protocol: dict, lock: dict, lock_sha: str) -> None:
    if (protocol.get("schema_version") != 1
        or protocol.get("status") != PROTOCOL
        or protocol.get("research_scope") != "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY"
        or protocol.get("policy_lock_sha256") != lock_sha
        or lock.get("status") != LOCK
        or lock.get("selection_scope") != "DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION"
        or lock.get("eligible_for_production") is not False
        or protocol.get("selected_policy") != lock.get("selected_policy")
        or lock.get("selected_policy") != "hard_nms_ios_0.30"
        or protocol.get("development_batch_sha256") != lock.get("source_development_batch_sha256")
        or protocol.get("source_video_sha256") != lock.get("source_development_video_sha256")
        or protocol.get("source_fluid_sha256") != lock.get("source_fluid_sha256")
        or protocol.get("source_model_sha256") != lock.get("source_model_sha256")
        or protocol.get("predeclared_numeric_gates") != lock.get("predeclared_limits")
        or protocol.get("new_source_established_untouched") is not False
        or protocol.get("same_window_t000_baseline_available") is not False
        or protocol.get("independent_holdout_pass") is not False
        or protocol.get("production_release_approved") is not False):
        raise IntakeError("Policy and protocol must remain the original aligned, unpromoted lock")
    settings = lock.get("selected_settings")
    if (not isinstance(settings, dict)
        or settings.get("policy") != "hard_nms"
        or settings.get("threshold") != 0.3
        or settings.get("metric") != "ios"
        or settings.get("class_agnostic") is not False
        or settings.get("score_floor") != 0.15):
        raise IntakeError("Frozen W04 policy settings were changed")


def source(path: Path | None) -> dict:
    if path is None:
        return {"status": "NOT_SUPPLIED"}
    result: dict[str, Any] = {"path": str(path), "status": "MISSING"}
    if path.is_symlink():
        result["status"] = "SYMLINK_NOT_ACCEPTED"
    elif path.is_file():
        result.update(status="HASHED", sha256=digest(path),
                      bytes=path.stat().st_size)
    return result


def intake(*, protocol_file: Path, lock_file: Path,
           video: Path | None = None, truth: Path | None = None,
           baseline: Path | None = None, baseline_kind: str = "none") -> dict:
    protocol, lock = read_json(protocol_file), read_json(lock_file)
    if baseline_kind not in BASELINES:
        raise IntakeError("Unrecognized baseline evidence type")
    validate_lock(protocol, lock, digest(lock_file))
    video_info, truth_info, baseline_info = source(video), source(truth), source(baseline)
    notes = []
    if video is None and truth is None:
        notes.append("No unseen candidate video and matching original truth supplied")
    elif video is None or truth is None:
        notes.append("Candidate video and original truth must be supplied as a pair")
    if video is not None and video_info["status"] != "HASHED":
        notes.append("Candidate video is unavailable or not a regular local file")
    if truth is not None and truth_info["status"] != "HASHED":
        notes.append("Candidate FLUID truth is unavailable or not a regular local file")
    baseline_type = baseline_kind
    if baseline_kind == "none" and baseline is not None:
        notes.append("Baseline bytes supplied without declaring original_t000 or new_replacement")
    if baseline_kind != "none" and baseline is None:
        notes.append("Declared baseline lacks original track bytes")
    if baseline is not None and baseline_info["status"] != "HASHED":
        notes.append("Baseline tracking file is missing or unreadable")
    if baseline_kind == "original_t000" and baseline is not None:
        notes.append("Original T000 provenance and same-window equivalence not independently verified")
    if baseline_kind == "new_replacement" and baseline is not None:
        notes.append("New replacement baseline is not recovered historical T000; same-window replay required")

    identical_video = (video_info.get("sha256") is not None
                       and video_info["sha256"] == protocol["source_video_sha256"])
    identical_truth = (truth_info.get("sha256") is not None
                       and truth_info["sha256"] == protocol["source_fluid_sha256"])
    if identical_video:
        notes.append("Candidate reuses byte-identical tuned W04 development video")
    if identical_truth:
        notes.append("Candidate reuses exact W04 development FLUID labels")
    role = KNOWN.get(video.name.lower()) if video is not None else None
    if role:
        notes.append(f"Video filename matches preexisting FLUID cohort role: {role}; unseen status not established")
    if video is not None and truth is not None and video_info.get("sha256") and truth_info.get("sha256"):
        if not identical_video and not identical_truth:
            notes.append("Distinct source bytes do not independently prove no earlier tuning or source leakage")
    notes.extend([
        "Independent evidence of untouched status and camera/source independence is required",
        "No candidate inference, same-window comparison, two-reviewer truth audit, or measured CPU p95 performed",
    ])
    if identical_video or identical_truth:
        status = "W04_CANDIDATE_REJECTED_REUSED_DEVELOPMENT_SOURCE"
    elif (video_info["status"] != "HASHED" or truth_info["status"] != "HASHED"):
        status = "W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH"
    elif role is not None:
        status = "W04_HISTORICAL_DATASET_ROLE_REQUIRES_EXTERNAL_REVIEW"
    else:
        status = "W04_DISTINCT_BYTES_INDEPENDENCE_UNVERIFIED"
    return {
        "schema_version": 1, "status": status,
        "selected_policy": lock["selected_policy"],
        "protocol_sha256": digest(protocol_file),
        "policy_lock_sha256": digest(lock_file),
        "candidate_video": video_info, "candidate_truth": truth_info,
        "candidate_baseline": baseline_info, "baseline_kind": baseline_type,
        "known_historical_video_role": role,
        "reuses_development_video_sha": bool(identical_video),
        "reuses_development_fluid_sha": bool(identical_truth),
        "candidate_bytes_distinct_from_development": bool(
            video_info.get("sha256") and truth_info.get("sha256")
            and not identical_video and not identical_truth
        ),
        "candidate_truly_unseen_verified": False,
        "same_window_baseline_and_alignment_verified": False,
        "independent_label_adjudication_verified": False,
        "end_to_end_cpu_p95_measured": False,
        "actual_holdout_scores_computed": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "reasons": notes,
        "next_steps": [
            "Verify independent acquisition/research-use rights and prior-exposure chronology externally",
            "Pin matching source, annotation, frozen model, first/end frames and warmup before evaluating",
            "Obtain same-window baseline bytes; distinguish authentic original T000 from new replacement",
            "Run frozen policy on consecutive candidate frames without tuning, then replay fixed +1/50px metrics",
            "Secure two independent physical-object reviewers and measure end-to-end CPU p95",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHA-bound holdout candidate intake, no evaluation")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-lock", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-kind", choices=BASELINES, default="none")
    parser.add_argument("--output", type=Path, required=True, help="Create a fresh JSON receipt")
    args = parser.parse_args(argv)
    result = intake(protocol_file=args.protocol, lock_file=args.policy_lock,
                    video=args.video, truth=args.truth,
                    baseline=args.baseline, baseline_kind=args.baseline_kind)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_bytes_distinct_from_development": result["candidate_bytes_distinct_from_development"],
        "candidate_truly_unseen_verified": False,
        "independent_holdout_pass": False,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
