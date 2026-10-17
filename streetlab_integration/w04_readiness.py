"""Read-only W04 local source/model audit. Does not run inference or score a blind holdout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

HEX = re.compile(r"^[0-9a-f]{64}$")
STUDY_ID = "fluid_fidrt_20250526"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and bool(HEX.fullmatch(value))


def _check_file(path: Path | None, expected: str | None) -> dict:
    if path is None:
        return {"status": "NOT_PROVIDED", "sha256_verified": False}
    result = {"path": str(path), "status": "MISSING", "sha256_verified": False}
    if not path.is_file() or path.is_symlink():
        return result
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = file_sha256(path)
    if expected and _valid_hash(expected):
        result["expected_sha256"] = expected
        result["sha256_verified"] = result["sha256"] == expected
        result["status"] = "VERIFIED_FROZEN_SHA" if result["sha256_verified"] else "SHA_MISMATCH"
    else:
        result["status"] = "HASHED_NO_FROZEN_REFERENCE"
    return result


def inspect_assets(*, video: Path, model_dir: Path, provenance: Path,
                   truth: Path | None = None,
                   model_hasher: Callable[[Path], str] | None = None) -> dict:
    """Never promote W04's tuned development window to an independent holdout."""
    blockers: list[str] = []
    origin: dict = {}
    if not provenance.is_file() or provenance.is_symlink():
        blockers.append("Frozen provenance missing or symlinked")
    else:
        try:
            candidate = json.loads(provenance.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                origin = candidate
            else:
                blockers.append("Provenance must be a JSON object")
        except (ValueError, UnicodeError, OSError):
            blockers.append("Frozen provenance unreadable or invalid JSON")
    for field in ("video_sha256", "openvino_model_sha256"):
        if not _valid_hash(origin.get(field)):
            blockers.append("Missing/invalid frozen provenance fingerprint: " + field)
    if origin.get("eligible_for_promotion") is not False:
        blockers.append("Frozen provenance must explicitly deny promotion")
    video_state = _check_file(video, origin.get("video_sha256"))
    if not video_state["sha256_verified"]:
        blockers.append("Original video must match frozen source SHA-256")

    model = {"path": str(model_dir), "status": "MISSING", "sha256_verified": False}
    if model_dir.is_dir() and not model_dir.is_symlink():
        if any(model_dir.glob("*.xml")) and any(model_dir.glob("*.bin")):
            try:
                if model_hasher is None:
                    from streetlab_phase3.video.openvino_export import hash_model_tree
                    model_hasher = hash_model_tree
                actual = model_hasher(model_dir)
                model["sha256"] = actual
                model["expected_sha256"] = origin.get("openvino_model_sha256")
                model["sha256_verified"] = bool(
                    _valid_hash(actual) and actual == origin.get("openvino_model_sha256"))
                model["status"] = "VERIFIED_FROZEN_SHA" if model["sha256_verified"] else "SHA_MISMATCH"
            except (ValueError, OSError) as exc:
                model["status"] = "MODEL_INTEGRITY_CHECK_FAILED"
                model["error"] = type(exc).__name__
        else:
            model["status"] = "INCOMPLETE_OPENVINO_XML_BIN"
    if not model["sha256_verified"]:
        blockers.append("OpenVINO XML/BIN model must match frozen model-tree SHA-256")

    truth_state = _check_file(truth, origin.get("fluid_sha256"))
    if not truth_state["sha256_verified"]:
        blockers.append("Matching frozen FLUID labels not verified locally")
    if origin.get("video_frame_window") is not None and origin["video_frame_window"] != [10750, 10950]:
        blockers.append("Development evaluation window differs from frozen W04 QA")
    core_blocker = any(
        ("provenance" in s.lower() or "promotion" in s.lower()
         or "window" in s.lower()) for s in blockers)
    source_ready = bool(video_state["sha256_verified"] and model["sha256_verified"]
                        and not core_blocker)
    return {
        "schema_version": 1,
        "status": "W04_LOCAL_DEVELOPMENT_ASSETS_VERIFIED" if source_ready
                  else "W04_LOCAL_ASSET_AUDIT_BLOCKED",
        "study_id": STUDY_ID,
        "assessment_scope": "READ_ONLY_DEVELOPMENT_ASSET_AUDIT_NOT_INFERENCE_OR_HOLDOUT",
        "files": {
            "provenance": {"path": str(provenance),
                           "sha256": file_sha256(provenance)
                           if provenance.is_file() and not provenance.is_symlink() else None},
            "original_video": video_state,
            "openvino_model": model,
            "fluid_truth": truth_state,
        },
        "development_window_source_frames": [10750, 10950],
        "existing_w04_201_frame_metrics_are_development_only": True,
        "source_video_and_model_verified": source_ready,
        "matching_fluid_truth_verified": truth_state["sha256_verified"],
        "new_video_inference_performed": False,
        "new_metric_scoring_performed": False,
        "independent_holdout_pass": False,
        "eligible_for_production": False,
        "blockers": blockers,
        "next_actions": [
            "Inspect immutable Build 3 batch if available, using existing phase3_holdout_validation.py inspect",
            "Treat W04 201-frame results as tuned development only; secure a genuinely untouched different video/camera",
            "Use the existing Phase 3 policy lock and same-window baseline validation with two-reviewer physical audit",
            "Do not mutate frozen model, FLUID labels, W04 scorecards or failed Chennai holdout",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only StreetLab W04 source/model audit")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--truth", type=Path, help="Exact FLUID scoring file, if available")
    parser.add_argument("--output", type=Path, required=True, help="NEW JSON file, no overwrite")
    args = parser.parse_args(argv)
    result = inspect_assets(video=args.video, model_dir=args.model_dir,
                            provenance=args.provenance, truth=args.truth)
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "status": result["status"],
        "source_video_and_model_verified": result["source_video_and_model_verified"],
        "matching_fluid_truth_verified": result["matching_fluid_truth_verified"],
        "independent_holdout_pass": False,
        "blockers": result["blockers"],
        "output": str(args.output),
    }, indent=2))
    return 0 if result["source_video_and_model_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
