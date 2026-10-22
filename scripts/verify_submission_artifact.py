"""Verify downloaded StreetLab submission artifact without third-party dependencies.

This checks byte integrity of the actual native-SUMO synthetic evidence and
records its strict limitations. It does not independently verify real-world
traffic labels or the private May25 original source video.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile


class VerificationError(ValueError):
    pass


def _json(data: bytes, label: str) -> dict:
    try:
        result = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise VerificationError(f"Unreadable {label}") from exc
    if not isinstance(result, dict):
        raise VerificationError(f"{label} must be a JSON object")
    return result


def _sha(blob: bytes) -> str:
    return sha256(blob).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def verify(folder: Path, expected_commit: str | None = None) -> dict:
    """Fail closed on tampering, missing items or inflated scientific claims."""
    folder = Path(folder)
    receipt_file = folder / "submission_build_receipt.json"
    _require(receipt_file.is_file() and not receipt_file.is_symlink(),
             "Missing submission_build_receipt.json")
    receipt = _json(receipt_file.read_bytes(), receipt_file.name)
    _require(receipt.get("schema_version") == 1
             and receipt.get("release_scope") == "BOUNDED_RESEARCH_PROTOTYPE",
             "Unexpected release receipt")
    commit = receipt.get("commit_sha")
    _require(isinstance(commit, str) and len(commit) == 40
             and all(c in "0123456789abcdef" for c in commit),
             "Missing source commit fingerprint")
    if expected_commit is not None:
        _require(commit == expected_commit, "Submission was not built from the requested commit")
    for name in ("may25_original_video_rehashed_in_ci",
                 "canonical_may25_full_source_audit_verified_in_ci",
                 "chennai_holdout_generalization_pass",
                 "independent_holdout_pass", "production_release_approved"):
        _require(receipt.get(name) is False, "Unexpected or promoted research claim: " + name)
    _require(receipt.get("native_sumo_execution") ==
             "ACTUAL_ENGINE_SYNTHETIC_NETWORK_ONLY", "Unverified simulator claim")

    files = {
        "streetlab_research_bundle.zip": "research_bundle_sha256",
        "README_SUBMISSION.md": "submission_guide_sha256",
        "MAY25_FULL_COHORT_REPLAY_2026-10-23.md": "saved_cohort_replay_sha256",
    }
    for name, key in files.items():
        path = folder / name
        _require(path.is_file() and not path.is_symlink(), "Missing submission file: " + name)
        _require(isinstance(receipt.get(key), str) and _sha(path.read_bytes()) == receipt[key],
                 "Submission SHA mismatch: " + name)

    archive = folder / "streetlab_research_bundle.zip"
    try:
        with ZipFile(archive) as z:
            names = z.namelist()
            _require(len(names) == len(set(names)), "Duplicate ZIP entries")
            _require("manifest.json" in names, "Missing ZIP manifest")
            manifest = _json(z.read("manifest.json"), "ZIP manifest")
            _require(manifest.get("release_type") == "BOUNDED_RESEARCH_PROTOTYPE"
                     and manifest.get("native_sumo") == "SYNTHETIC_ONLY"
                     and manifest.get("may25_audit_status") == "NOT_INCLUDED_OR_NOT_VERIFIED"
                     and manifest.get("chennai_holdout_generalization_pass") is False
                     and manifest.get("independent_holdout_pass") is False
                     and manifest.get("production_release_approved") is False,
                     "ZIP manifest contains unsupported research claims")
            hashes = manifest.get("files_sha256")
            _require(isinstance(hashes, dict), "Missing ZIP SHA manifest")
            expected_paths = {"manifest.json"} | {
                ("README.txt" if n == "README.txt" else "evidence/" + n)
                for n in hashes
            }
            _require(set(names) == expected_paths, "Unexpected or unlisted ZIP entry")
            for name, value in hashes.items():
                _require(isinstance(name, str) and isinstance(value, str)
                         and len(value) == 64, "Malformed ZIP digest record")
                entry = "README.txt" if name == "README.txt" else "evidence/" + name
                _require(_sha(z.read(entry)) == value, "ZIP evidence hash mismatch: " + name)
            evidence = _json(z.read("evidence/submission_evidence.json"),
                             "native SUMO evidence")
            _require(evidence.get("research_workflow_status") ==
                     "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION"
                     and evidence.get("production_release_approved") is False
                     and isinstance(evidence.get("native_sumo"), dict)
                     and evidence["native_sumo"].get("status") == "SUMO_EXECUTED_SYNTHETIC_ONLY"
                     and isinstance(evidence.get("dataset_summaries"), dict)
                     and isinstance(evidence["dataset_summaries"].get("chennai"), dict)
                     and evidence["dataset_summaries"]["chennai"].get("holdout_generalization_pass") is False,
                     "Actual submission source no longer matches research limitations")
    except (KeyError, OSError, EOFError, RuntimeError, ValueError) as exc:
        if isinstance(exc, VerificationError):
            raise
        raise VerificationError("Unreadable or altered evidence ZIP") from exc
    return {
        "verification_status": "INTEGRITY_PASS_SYNTHETIC_RESEARCH_ONLY",
        "source_commit_sha": commit,
        "integrity_file_count": len(files) + len(hashes),
        "original_may25_video_verified": False,
        "production_release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify SHA-locked StreetLab artifact")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--commit", help="Optional exact expected full Git commit SHA")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.directory, args.commit), indent=2))
    except (OSError, VerificationError) as exc:
        parser.exit(2, "Submission integrity FAILED: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
