"""Bounded read-only inventory of local FLUID video/trajectory candidates.

The inventory discovers possible file *pairs*, not independent footage.
No video frames, model inference, label corrections, metrics or promotion.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from .w04_holdout_intake import digest, read_json, validate_lock

STATUS = "W04_LOCAL_VIDEO_TRUTH_INVENTORY_NOT_BLIND_VALIDATION"
VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}
SKIP = {".git", "node_modules", "__pycache__", ".venv", ".venv-sahi-audit",
        ".venv-openvino", "artifacts", "checkpoints", "models"}
SUFFIXES = ("_traj", "_trajectory", "_tracks", "_truth", "_ground_truth",
            "_annotations", "_labels")
REQUIRED = {"frame", "id", "cx", "cy", "type"}
DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


class InventoryError(ValueError):
    """Input or scan could not be inspected without silently omitting evidence."""


def _header(path: Path) -> dict:
    """Read just the header; incompatible files are not silently paired."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream), [])
        normalized = {v.strip().lower() for v in columns}
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"valid_fluid_schema": False, "reason": type(exc).__name__}
    missing = sorted(REQUIRED - normalized)
    return {"valid_fluid_schema": not missing,
            "missing_columns": missing}


def _role(name: str, manifest: dict) -> dict:
    for item in manifest.get("studies", []):
        if name.casefold() == str(item.get("video", "")).casefold():
            role = item.get("role")
            return {"study_id": item.get("study_id"),
                    "historical_role": role,
                    "source_history": ("TUNED_DEVELOPMENT_NO_BLIND_USE"
                                       if name.casefold() == "20250526_video.mp4"
                                       else "HISTORICAL_ROLE_NEEDS_INDEPENDENT_REVIEW")}
    return {"study_id": None, "historical_role": None,
            "source_history": "UNKNOWN_NOT_AUTHENTICATED_AS_UNSEEN"}


def _base(stem: str) -> str:
    lower = stem.casefold()
    for suffix in SUFFIXES:
        if lower.endswith(suffix):
            return lower[:-len(suffix)]
    return lower


def _relation(video: Path, truth: Path, manifest: dict) -> str | None:
    stem = video.stem.casefold()
    truth_stem = truth.stem.casefold()
    for study in manifest.get("studies", []):
        if video.name.casefold() == str(study.get("video", "")).casefold():
            if truth.name.casefold() == str(study.get("tracks", "")).casefold():
                return "MANIFEST_NAMED_PAIR_NOT_CONTENT_VALIDATED"
    if _base(truth_stem) == stem:
        return "SHARED_VIDEO_STEM_NOT_CONTENT_VALIDATED"
    video_dates = set(DATE.findall(stem))
    truth_dates = set(DATE.findall(truth_stem))
    if video_dates and video_dates & truth_dates:
        return "SHARED_DATE_POSSIBLE_MISMATCH_REVIEW_REQUIRED"
    return None


def _files(roots: list[Path], *, max_depth: int, max_entries: int) -> tuple[list[Path], dict]:
    if not roots:
        raise InventoryError("Provide at least one local dataset root")
    found = []
    checked = 0
    symlinks_skipped = 0
    unique: set[str] = set()
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise InventoryError(f"Dataset root must be a real existing directory: {root}")
        resolved = str(root.resolve()).casefold()
        if resolved in unique:
            continue
        unique.add(resolved)

        def fail(exc):
            raise InventoryError(f"Cannot safely enumerate local research directory: {exc}")

        for base, directories, files in os.walk(root, topdown=True,
                                                followlinks=False, onerror=fail):
            rel = Path(base).relative_to(root)
            depth = len(rel.parts) if rel != Path(".") else 0
            keep = []
            for name in sorted(directories):
                child = Path(base) / name
                if child.is_symlink():
                    symlinks_skipped += 1
                elif name.casefold() not in SKIP and depth < max_depth:
                    keep.append(name)
            directories[:] = keep
            for name in sorted(files):
                checked += 1
                if checked > max_entries:
                    raise InventoryError(
                        "File scan cap reached; narrow roots or increase --max-entries")
                candidate = Path(base) / name
                if candidate.is_symlink():
                    symlinks_skipped += 1
                elif candidate.suffix.casefold() in VIDEOS | {".csv"}:
                    found.append(candidate)
    return found, {"entries_seen": checked, "symlinks_skipped": symlinks_skipped,
                   "roots_scanned": sorted(unique)}


def inventory(*, roots: list[Path], protocol_file: Path, lock_file: Path,
              manifest_file: Path, max_depth: int = 4,
              max_entries: int = 20000) -> dict:
    if not 0 <= max_depth <= 12 or not 1 <= max_entries <= 200000:
        raise InventoryError("Unsafe scan depth or file-count limit")
    protocol, lock = read_json(protocol_file), read_json(lock_file)
    validate_lock(protocol, lock, digest(lock_file))
    manifest = read_json(manifest_file)
    if not isinstance(manifest.get("studies"), list):
        raise InventoryError("Missing FLUID study manifest")
    candidates, scan = _files(roots, max_depth=max_depth, max_entries=max_entries)
    video = sorted((p for p in candidates if p.suffix.casefold() in VIDEOS),
                   key=lambda x: str(x).casefold())
    csvs = sorted((p for p in candidates if p.suffix.casefold() == ".csv"),
                  key=lambda x: str(x).casefold())
    valid = []
    incompatible = []
    for file in csvs:
        schema = _header(file)
        if schema["valid_fluid_schema"]:
            valid.append(file)
        else:
            incompatible.append({
                "path": str(file), "missing_columns": schema.get("missing_columns"),
                "reason": schema.get("reason", "NOT_FLUID_TRAJECTORY_SCHEMA"),
            })
    videos = []
    paired_count = 0
    for file in video:
        role = _role(file.name, manifest)
        matches = []
        for truth in valid:
            relationship = _relation(file, truth, manifest)
            if relationship is not None:
                matches.append({
                    "truth_path": str(truth), "match_basis": relationship,
                    "truth_contents_matched_to_video": False,
                })
        rank = {"MANIFEST_NAMED_PAIR_NOT_CONTENT_VALIDATED": 0,
                "SHARED_VIDEO_STEM_NOT_CONTENT_VALIDATED": 1,
                "SHARED_DATE_POSSIBLE_MISMATCH_REVIEW_REQUIRED": 2}
        matches.sort(key=lambda v: (rank[v["match_basis"]],
                                     v["truth_path"].casefold()))
        paired_count += bool(matches)
        videos.append({
            "video_path": str(file), "bytes": file.stat().st_size,
            **role, "possible_fluid_truths": matches,
            "pairing_verified_by_video_frame_content": False,
            "source_sha256_computed": False,
            "untouched_holdout_verified": False,
        })
    unmatched = [str(file) for file in valid
                 if not any(_relation(video_file, file, manifest) for video_file in video)]
    return {
        "schema_version": 1, "status": STATUS,
        "policy": lock["selected_policy"],
        "policy_lock_sha256": digest(lock_file),
        "protocol_sha256": digest(protocol_file),
        "manifest_sha256": digest(manifest_file),
        "scan": scan,
        "candidate_video_count": len(video),
        "candidate_fluid_label_csv_count": len(valid),
        "video_count_with_possible_truth": paired_count,
        "videos": videos,
        "unpaired_fluid_truth_files": unmatched,
        "incompatible_csv_files": incompatible,
        "video_sha256_not_computed_for_bulk_inventory": True,
        "source_video_frame_decoding_not_performed": True,
        "annotation_to_video_alignment_not_proven": True,
        "previous_source_exposure_cannot_be_inferred_from_files": True,
        "new_independent_test_verified": False,
        "production_release_approved": False,
        "next_step": (
            "Inspect discovered pair candidates and their prior use; choose a genuinely "
            "untouched source only if independently proven, then run "
            "STREETLAB_W04_HOLDOUT_INTAKE.ps1 with exact selected paths."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List local video/FLUID file pair candidates; no scoring")
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-entries", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = inventory(
        roots=args.root, protocol_file=args.protocol,
        lock_file=args.policy_lock, manifest_file=args.manifest,
        max_depth=args.max_depth, max_entries=args.max_entries,
    )
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": STATUS,
        "videos_found": result["candidate_video_count"],
        "matching_trajectory_csvs_found": result["candidate_fluid_label_csv_count"],
        "videos_with_possible_pairs": result["video_count_with_possible_truth"],
        "source_independence_confirmed": False,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
