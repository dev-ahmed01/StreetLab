"""One bounded, read-only forensic inventory for historical StreetLab T000 outputs.

Finds original track, manifests, summary, and archives across both checkouts.
Does not infer T000 provenance from a name or run Geo-trax inference.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

SKIP_NAMES = {".git", "node_modules", "__pycache__", ".next", "site-packages",
              "dist", "build", "checkpoint_openvino_model", "frames", "images"}
TRACK_EXT = {".txt", ".csv", ".parquet", ".jsonl"}
OTHER_EXT = {".json", ".yaml", ".yml", ".toml", ".zip", ".ps1"}
RELEVANT = re.compile(r"t000|baseline|geotrax|tuning|20250526_video|pixel_scorecard", re.I)
START, END = 10750, 10950
MAX_FILES = 100000
MAX_MATCHES = 1000


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def track_sample(path: Path) -> dict:
    """Sample a small number of rows; no claim about full-file validity."""
    sample = {"sampled_14_column_rows": 0, "sampled_rows": 0,
              "sample_in_w04_window": 0, "possible_14col_geotrax": False}
    if path.suffix.lower() not in {".txt", ".csv"}:
        return sample
    try:
        with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= 1000:
                    break
                sample["sampled_rows"] += 1
                if len(row) == 14:
                    sample["sampled_14_column_rows"] += 1
                    try:
                        frame = int(row[0])
                        int(row[1])
                        if START <= frame <= END:
                            sample["sample_in_w04_window"] += 1
                    except (ValueError, IndexError):
                        pass
    except OSError as exc:
        sample["sample_error"] = str(exc)
    sample["possible_14col_geotrax"] = bool(
        sample["sampled_rows"] and sample["sampled_rows"] == sample["sampled_14_column_rows"]
    )
    return sample


def classify(path: Path) -> str:
    name = str(path).lower()
    if path.suffix.lower() in TRACK_EXT and RELEVANT.search(name):
        return "possible_track_or_tabular_evidence"
    if path.suffix.lower() == ".zip" and RELEVANT.search(name):
        return "possible_archived_evidence"
    if path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"} and RELEVANT.search(name):
        return "possible_run_provenance_or_scorecard"
    if path.suffix.lower() == ".ps1" and RELEVANT.search(name):
        return "historical_runner"
    return ""


def scan(desktop: Path, max_entries: int = MAX_FILES) -> dict:
    desktop = desktop.resolve()
    projects = [desktop / "StreetLab", desktop / "StreetLab-engine-trial"]
    project_roots = []
    for project in projects:
        for suffix in ("artifacts", "outputs", "results", "runs", "experiments",
                       "data/benchmarks", "data/processed"):
            project_roots.append(project / suffix)
    # Other copies with StreetLab in the name, at most first-level Desktop folders.
    if desktop.is_dir():
        for child in sorted(desktop.iterdir()):
            if child.is_dir() and "streetlab" in child.name.lower() and child not in projects:
                project_roots.extend([child / "artifacts", child / "outputs",
                                     child / "runs", child / "experiments"])
    explicit = [
        desktop / "StreetLab/artifacts/phase3/geotrax/fluid_fidrt_20250526/full/20250526_video.txt",
        desktop / "StreetLab/artifacts/phase3/geotrax_tuning/experiments/T000_BASELINE",
        desktop / "StreetLab-engine-trial/artifacts/phase3/geotrax/fluid_fidrt_20250526/full/20250526_video.txt",
        desktop / "StreetLab-engine-trial/artifacts/phase3/geotrax_tuning/experiments/T000_BASELINE",
        desktop / "StreetLab/artifacts/phase3/geotrax_tuning/summary.json",
        desktop / "StreetLab-engine-trial/artifacts/phase3/geotrax_tuning/summary.json",
    ]
    folder_status = [{"path": str(p), "exists": p.is_dir()} for p in projects]
    for root in project_roots:
        folder_status.append({"path": str(root), "exists": root.is_dir()})
    explicit_status = [{"path": str(p), "exists": p.exists(),
                        "kind": "file" if p.is_file() else "directory" if p.is_dir() else "missing"}
                       for p in explicit]
    seen_dirs = set()
    matched = []
    visited = 0
    truncated = False
    errors = []
    for root in project_roots:
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                resolved = str(directory.resolve()).lower()
                if resolved in seen_dirs:
                    continue
                seen_dirs.add(resolved)
                with os.scandir(directory) as directory_listing:
                    entries = list(directory_listing)
            except (OSError, PermissionError) as exc:
                errors.append({"path": str(directory), "error": str(exc)})
                continue
            for entry in entries:
                visited += 1
                if visited > max_entries or len(matched) >= MAX_MATCHES:
                    truncated = True
                    break
                name = entry.name
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if (name in SKIP_NAMES or name.startswith(".venv") or
                            name.startswith(".venv-") or name == "node_modules"):
                            continue
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    typ = classify(path)
                    if not typ:
                        continue
                    st = entry.stat(follow_symlinks=False)
                    item = {"path": str(path), "kind": typ, "bytes": st.st_size,
                            "last_modified_utc": dt.datetime.fromtimestamp(
                                st.st_mtime, dt.timezone.utc).isoformat()}
                    if typ == "possible_track_or_tabular_evidence":
                        item.update(track_sample(path))
                    if (typ == "possible_run_provenance_or_scorecard"
                        and path.suffix.lower() == ".json" and st.st_size <= 2_000_000):
                        try:
                            meta = json.loads(path.read_text(encoding="utf-8-sig"))
                            if isinstance(meta, dict):
                                item["json_top_level_keys"] = list(meta)[:45]
                                item["reported_candidate"] = meta.get("candidate", meta.get("candidate_id"))
                                item["reported_status"] = meta.get("status")
                        except (OSError, ValueError, UnicodeError) as exc:
                            item["json_parse_issue"] = str(exc)[:160]
                    if st.st_size <= 30_000_000:
                        item["sha256"] = hash_file(path)
                    matched.append(item)
                except (OSError, ValueError, OverflowError) as exc:
                    errors.append({"path": entry.path, "error": str(exc)})
            if truncated:
                break
        if truncated:
            break
    matched.sort(key=lambda i:(0 if re.search(r"T000",i["path"],re.I) else
                               1 if "geotrax" in i["path"].lower() else 2,i["path"]))
    return {
        "status": "T000_FORENSIC_INVENTORY_NOT_AUTHENTICATED",
        "eligible_for_production": False, "desktop": str(desktop),
        "original_source_frame_window": [START, END],
        "fluid_scoring_frame_window": [START+1, END+1],
        "historical_expected_path_checks": explicit_status,
        "searched_folders": folder_status,
        "visited_entries": visited, "matched_artifact_files": len(matched),
        "truncated": truncated, "errors": errors[:50], "matches": matched,
        "limitations": ("Metadata and limited row samples only. Track frame coverage, original "
                        "T000 command/model provenance, +1/50px score comparability and physical "
                        "class truth still require independent verification. Do not reuse "
                        "experimental policy exports as T000.")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desktop", type=Path, default=Path.home() / "Desktop")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Immutable output file exists; choose another name")
    result = scan(args.desktop)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"status": result["status"], "checked_paths":
                      result["historical_expected_path_checks"],
                      "matched_artifacts":result["matched_artifact_files"],
                      "visited_entries":result["visited_entries"],
                      "truncated":result["truncated"],"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
