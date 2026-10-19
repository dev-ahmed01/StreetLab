"""Create a source-integrity-gated research-only archive from completed SUMO outputs.

This is NOT a physical site, detector or production release. It packages already
executed native-SUMO synthetic research output; no rerun or parameter tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

FIGURES = (
    "figure_01_queue.svg", "figure_02_arrivals.svg",
    "figure_03_speed.svg", "figure_04_waiting.svg",
    "figure_05_reroutes.svg",
)
SOURCES = ("submission_evidence.json", "native_sumo_study.json")
PRODUCED = ("research_results.md", "research_metrics.csv") + FIGURES
OTHERS = ("submission_summary.md", "native_sumo.log", "research_results_manifest.json")
NAMES = SOURCES + PRODUCED + OTHERS
RESEARCH_STATUS = "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION"


class ReleaseError(ValueError):
    """Source, scorecard, or release status failed audit."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise ReleaseError("Invalid required research JSON: " + path.name) from exc
    if not isinstance(value, dict):
        raise ReleaseError("Invalid required research document: " + path.name)
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_read(folder: Path, name: str) -> bytes:
    path = folder / name
    if path.is_symlink() or not path.is_file():
        raise ReleaseError("Missing/untrusted research evidence: " + name)
    if path.stat().st_size > 20_000_000:
        raise ReleaseError("Research evidence exceeds size bound: " + name)
    return path.read_bytes()


def validate(run_dir: Path, may25_audit: Path | None = None) -> tuple[dict[str, bytes], dict]:
    """Return a checked, immutable-in-memory snapshot with explicit provenance."""
    run_dir = Path(run_dir)
    files = {name: _safe_read(run_dir, name) for name in NAMES}
    evidence = _read_json(run_dir / "submission_evidence.json")
    summary = _read_json(run_dir / "research_results_manifest.json")
    if (evidence.get("research_workflow_status") != RESEARCH_STATUS
        or evidence.get("track") != "RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY"
        or evidence.get("production_release_approved") is not False
        or evidence.get("external_site_transfer_validated") is not False
        or evidence.get("real_world_intervention_effect_proven") is not False
        or evidence.get("dataset_summaries", {}).get("chennai", {}).get("holdout_generalization_pass") is not False
        or evidence.get("native_sumo", {}).get("status") != "SUMO_EXECUTED_SYNTHETIC_ONLY"
        or evidence.get("native_sumo", {}).get("production_release_approved") is not False
        or evidence.get("native_sumo", {}).get("site_specific_field_validation") is not False):
        raise ReleaseError("Research source is not a completed limited synthetic SUMO run")
    if summary.get("classification") != "SYNTHETIC_RESEARCH_ONLY":
        raise ReleaseError("Results manifest is not synthetic-only")
    source_hashes, produced_hashes = summary.get("source_artifacts_sha256"), summary.get("produced_sha256")
    if (not isinstance(source_hashes, dict) or not isinstance(produced_hashes, dict)
        or set(source_hashes) != set(SOURCES) or set(produced_hashes) != set(PRODUCED)):
        raise ReleaseError("Results manifest does not cover the complete research source set")
    for name in SOURCES:
        if _digest(files[name]) != source_hashes[name]:
            raise ReleaseError("Native source hash mismatch: " + name)
    for name in PRODUCED:
        if _digest(files[name]) != produced_hashes[name]:
            raise ReleaseError("Derived research artifact hash mismatch: " + name)
    # Prevent a file changing between validation and archive creation.
    if any(files[name] != _safe_read(run_dir, name) for name in NAMES):
        raise ReleaseError("Source evidence changed while preparing bundle")
    audit_status = "NOT_INCLUDED_OR_NOT_VERIFIED"
    if may25_audit is not None:
        from streetlab_integration.research_audit_display import read_audit
        # The autonomous audit runner itself must already have checked full source SHA.
        read_audit(Path(may25_audit))
        audit_bytes = Path(may25_audit).read_bytes()
        if len(audit_bytes) > 10_000_000:
            raise ReleaseError("May25 audit too large")
        files["automated_object_audit.json"] = audit_bytes
        audit_status = "COMPLETED_SAVED_AUDIT_INCLUDED_LABEL_RELATIVE"
    manifest = {
        "schema_version": 1,
        "release_type": "BOUNDED_RESEARCH_PROTOTYPE",
        "native_sumo": "SYNTHETIC_ONLY",
        "may25_audit_status": audit_status,
        "chennai_holdout_generalization_pass": False,
        "independent_holdout_pass": False,
        "field_intervention_effect_proven": False,
        "production_release_approved": False,
        "files_sha256": {name: _digest(files[name]) for name in sorted(files)},
        "scope": "Native SUMO synthetic scenario and saved provenance, not field-calibrated traffic control",
    }
    return files, manifest


def _entry(name: str) -> ZipInfo:
    entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = ZIP_DEFLATED
    entry.external_attr = 0o600 << 16
    return entry


def package(run_dir: Path, output: Path, may25_audit: Path | None = None) -> dict:
    files, manifest = validate(run_dir, may25_audit)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    readme = (
        "StreetLab research prototype — evidence export\n"
        "--------------------------------------------\n"
        "This bundle records already executed native SUMO simulations on an ASSUMED network.\n"
        "Chennai independent held-out generalization failed. W04 is tuned development.\n"
        "May25 is historical label-relative regression, not independently adjudicated traffic truth.\n"
        "No field-effect, real-time inference, or production approval is claimed.\n"
        "manifest.json binds each packaged file to its original SHA256.\n"
    ).encode("utf-8")
    try:
        with output.open("xb") as handle:
            with ZipFile(handle, "w") as archive:
                for name in sorted(files):
                    archive.writestr(_entry("evidence/" + name), files[name])
                archive.writestr(_entry("manifest.json"),
                                 (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"))
                archive.writestr(_entry("README.txt"), readme)
    except Exception:
        # Failed exports must not be mistaken for a finished release.
        output.unlink(missing_ok=True)
        raise
    return {"path": str(output), "classification": manifest["release_type"],
            "may25_audit_status": manifest["may25_audit_status"],
            "file_count": len(files), "production_release_approved": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive source-hashed bounded StreetLab research results")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--may25-audit", type=Path, default=None)
    args = parser.parse_args()
    try:
        print(json.dumps(package(args.run_dir, args.output, args.may25_audit), indent=2))
    except (ReleaseError, OSError, ValueError) as exc:
        parser.exit(2, "Release packaging rejected: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
