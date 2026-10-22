"""Verified, read-only presentation of an existing native SUMO research ZIP.

Only source-hashed synthetic simulations are displayed. Neither this adapter
nor a green UI response claims real site calibration or independent holdout.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .research_release import NAMES, SOURCES, PRODUCED, RESEARCH_STATUS
from .research_results import BRANCHES, METRICS, ResultsError, analyze

MAX_ZIP_BYTES = 50_000_000
MAX_MEMBER_BYTES = 20_000_000
MAX_TOTAL_BYTES = 100_000_000


class BundleDisplayError(ValueError):
    """The selected ZIP cannot support an authentic research display."""


def _json(raw: bytes, what: str) -> dict:
    try:
        data = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise BundleDisplayError("Unreadable ZIP evidence: " + what) from exc
    if not isinstance(data, dict):
        raise BundleDisplayError("Invalid ZIP evidence: " + what)
    return data


def _check(condition: bool, why: str) -> None:
    if not condition:
        raise BundleDisplayError(why)


def read_bundle(path: Path) -> dict:
    """Recompute archive and native study invariants before showing any metric."""
    path = Path(path)
    _check(path.is_file() and not path.is_symlink(), "Research bundle missing or linked")
    _check(path.stat().st_size <= MAX_ZIP_BYTES, "Research bundle too large")
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            names = [x.filename for x in members]
            _check(len(set(names)) == len(names), "Duplicate ZIP member")
            _check(all(x.file_size <= MAX_MEMBER_BYTES for x in members)
                   and sum(x.file_size for x in members) <= MAX_TOTAL_BYTES,
                   "Oversized ZIP member or uncompressed evidence")
            _check(all(((x.external_attr >> 16) & 0o170000) in (0, 0o100000)
                       for x in members), "Linked or special ZIP entry")
            _check("manifest.json" in names, "Missing evidence manifest")
            manifest = _json(archive.read("manifest.json"), "manifest")
            _check(manifest.get("release_type") == "BOUNDED_RESEARCH_PROTOTYPE"
                   and manifest.get("native_sumo") == "SYNTHETIC_ONLY"
                   and manifest.get("chennai_holdout_generalization_pass") is False
                   and manifest.get("independent_holdout_pass") is False
                   and manifest.get("field_intervention_effect_proven") is False
                   and manifest.get("production_release_approved") is False,
                   "Invalid research release classification")
            optional_audit = manifest.get("may25_audit_status")
            _check(optional_audit in ("NOT_INCLUDED_OR_NOT_VERIFIED",
                                      "COMPLETED_SAVED_AUDIT_INCLUDED_LABEL_RELATIVE"),
                   "Unrecognized May25 audit status")
            expected = set(NAMES) | {"README.txt"}
            if optional_audit == "COMPLETED_SAVED_AUDIT_INCLUDED_LABEL_RELATIVE":
                expected.add("automated_object_audit.json")
            hashes = manifest.get("files_sha256")
            _check(isinstance(hashes, dict) and set(hashes) == expected,
                   "Incomplete evidence SHA inventory")
            actual_names = {"manifest.json", "README.txt"} | {
                "evidence/" + name for name in expected if name != "README.txt"
            }
            _check(set(names) == actual_names, "Unexpected or missing ZIP member")
            blobs = {}
            for name in sorted(expected):
                name_in_zip = name if name == "README.txt" else "evidence/" + name
                digest = hashes[name]
                _check(isinstance(digest, str) and len(digest) == 64
                       and all(c in "0123456789abcdef" for c in digest),
                       "Malformed SHA inventory")
                content = archive.read(name_in_zip)
                _check(sha256(content).hexdigest() == digest,
                       "Evidence SHA mismatch: " + name)
                blobs[name] = content
    except (BadZipFile, KeyError, OSError, RuntimeError, EOFError) as exc:
        raise BundleDisplayError("Cannot read valid bounded evidence ZIP") from exc

    evidence = _json(blobs["submission_evidence.json"], "source")
    study = _json(blobs["native_sumo_study.json"], "native study")
    provenance = _json(blobs["research_results_manifest.json"], "results manifest")
    _check(provenance.get("classification") == "SYNTHETIC_RESEARCH_ONLY",
           "Invalid results provenance")
    for property_name, required_names in (
        ("source_artifacts_sha256", set(SOURCES)),
        ("produced_sha256", set(PRODUCED)),
        ("ancillary_sha256", {"submission_summary.md", "native_sumo.log"}),
    ):
        claimed = provenance.get(property_name)
        _check(isinstance(claimed, dict) and set(claimed) == required_names,
               "Incomplete original results provenance")
        for filename in required_names:
            _check(claimed[filename] == sha256(blobs[filename]).hexdigest(),
                   "Original run evidence changed: " + filename)
    _check(evidence.get("research_workflow_status") == RESEARCH_STATUS,
           "Native experiment not completed")
    try:
        data = analyze(evidence, study)
    except (ResultsError, AttributeError, KeyError, TypeError) as exc:
        raise BundleDisplayError("Inconsistent synthetic scenario evidence") from exc

    return {
        "status": "VERIFIED_NATIVE_SUMO_SYNTHETIC_RESEARCH",
        "source": "USER_CONFIGURED_SHA_CHECKED_BUNDLE",
        "bundle_sha256": sha256(path.read_bytes()).hexdigest(),
        "ensemble_members": data["members"],
        "metrics": [
            {
                "key": row["key"], "label": row["name"], "unit": row["unit"],
                "medians": {name: row["branches"][name]["median"] for name in BRANCHES},
                "ranges": {name: {
                    "min": row["branches"][name]["min"], "max": row["branches"][name]["max"]
                } for name in BRANCHES},
            } for row in data["metrics"]
        ],
        "paired_median_deltas": {
            name: {metric: value["median"] for metric, value in data["paired"][name].items()}
            for name in BRANCHES[1:]
        },
        "reduced_arrival_branches": data["reduced_arrival_branches"],
        "historical_may25_audit_verified_by_this_endpoint": False,
        "chennai_holdout_generalization_pass": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "caveat": "Original native SUMO on assumed junction geometry and demand; synthetic paired assignment sensitivity, not measured traffic impact.",
    }
