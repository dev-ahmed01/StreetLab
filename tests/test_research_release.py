"""Release bundle integrity, conservation and overwrite regression tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from streetlab_integration.research_release import (
    NAMES, SOURCES, PRODUCED, ReleaseError, package,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(folder: Path) -> None:
    folder.mkdir()
    blobs = {name: ("test content for " + name).encode("utf-8") for name in NAMES}
    evidence = {
        "track": "RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY",
        "research_workflow_status": "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION",
        "production_release_approved": False,
        "external_site_transfer_validated": False,
        "real_world_intervention_effect_proven": False,
        "dataset_summaries": {"chennai": {"holdout_generalization_pass": False}},
        "native_sumo": {"status": "SUMO_EXECUTED_SYNTHETIC_ONLY",
                        "production_release_approved": False,
                        "site_specific_field_validation": False},
    }
    blobs["submission_evidence.json"] = json.dumps(evidence).encode()
    manifest = {
        "classification": "SYNTHETIC_RESEARCH_ONLY",
        "source_artifacts_sha256": {name: _sha(blobs[name]) for name in SOURCES},
        "produced_sha256": {name: _sha(blobs[name]) for name in PRODUCED},
        "ancillary_sha256": {name: _sha(blobs[name]) for name in ("submission_summary.md", "native_sumo.log")},
    }
    blobs["research_results_manifest.json"] = json.dumps(manifest).encode()
    for name, data in blobs.items():
        (folder / name).write_bytes(data)


def test_release_archive_hashes_and_explicit_scientific_limits(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    result = package(src, out)
    assert result["file_count"] == len(NAMES)
    assert result["production_release_approved"] is False
    assert result["may25_audit_status"] == "NOT_INCLUDED_OR_NOT_VERIFIED"
    with ZipFile(out) as archive:
        assert "manifest.json" in archive.namelist()
        values = json.loads(archive.read("manifest.json"))
        assert values["release_type"] == "BOUNDED_RESEARCH_PROTOTYPE"
        assert values["independent_holdout_pass"] is False
        assert values["field_intervention_effect_proven"] is False
        assert values["chennai_holdout_generalization_pass"] is False
        assert "evidence/automated_object_audit.json" not in archive.namelist()
        for name in NAMES:
            assert _sha(archive.read("evidence/"+name)) == values["files_sha256"][name]
        assert b"ASSUMED network" in archive.read("README.txt")
        assert _sha(archive.read("README.txt")) == values["files_sha256"]["README.txt"]


def test_tampered_source_rejected_without_output(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    (src/"research_metrics.csv").write_text("changed metric", encoding="utf-8")
    with pytest.raises(ReleaseError, match="hash mismatch"):
        package(src, out)
    assert not out.exists()


def test_production_claim_rejected_without_output(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    p=src/"submission_evidence.json"
    j=json.loads(p.read_text())
    j["production_release_approved"] = True
    p.write_text(json.dumps(j))
    with pytest.raises(ReleaseError):
        package(src,out)
    assert not out.exists()


def test_original_release_refuses_overwrite_and_preserves_bytes(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    out.write_bytes(b"PROTECTED EXISTING RELEASE")
    with pytest.raises(FileExistsError):
        package(src, out)
    assert out.read_bytes() == b"PROTECTED EXISTING RELEASE"


def test_invalid_optional_may25_audit_never_included(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    invalid=tmp_path/"invalid-audit.json"
    invalid.write_text('{"status":"not_run"}')
    with pytest.raises(ValueError):
        package(src, out, invalid)
    assert not out.exists()


def test_ancillary_log_or_summary_tampering_rejected(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    for name in ("native_sumo.log", "submission_summary.md"):
        original = (src/name).read_bytes()
        (src/name).write_bytes(b"edited after successful research run")
        with pytest.raises(ReleaseError, match="Ancillary evidence hash mismatch"):
            package(src,out)
        assert not out.exists()
        (src/name).write_bytes(original)


def test_unhashed_ancillary_evidence_rejected(tmp_path):
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    manifest = src/"research_results_manifest.json"
    data = json.loads(manifest.read_text())
    del data["ancillary_sha256"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ReleaseError, match="complete research source set"):
        package(src,out)
    assert not out.exists()


def test_optional_may25_audit_changed_after_validation_rejected(tmp_path, monkeypatch):
    from streetlab_integration import research_audit_display
    src, out = tmp_path/"research", tmp_path/"release.zip"
    _fixture(src)
    audit = tmp_path/"completed-may25.json"
    audit.write_bytes(b"original validated bytes")
    def mutate_after_validation(path):
        Path(path).write_bytes(b"replaced after validation")
        return {"status": "verified_saved_audit_display"}
    monkeypatch.setattr(research_audit_display, "read_audit", mutate_after_validation)
    with pytest.raises(ReleaseError, match="May25 audit changed during verification"):
        package(src, out, audit)
    assert not out.exists()
