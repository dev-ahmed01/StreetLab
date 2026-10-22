"""Submission ZIP integrity and research-claim regression tests."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.verify_submission_artifact import VerificationError, verify


def sha(data):
    return sha256(data).hexdigest()


def fixture(folder: Path):
    guide = b"StreetLab submission research prototype"
    replay = b"canonical full-source audit still pending; 120 historical regression frames"
    (folder / "README_SUBMISSION.md").write_bytes(guide)
    (folder / "MAY25_FULL_COHORT_REPLAY_2026-10-23.md").write_bytes(replay)
    source = json.dumps({
        "research_workflow_status": "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION",
        "production_release_approved": False,
        "native_sumo": {"status": "SUMO_EXECUTED_SYNTHETIC_ONLY"},
        "dataset_summaries": {"chennai": {"holdout_generalization_pass": False}},
    }).encode()
    info = {
        "release_type": "BOUNDED_RESEARCH_PROTOTYPE",
        "native_sumo": "SYNTHETIC_ONLY",
        "may25_audit_status": "NOT_INCLUDED_OR_NOT_VERIFIED",
        "chennai_holdout_generalization_pass": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "files_sha256": {"README.txt": sha(b"limited"), "submission_evidence.json": sha(source)},
    }
    archive = folder / "streetlab_research_bundle.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("README.txt", b"limited")
        z.writestr("evidence/submission_evidence.json", source)
        z.writestr("manifest.json", json.dumps(info))
    receipt = {
        "schema_version": 1,
        "release_scope": "BOUNDED_RESEARCH_PROTOTYPE",
        "commit_sha": "a"*40,
        "native_sumo_execution": "ACTUAL_ENGINE_SYNTHETIC_NETWORK_ONLY",
        "may25_original_video_rehashed_in_ci": False,
        "canonical_may25_full_source_audit_verified_in_ci": False,
        "chennai_holdout_generalization_pass": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "research_bundle_sha256": sha(archive.read_bytes()),
        "submission_guide_sha256": sha(guide),
        "saved_cohort_replay_sha256": sha(replay),
    }
    (folder / "submission_build_receipt.json").write_text(json.dumps(receipt))
    return receipt


def test_valid_research_submission_is_integrity_pass_not_field_release(tmp_path):
    fixture(tmp_path)
    result = verify(tmp_path, "a"*40)
    assert result["verification_status"] == "INTEGRITY_PASS_SYNTHETIC_RESEARCH_ONLY"
    assert result["original_may25_video_verified"] is False
    assert result["production_release_approved"] is False


def test_wrong_expected_commit_rejected(tmp_path):
    fixture(tmp_path)
    with pytest.raises(VerificationError, match="requested commit"):
        verify(tmp_path, "b"*40)


@pytest.mark.parametrize("filename", [
    "README_SUBMISSION.md", "MAY25_FULL_COHORT_REPLAY_2026-10-23.md",
    "streetlab_research_bundle.zip",
])
def test_changed_artifact_bytes_rejected(tmp_path, filename):
    fixture(tmp_path)
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(VerificationError, match="SHA mismatch"):
        verify(tmp_path)


@pytest.mark.parametrize("field", [
    "production_release_approved", "independent_holdout_pass",
    "canonical_may25_full_source_audit_verified_in_ci",
])
def test_promoted_receipt_claim_rejected(tmp_path, field):
    receipt = fixture(tmp_path)
    receipt[field] = True
    (tmp_path / "submission_build_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(VerificationError, match="promoted"):
        verify(tmp_path)


def test_archive_added_unlisted_entry_rejected_even_with_updated_outer_sha(tmp_path):
    receipt = fixture(tmp_path)
    path = tmp_path / "streetlab_research_bundle.zip"
    with ZipFile(path, "a") as z:
        z.writestr("evidence/extra_unchecked_data.txt", "unverified")
    receipt["research_bundle_sha256"] = sha(path.read_bytes())
    (tmp_path / "submission_build_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(VerificationError, match="unlisted"):
        verify(tmp_path)


def test_production_claim_inside_zip_rejected_even_with_rehashed_receipt(tmp_path):
    receipt = fixture(tmp_path)
    original = tmp_path / "streetlab_research_bundle.zip"
    with ZipFile(original) as z:
        records = {name: z.read(name) for name in z.namelist()}
    manifest = json.loads(records["manifest.json"])
    manifest["production_release_approved"] = True
    records["manifest.json"] = json.dumps(manifest).encode()
    with ZipFile(original, "w") as z:
        for name, data in records.items():
            z.writestr(name, data)
    receipt["research_bundle_sha256"] = sha(original.read_bytes())
    (tmp_path / "submission_build_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(VerificationError, match="unsupported research claims"):
        verify(tmp_path)
