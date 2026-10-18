"""Read-only prospective source-intake tests, with fake file bytes only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from streetlab_integration.w04_holdout_intake import (
    IntakeError, digest, intake, main,
)


def receipts(tmp_path):
    video = tmp_path / "dev.mp4"
    truth = tmp_path / "dev.csv"
    video.write_bytes(b"fake development video")
    truth.write_bytes(b"fake development labels")
    lock = {
        "status": "PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT",
        "selection_scope": "DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION",
        "eligible_for_production": False,
        "selected_policy": "hard_nms_ios_0.30",
        "selected_settings": {
            "policy": "hard_nms", "threshold": .3, "metric": "ios",
            "class_agnostic": False, "score_floor": .15,
        },
        "source_development_batch_sha256": "a" * 64,
        "source_development_video_sha256": digest(video),
        "source_fluid_sha256": digest(truth),
        "source_model_sha256": "b" * 64,
        "predeclared_limits": {"min_holdout_frames": 120},
    }
    lock_file = tmp_path / "policy_lock.json"
    lock_file.write_text(json.dumps(lock))
    protocol = {
        "schema_version": 1,
        "status": "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED",
        "research_scope": "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY",
        "policy_lock_sha256": digest(lock_file),
        "selected_policy": "hard_nms_ios_0.30",
        "development_batch_sha256": "a" * 64,
        "source_video_sha256": digest(video),
        "source_fluid_sha256": digest(truth),
        "source_model_sha256": "b" * 64,
        "predeclared_numeric_gates": {"min_holdout_frames": 120},
        "new_source_established_untouched": False,
        "same_window_t000_baseline_available": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }
    protocol_file = tmp_path / "holdout_protocol.json"
    protocol_file.write_text(json.dumps(protocol))
    return protocol_file, lock_file, video, truth


def run(protocol, lock, **args):
    return intake(protocol_file=protocol, lock_file=lock, **args)


def test_missing_inputs_report_data_requirement_not_validation(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    result = run(protocol, lock)
    assert result["status"] == "W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH"
    assert result["candidate_truly_unseen_verified"] is False
    assert result["actual_holdout_scores_computed"] is False
    assert result["production_release_approved"] is False


def test_reusing_frozen_video_blocked_even_if_annotation_differs(tmp_path):
    protocol, lock, video, _ = receipts(tmp_path)
    new_truth = tmp_path / "alternative.csv"
    new_truth.write_text("different truth")
    result = run(protocol, lock, video=video, truth=new_truth)
    assert result["status"] == "W04_CANDIDATE_REJECTED_REUSED_DEVELOPMENT_SOURCE"
    assert result["reuses_development_video_sha"] is True
    assert result["candidate_bytes_distinct_from_development"] is False


def test_reusing_frozen_truth_blocked_even_if_video_differs(tmp_path):
    protocol, lock, _, truth = receipts(tmp_path)
    new_video = tmp_path / "different_camera.mp4"
    new_video.write_bytes(b"another camera video")
    result = run(protocol, lock, video=new_video, truth=truth)
    assert result["status"] == "W04_CANDIDATE_REJECTED_REUSED_DEVELOPMENT_SOURCE"
    assert result["reuses_development_fluid_sha"] is True


def test_different_source_hashes_never_certify_unseen(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    candidate = tmp_path / "new_camera.mp4"
    truth = tmp_path / "new_camera.csv"
    candidate.write_bytes(b"other video")
    truth.write_bytes(b"other annotations")
    result = run(protocol, lock, video=candidate, truth=truth)
    assert result["status"] == "W04_DISTINCT_BYTES_INDEPENDENCE_UNVERIFIED"
    assert result["candidate_bytes_distinct_from_development"] is True
    assert result["candidate_truly_unseen_verified"] is False
    assert result["same_window_baseline_and_alignment_verified"] is False
    assert result["independent_holdout_pass"] is False


def test_prior_manifest_roles_never_silently_upgrade_to_blind(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    candidate = tmp_path / "20250525_video.mp4"
    truth = tmp_path / "20250525_video_Traj.csv"
    candidate.write_bytes(b"regression date video")
    truth.write_bytes(b"regression labels")
    result = run(protocol, lock, video=candidate, truth=truth)
    assert result["status"] == "W04_HISTORICAL_DATASET_ROLE_REQUIRES_EXTERNAL_REVIEW"
    assert result["known_historical_video_role"] == "REGRESSION_REFERENCE"
    assert result["candidate_truly_unseen_verified"] is False


def test_missing_pair_does_not_claim_video_distinct(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    candidate = tmp_path / "some.mp4"
    candidate.write_bytes(b"some bytes")
    result = run(protocol, lock, video=candidate)
    assert result["status"] == "W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH"
    assert result["candidate_bytes_distinct_from_development"] is False


def test_claimed_original_t000_is_unverified_without_external_provenance(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    candidate = tmp_path / "fresh.mp4"
    truth = tmp_path / "fresh.csv"
    baseline = tmp_path / "original_t000.txt"
    candidate.write_bytes(b"fresh")
    truth.write_bytes(b"truth")
    baseline.write_bytes(b"baseline")
    result = run(protocol, lock, video=candidate, truth=truth,
                 baseline=baseline, baseline_kind="original_t000")
    assert result["candidate_baseline"]["status"] == "HASHED"
    assert result["same_window_baseline_and_alignment_verified"] is False
    assert any("Original T000 provenance" in s for s in result["reasons"])


@pytest.mark.parametrize("tamper", ["lock", "protocol", "limits", "policy"])
def test_tampered_lock_protocol_rejected(tmp_path, tamper):
    protocol, lock, _, _ = receipts(tmp_path)
    p = json.loads(protocol.read_text())
    l = json.loads(lock.read_text())
    if tamper == "lock":
        l["source_model_sha256"] = "x" * 64
        lock.write_text(json.dumps(l))
    elif tamper == "protocol":
        p["independent_holdout_pass"] = True
        protocol.write_text(json.dumps(p))
    elif tamper == "limits":
        l["predeclared_limits"]["min_holdout_frames"] = 1
        lock.write_text(json.dumps(l))
        p["policy_lock_sha256"] = digest(lock)
        protocol.write_text(json.dumps(p))
    else:
        l["selected_settings"]["threshold"] = .7
        lock.write_text(json.dumps(l))
        p["policy_lock_sha256"] = digest(lock)
        protocol.write_text(json.dumps(p))
    with pytest.raises(IntakeError):
        run(protocol, lock)


def test_cli_writes_new_file_never_overwrites(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    output = tmp_path / "intake.json"
    args = ["--protocol", str(protocol), "--policy-lock", str(lock),
            "--output", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH"
    assert result["candidate_truly_unseen_verified"] is False
    with pytest.raises(FileExistsError):
        main(args)


def test_symlink_not_treated_as_verified_source(tmp_path):
    protocol, lock, _, _ = receipts(tmp_path)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"candidate")
    symlink = tmp_path / "linked.mp4"
    symlink.symlink_to(other)
    truth = tmp_path / "candidate.csv"
    truth.write_bytes(b"new labels")
    result = run(protocol, lock, video=symlink, truth=truth)
    assert result["candidate_video"]["status"] == "SYMLINK_NOT_ACCEPTED"
    assert result["status"] == "W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH"
