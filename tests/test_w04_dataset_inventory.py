"""Read-only candidate discovery tests. All videos and CSV rows are synthetic."""
from __future__ import annotations

import json

import pytest

from streetlab_integration import w04_dataset_inventory as mod
from streetlab_integration.w04_holdout_intake import digest


def setup_files(tmp_path):
    root = tmp_path / "video_repository"
    (root/"Video_2").mkdir(parents=True)
    (root/"annotations").mkdir()
    known = root/"Video_2"/"20250526_video.mp4"
    known.write_bytes(b"synthetic W04 reference")
    possible = root/"Video_2"/"20260601_video.mp4"
    possible.write_bytes(b"synthetic unrelated video")
    (root/"annotations"/"20250526_video_Traj.csv").write_text(
        "frame,id,cx,cy,type\n1,1,2,3,car\n")
    (root/"annotations"/"20260601_video_Traj.csv").write_text(
        "frame,id,cx,cy,type\n1,1,2,3,car\n")
    (root/"annotations"/"FIDRT_20260601_video.csv").write_text(
        "timestamp,x,y,speed\n1,2,3,4\n")
    lock = {
        "status": "PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT",
        "selection_scope": "DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION",
        "eligible_for_production": False,
        "selected_policy": "hard_nms_ios_0.30",
        "selected_settings": {
            "policy": "hard_nms", "threshold": .3, "metric": "ios",
            "class_agnostic": False, "score_floor": .15,
        },
        "source_development_batch_sha256": "a"*64,
        "source_development_video_sha256": "v"*64,
        "source_fluid_sha256": "f"*64,
        "source_model_sha256": "m"*64,
        "predeclared_limits": {"max_cpu_seconds_per_frame": 2.0},
    }
    lock_file = tmp_path/"policy_lock.json"
    lock_file.write_text(json.dumps(lock))
    protocol = {
        "schema_version": 1,
        "status": "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED",
        "research_scope": "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY",
        "policy_lock_sha256": digest(lock_file),
        "selected_policy": "hard_nms_ios_0.30",
        "development_batch_sha256": "a"*64,
        "source_video_sha256": "v"*64,
        "source_fluid_sha256": "f"*64,
        "source_model_sha256": "m"*64,
        "predeclared_numeric_gates": {"max_cpu_seconds_per_frame": 2.0},
        "new_source_established_untouched": False,
        "same_window_t000_baseline_available": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }
    protocol_file = tmp_path/"holdout_protocol.json"
    protocol_file.write_text(json.dumps(protocol))
    manifest_file = tmp_path/"manifest.json"
    manifest_file.write_text(json.dumps({"studies": [
        {"study_id": "fluid_fidrt_20250526", "role": "held_out_validation",
         "video": "20250526_video.mp4", "tracks": "20250526_video_Traj.csv"},
        {"study_id": "fluid_fidrt_20250525", "role": "regression_reference",
         "video": "20250525_video.mp4", "tracks": "20250525_video_Traj.csv"},
    ]}))
    return root, protocol_file, lock_file, manifest_file


def inspect(tmp_path, **kwargs):
    root, protocol, lock, manifest = setup_files(tmp_path)
    return mod.inventory(roots=[root], protocol_file=protocol,
                         lock_file=lock, manifest_file=manifest, **kwargs)


def test_known_tuned_pair_never_claims_blind(tmp_path):
    result = inspect(tmp_path)
    assert result["status"] == mod.STATUS
    assert result["candidate_video_count"] == 2
    assert result["candidate_fluid_label_csv_count"] == 2
    assert result["video_count_with_possible_truth"] == 2
    assert result["source_video_frame_decoding_not_performed"] is True
    assert result["video_sha256_not_computed_for_bulk_inventory"] is True
    assert result["new_independent_test_verified"] is False
    assert result["production_release_approved"] is False
    known = next(v for v in result["videos"] if "20250526" in v["video_path"])
    assert known["source_history"] == "TUNED_DEVELOPMENT_NO_BLIND_USE"
    assert known["historical_role"] == "held_out_validation"
    assert known["possible_fluid_truths"][0]["match_basis"] == "MANIFEST_NAMED_PAIR_NOT_CONTENT_VALIDATED"
    other = next(v for v in result["videos"] if "20260601" in v["video_path"])
    assert other["source_history"] == "UNKNOWN_NOT_AUTHENTICATED_AS_UNSEEN"
    assert other["untouched_holdout_verified"] is False
    assert len(result["incompatible_csv_files"]) == 1


def test_bounded_scan_does_not_descend_into_artifacts(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    hidden = root/"artifacts"/"private"
    hidden.mkdir(parents=True)
    (hidden/"private_video.mp4").write_bytes(b"not to walk")
    r = mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest)
    assert r["candidate_video_count"] == 2


def test_depth_limit_reported_pair_not_false_when_files_out_of_scope(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    r = mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest,
                      max_depth=0)
    assert r["candidate_video_count"] == 0
    assert r["video_count_with_possible_truth"] == 0
    assert r["new_independent_test_verified"] is False


def test_fail_closed_on_truncation(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    with pytest.raises(mod.InventoryError, match="scan cap"):
        mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest, max_entries=2)


def test_frozen_lock_tampering_blocks_inventory(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    p = json.loads(protocol.read_text())
    p["independent_holdout_pass"] = True
    protocol.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="aligned"):
        mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest)


def test_no_truth_pairs_not_suggested_as_labelled(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    for item in (root/"annotations").iterdir():
        item.unlink()
    r = mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest)
    assert r["candidate_fluid_label_csv_count"] == 0
    assert r["video_count_with_possible_truth"] == 0
    assert all(not item["possible_fluid_truths"] for item in r["videos"])


def test_only_same_date_is_ambiguously_matched(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    (root/"annotations"/"20260601_other_camera.csv").write_text("frame,id,cx,cy,type\n")
    r = mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest)
    record = next(x for x in r["videos"] if "20260601" in x["video_path"])
    matches = record["possible_fluid_truths"]
    assert matches[0]["match_basis"] == "SHARED_VIDEO_STEM_NOT_CONTENT_VALIDATED"
    assert any(x["match_basis"] == "SHARED_DATE_POSSIBLE_MISMATCH_REVIEW_REQUIRED"
               for x in matches)


def test_symlinks_skipped_in_inventory(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    linked = root/"video_link.mp4"
    linked.symlink_to(root/"Video_2"/"20250526_video.mp4")
    r = mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest)
    assert r["candidate_video_count"] == 2
    assert r["scan"]["symlinks_skipped"] >= 1


def test_cli_output_created_once(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    output = tmp_path/"inventory.json"
    args = [
        "--root", str(root), "--protocol", str(protocol),
        "--policy-lock", str(lock), "--manifest", str(manifest),
        "--output", str(output),
    ]
    assert mod.main(args) == 0
    assert json.loads(output.read_text())["candidate_video_count"] == 2
    with pytest.raises(FileExistsError):
        mod.main(args)


def test_manifest_absence_and_bad_limits_block(tmp_path):
    root, protocol, lock, manifest = setup_files(tmp_path)
    manifest.write_text("{}")
    with pytest.raises(mod.InventoryError, match="manifest"):
        mod.inventory(roots=[root], protocol_file=protocol,
                      lock_file=lock, manifest_file=manifest)
    for kwargs in ({"max_depth": -1}, {"max_entries": 0}):
        with pytest.raises(mod.InventoryError, match="Unsafe"):
            mod.inventory(roots=[root], protocol_file=protocol,
                          lock_file=lock, manifest_file=manifest, **kwargs)
