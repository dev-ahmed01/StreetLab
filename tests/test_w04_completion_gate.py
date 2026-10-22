"""Synthetic receipt tests for finite six-phase research acceptance.

No GitHub CI test pretends it executed the user's private May25 MP4.
"""
import json

import pytest

from streetlab_integration import w04_completion_gate as gate
from streetlab_integration.w04_holdout_intake import digest


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return path


def receipts(tmp):
    root = tmp
    evidence = root / "artifacts"
    lockdir = evidence / "w04_policy_lock_test"
    run = evidence / "w04_may25_crossdate_test"
    forensicdir = evidence / "w04_may25_forensics_test"
    auditdir = evidence / "w04_autonomous_research_test"
    simdir = evidence / "research_submission_test"
    lock = {
        "status": "PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT",
        "selection_scope": "DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION",
        "eligible_for_production": False,
        "selected_policy": "hard_nms_ios_0.30",
        "selected_settings": {
            "policy": "hard_nms", "threshold": 0.3,
            "metric": "ios", "class_agnostic": False, "score_floor": 0.15
        },
        "source_development_batch_sha256": "a"*64,
        "source_development_video_sha256": "b"*64,
        "source_fluid_sha256": "c"*64,
        "source_model_sha256": "d"*64,
        "predeclared_limits": {"max_cpu_seconds_per_frame": 2.0},
    }
    lockp = write(lockdir / "policy_lock.json", lock)
    protocol = {
        "schema_version": 1,
        "status": "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED",
        "research_scope": "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY",
        "policy_lock_sha256": digest(lockp),
        "selected_policy": lock["selected_policy"],
        "development_batch_sha256": lock["source_development_batch_sha256"],
        "source_video_sha256": lock["source_development_video_sha256"],
        "source_fluid_sha256": lock["source_fluid_sha256"],
        "source_model_sha256": lock["source_model_sha256"],
        "predeclared_numeric_gates": lock["predeclared_limits"],
        "new_source_established_untouched": False,
        "same_window_t000_baseline_available": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }
    protop = write(lockdir / "holdout_protocol.json", protocol)
    pre = write(run / "preflight.json", {
        "status": "W04_MAY25_CROSSDATE_PREFLIGHT_VERIFIED_REGRESSION_ONLY",
        "source_video_sha256": "e"*64,
        "source_truth_sha256": "f"*64,
        "model_tree_sha256": "d"*64,
        "policy_lock_sha256": digest(lockp),
        "protocol_sha256": digest(protop),
    })
    summary = write(run / "cross_date_summary.json", {
        "status": "W04_MAY25_CROSSDATE_REGRESSION_RECORDED_NOT_BLIND_HOLDOUT",
        "cohort_role": "PREVIOUSLY_USED_MAY25_REGRESSION_REFERENCE",
        "selected_policy": "hard_nms_ios_0.30",
        "source_video_sha256": "e"*64,
        "original_fluid_sha256": "f"*64,
        "frozen_w04_model_sha256": "d"*64,
        "source_window": [60, 179], "decoded_frames": 150,
        "evaluation_frames": 120,
        "truth_label_f1": .9142034068,
        "per_class_fluid_label_diagnostics": {
            "HEAVY_VEHICLE": {"class_errors_on_spatial_matches": 236}
        },
        "original_saved_batch_sha256": "1"*64,
        "not_a_blind_or_untouched_holdout": True,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    })
    forensic = write(forensicdir / "regression_forensics.json", {
        "status": "W04_MAY25_REAL_REGRESSION_REPLAYED_WITH_REVIEW_QUEUE_NOT_BLIND",
        "original_video_and_FLUID_SHA_verified_from_saved_batch": True,
        "original_saved_scores_recomputed_from_tracks_and_FLUID": True,
        "saved_regression_summary_sha256": digest(summary),
        "preflight_sha256": digest(pre),
        "saved_regression_batch_sha256": "1"*64,
        "independent_holdout_pass": False,
    })
    auto = write(auditdir / "automated_object_audit.json", {
        "status": "W04_MAY25_AUTOMATED_LABEL_RELATIVE_OBJECT_AUDIT_COMPLETE",
        "original_saved_regression_summary_sha256": digest(summary),
        "original_source_forensics_sha256": digest(forensic),
        "HEAVY_wrong_class_matched_points": 236,
        "original_saved_FLUID_label_F1": .9142034068,
        "source_video_frame_window": [60, 179],
        "no_new_detector_inference": True,
        "no_FLUID_relabeling": True,
        "physical_object_labels_independently_adjudicated": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    })
    study = write(simdir / "submission_evidence.json", {
        "research_workflow_status":
            "RESEARCH_DEMO_EXECUTED_WITH_REPORTED_FAILED_REAL_WORLD_GENERALIZATION",
        "track": "RESEARCH_SUBMISSION_NO_NEW_FIELD_SURVEY",
        "field_census_claimed": False,
        "datasets_are_from_different_sites": True,
        "real_world_intervention_effect_proven": False,
        "native_sumo": {
            "status": "SUMO_EXECUTED_SYNTHETIC_ONLY",
            "site_specific_field_validation": False,
            "simulation_not_real_world_effect": True,
            "production_release_approved": False,
        },
        "dataset_summaries": {
            "chennai": {
                "holdout_generalization_pass": False,
                "capacity_gate_pass": False,
                "calibration_locked": True,
                "retuning_allowed": False,
            },
            "fluid_w04": {"scores_recomputed_in_this_run": False},
        },
        "external_site_transfer_validated": False,
        "production_release_approved": False,
    })
    return root, {"policy_lock": lockp, "protocol": protop,
                  "preflight": pre, "summary": summary,
                  "forensics": forensic, "audit": auto, "submission": study}


def test_empty_workspace_does_not_claim_completion(tmp_path):
    out = gate.evaluate(gate.sources(tmp_path))
    assert out["status"] == gate.BLOCKED
    assert out["phases_completed"] == 0
    assert len(out["phases"]) == 6
    assert out["production_release_approved"] is False


def test_full_simulated_receipts_still_no_independent_field_release(tmp_path):
    root, paths = receipts(tmp_path)
    out = gate.evaluate(gate.sources(root))
    assert out["status"] == gate.READY
    assert out["phases_completed"] == 6
    assert all(p["status"] == "COMPLETE" for p in out["phases"])
    assert out["independent_holdout_pass"] is False
    assert out["chennai_real_world_generalization_pass"] is False
    assert out["native_sumo_real_world_calibration_pass"] is False
    assert out["physical_vehicle_ground_truth_adjudicated"] is False
    assert out["production_release_approved"] is False
    assert "synthetic" in gate.markdown(out).lower()


def test_missing_autonomous_source_blocks_only_phase_four_and_final(tmp_path):
    root, paths = receipts(tmp_path)
    paths["audit"].unlink()
    out = gate.evaluate(gate.sources(root))
    assert out["phases_completed"] == 4
    assert out["phases"][3]["status"] == "BLOCKED"
    assert out["phases"][4]["status"] == "COMPLETE"
    assert out["phases"][5]["status"] == "BLOCKED"


@pytest.mark.parametrize("target,field,value", [
    ("summary", "production_release_approved", True),
    ("summary", "evaluation_frames", 119),
    ("forensics", "saved_regression_summary_sha256", "0"*64),
    ("policy_lock", "selected_policy", "changed"),
    ("audit", "HEAVY_wrong_class_matched_points", 999),
    ("audit", "independent_holdout_pass", True),
    ("submission", "external_site_transfer_validated", True),
])
def test_forged_or_promoted_evidence_fails_closed(tmp_path, target, field, value):
    root, paths = receipts(tmp_path)
    data = json.loads(paths[target].read_text())
    data[field] = value
    write(paths[target], data)
    out = gate.evaluate(gate.sources(root))
    assert out["status"] == gate.BLOCKED
    assert out["production_release_approved"] is False


def test_output_is_exclusive_and_creates_json_and_markdown(tmp_path):
    root, _ = receipts(tmp_path)
    destination = tmp_path / "completed_report"
    assert gate.main(["--project-root", str(root),
                      "--output-dir", str(destination)]) == 0
    assert (destination / "completion_verdict.md").is_file()
    assert json.loads((destination / "completion_verdict.json").read_text())[
        "status"] == gate.READY
    with pytest.raises(SystemExit):
        gate.main(["--project-root", str(root),
                   "--output-dir", str(destination)])


def test_auto_pairs_hash_consistent_evidence_not_latest_stale_folders(tmp_path):
    import os
    import time

    root, good = receipts(tmp_path)
    folder = root / "artifacts"
    bad_run = write(folder / "w04_may25_crossdate_stale" / "cross_date_summary.json",
                    {"status": "HALF_WRITTEN", "independent_holdout_pass": False})
    bad_forensics = write(folder / "w04_may25_forensics_stale" / "regression_forensics.json",
                          {"status": "FORGED", "saved_regression_summary_sha256": "0"*64})
    bad_audit = write(folder / "w04_autonomous_research_stale" / "automated_object_audit.json",
                      {"status": "UNRELATED", "original_saved_regression_summary_sha256": "0"*64})
    bad_study = write(folder / "research_submission_stale" / "submission_evidence.json",
                      {"native_sumo": {"status": "NOT_EXECUTED"}})
    bad_lock = write(folder / "w04_policy_lock_stale" / "policy_lock.json",
                      {"selected_policy": "retuned"})
    write(bad_lock.parent / "holdout_protocol.json", {"status": "RETUNED"})
    for i, p in enumerate([bad_run, bad_forensics, bad_audit, bad_study, bad_lock]):
        os.utime(p.parent, (time.time()+100+i, time.time()+100+i))
    chosen = gate.sources(root)
    for key in good:
        assert chosen[key] == good[key], key
    out = gate.evaluate(chosen)
    assert out["status"] == gate.READY
    assert out["production_release_approved"] is False

def test_mismatched_newer_but_valid_policy_lock_cannot_relabel_old_regression(tmp_path):
    root, paths = receipts(tmp_path)
    old = paths["policy_lock"]
    lock = json.loads(old.read_text())
    lock["source_model_sha256"] = "9"*64
    alternative = write(root/"artifacts"/"w04_policy_lock_new"/"policy_lock.json", lock)
    proto = json.loads(paths["protocol"].read_text())
    proto["policy_lock_sha256"] = digest(alternative)
    proto["source_model_sha256"] = "9"*64
    write(alternative.parent/"holdout_protocol.json", proto)
    # The legitimate lock must be selected by the preflight hashes, not by timestamp.
    selected = gate.sources(root)
    assert selected["policy_lock"] == paths["policy_lock"]
    assert gate.evaluate(selected)["status"] == gate.READY
    # Now make the old lock unavailable. An unrelated valid lock cannot complete.
    paths["policy_lock"].unlink()
    selected = gate.sources(root)
    assert selected["policy_lock"] is None
    assert gate.evaluate(selected)["status"] == gate.BLOCKED


@pytest.mark.parametrize("target,field,value", [
    ("submission", "field_census_claimed", True),
    ("submission", "real_world_intervention_effect_proven", True),
    ("submission", "datasets_are_from_different_sites", False),
    ("submission", "native_sumo", {"status": "SUMO_EXECUTED_SYNTHETIC_ONLY",
                                   "site_specific_field_validation": True,
                                   "simulation_not_real_world_effect": True,
                                   "production_release_approved": False}),
    ("submission", "dataset_summaries", {"chennai": {"holdout_generalization_pass": False}}),
    ("summary", "per_class_fluid_label_diagnostics", []),
    ("submission", "native_sumo", []),
    ("submission", "dataset_summaries", "invalid"),
])
def test_contradictory_or_malformed_nested_evidence_fails_closed(tmp_path, target, field, value):
    root, paths = receipts(tmp_path)
    data = json.loads(paths[target].read_text())
    data[field] = value
    write(paths[target], data)
    result = gate.evaluate(gate.sources(root))
    assert result["status"] == gate.BLOCKED
    assert result["production_release_approved"] is False
    assert len(result["phases"]) == 6


def test_original_policy_hash_relationship_required_for_completion(tmp_path):
    root, paths = receipts(tmp_path)
    pre = json.loads(paths["preflight"].read_text())
    pre["policy_lock_sha256"] = "0"*64
    write(paths["preflight"], pre)
    # Broken preflight/forensics link must fail rather than silently substitute.
    assert gate.evaluate(gate.sources(root))["status"] == gate.BLOCKED
