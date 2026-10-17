"""Prospective W04 policy-lock tests use synthetic cohort records only."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from streetlab_integration import w04_preregistration as w04
from streetlab_phase3.video.continuous_box_tracking_lab import policy_matrix
from streetlab_phase3.video.holdout_validation import LOCK_STATUS


def fake_receipts():
    names = [x["name"] for x in policy_matrix()]
    assert len(names) == 25
    batch_policies, replay_cards = [], []
    for i, name in enumerate(names):
        if name == "hard_nms_ios_0.30":
            precision, recall, switches = .867, .914, 71
        elif name == "raw_unmerged":
            precision, recall, switches = .557, .929, 908
        else:
            precision, recall, switches = .70 + (i % 5)*.005, .915, 90+i
        h = hashlib.sha256(name.encode()).hexdigest()
        class_data = {}
        for cls in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE"):
            n = 0 if cls == "BUS" else 30
            class_data[cls] = {
                "truth_points": n,
                "correct_class_recall": None if not n else (.81 if name == "hard_nms_ios_0.30" else .85),
            }
        batch_policies.append({
            "name": name, "tracks_sha256": h,
            "pixel": {"point_precision": precision, "point_recall": recall},
            "identity": {"total_contiguous_id_switches": switches},
            "class_diagnostics": {"by_class": class_data},
            "postprocess_median_seconds": .01,
            "tracking_median_seconds": .01,
        })
        replay_cards.append({
            "policy": name, "track_sha256": h, "verified_metric_fields": 66,
            "previously_saved_precision": precision,
            "previously_saved_recall": recall,
            "previously_saved_identity_switches": switches,
        })
    batch = {
        "eligible_for_promotion": False,
        "evaluated_first_frame": 10750,
        "evaluated_last_frame": 10950,
        "warmup_frames_processed": 90,
        "same_window_T000": None,
        "detector_median_seconds": 1.25,
        "policies": batch_policies,
    }
    provenance = {
        "eligible_for_promotion": False, "fluid_sha256": "f"*64,
        "video_sha256": "a"*64, "openvino_model_sha256": "b"*64,
        "bundle_sha256": "c"*64,
    }
    verified = {
        "original_source_sha256_verified": True,
        "report": batch,
        "provenance": provenance,
        "batch_sha256": "d"*64, "policy_count": 25,
    }
    replay = {
        "status": "W04_DEVELOPMENT_SCORE_REPLAY_MATCHED_ORIGINAL_FLUID",
        "scope": "ORIGINAL_W04_TUNED_DEVELOPMENT_LABEL_REPLAY_ONLY",
        "source_window": [10750, 10950],
        "evaluation_fluid_frame_window": [10751, 10951],
        "source_video_model_truth_verified_against_frozen_provenance": True,
        "reuses_previously_evaluated_development_window": True,
        "same_window_t000_available": False,
        "independent_heldout_accuracy_claimed": False,
        "new_detector_inference_performed": False,
        "physical_object_labels_independently_adjudicated": False,
        "production_release_approved": False,
        "original_batch_sha256": "d"*64,
        "original_fluid_sha256": "f"*64,
        "policy_count": 25,
        "scorecards": replay_cards,
    }
    manifest = {"studies": [
        {"study_id": "fluid_fidrt_20250526", "role": "held_out_validation"},
        {"study_id": "fluid_fidrt_20250120", "role": "development_calibration"},
        {"study_id": "fluid_fidrt_20250525", "role": "regression_reference"},
    ]}
    return replay, verified, manifest


def call(replay, verified, manifest, policy="hard_nms_ios_0.30", budget=2.):
    return w04.assess(replay, verified, replay_sha="e"*64,
                      policy=policy, cpu_budget_s=budget, manifest=manifest)


def test_selected_dev_leader_remains_independent_holdout_blocked():
    replay, verified, manifest = fake_receipts()
    result = call(replay, verified, manifest)
    assert result["selected_policy"] == "hard_nms_ios_0.30"
    assert result["selected_development_metrics"]["f1"] > .88
    assert result["development_leaderboard_top_five"][0]["policy"] == result["selected_policy"]
    assert result["correct_class_tradeoffs_vs_raw"]["BUS"]["interpretation"] == "NOT_MEASURABLE_ZERO_TRUTH"
    assert result["new_source_established_untouched"] is False
    assert result["end_to_end_cpu_p95_measured"] is False
    assert result["production_release_approved"] is False
    assert all(x["established_unseen_for_this_preregistration"] is False
               for x in result["original_source_study_roles"])


@pytest.mark.parametrize("change", ["status", "holdout", "batch_sha", "fluid_sha", "score",
                                     "track_sha", "fields", "duplicate", "source_window",
                                     "fake_t000", "new_policy"])
def test_any_replay_or_batch_inconsistency_fails_closed(change):
    replay, verified, manifest = fake_receipts()
    if change == "status":
        replay["status"] = "W04_SYNTHETIC_TEST_INJECTION_NOT_SOURCE_REPLAY"
    elif change == "holdout":
        replay["independent_heldout_accuracy_claimed"] = True
    elif change == "batch_sha":
        replay["original_batch_sha256"] = "a"*64
    elif change == "fluid_sha":
        replay["original_fluid_sha256"] = "a"*64
    elif change == "score":
        replay["scorecards"][0]["previously_saved_recall"] = .01
    elif change == "track_sha":
        replay["scorecards"][0]["track_sha256"] = "9"*64
    elif change == "fields":
        replay["scorecards"][0]["verified_metric_fields"] = 5
    elif change == "duplicate":
        replay["scorecards"][1]["policy"] = replay["scorecards"][0]["policy"]
    elif change == "source_window":
        replay["source_window"] = [10700, 10950]
    elif change == "fake_t000":
        verified["report"]["same_window_T000"] = {"claimed": "new"}
    elif change == "new_policy":
        verified["report"]["policies"][0]["name"] = "injected"
    with pytest.raises(w04.ProtocolError):
        call(replay, verified, manifest)


def test_role_alias_does_not_establish_unseen_holdout():
    replay, verified, manifest = fake_receipts()
    manifest["studies"][0]["role"] = "independent_blind"
    with pytest.raises(w04.ProtocolError, match="manifest"):
        call(replay, verified, manifest)


def test_policy_override_refused_without_new_protocol():
    replay, verified, manifest = fake_receipts()
    with pytest.raises(w04.ProtocolError, match="pinned"):
        call(replay, verified, manifest, policy="hard_nms_ios_0.50")


def test_invalid_cpu_limit_blocked():
    replay, verified, manifest = fake_receipts()
    for limit in (0., -1., float("nan"), 62.):
        with pytest.raises(w04.ProtocolError):
            call(replay, verified, manifest, budget=limit)


def test_prepare_uses_existing_phase3_lock_contract_and_never_overwrites(tmp_path, monkeypatch):
    replay, verified, manifest = fake_receipts()
    project = tmp_path/"project"
    (project/"data/benchmarks/fluid_fidrt").mkdir(parents=True)
    (project/"data/benchmarks/fluid_fidrt/manifest.json").write_text(json.dumps(manifest))
    replay_path = tmp_path/"replay.json"
    replay_path.write_text(json.dumps(replay))
    monkeypatch.setattr(w04, "verify_batch", lambda path: verified)
    def fake_lock(batch_dir, policy, output, *, max_cpu_seconds_per_frame):
        value = {
            "status": LOCK_STATUS, "selected_policy": policy,
            "max_cpu_seconds_per_frame": max_cpu_seconds_per_frame,
            "eligible_for_production": False,
        }
        output.write_text(json.dumps(value))
        return value
    monkeypatch.setattr(w04, "lock_policy", fake_lock)
    destination = tmp_path/"protocol01"
    result = w04.prepare(batch_dir=tmp_path, replay_file=replay_path,
                         output_dir=destination, project_root=project)
    assert result["status"] == w04.STATUS
    assert result["development_sum_of_stage_medians_seconds_per_frame"] == pytest.approx(1.27)
    assert (destination/"policy_lock.json").is_file()
    saved = json.loads((destination/"holdout_protocol.json").read_text())
    assert saved["policy_lock_sha256"] == w04.sha(destination/"policy_lock.json")
    assert saved["independent_holdout_pass"] is False
    with pytest.raises(FileExistsError):
        w04.prepare(batch_dir=tmp_path, replay_file=replay_path,
                    output_dir=destination, project_root=project)


def test_known_unachievable_cpu_budget_refused_before_writing(tmp_path, monkeypatch):
    replay, verified, manifest = fake_receipts()
    project = tmp_path/"project"
    (project/"data/benchmarks/fluid_fidrt").mkdir(parents=True)
    (project/"data/benchmarks/fluid_fidrt/manifest.json").write_text(json.dumps(manifest))
    replay_path = tmp_path/"replay.json"
    replay_path.write_text(json.dumps(replay))
    monkeypatch.setattr(w04, "verify_batch", lambda path: verified)
    destination = tmp_path/"unwritten"
    with pytest.raises(w04.ProtocolError, match="budget"):
        w04.prepare(batch_dir=tmp_path, replay_file=replay_path,
                    output_dir=destination, project_root=project,
                    max_cpu_seconds_per_frame=.9)
    assert not destination.exists()
