"""Synthetic-only W04 replay contract tests; no FLUID benchmark pass is simulated."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest

from streetlab_phase3.video.holdout_validation import sha
from streetlab_integration.w04_score_replay import replay, ReplayError


def fixture(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    truth = tmp_path / "fluid.csv"
    truth.write_text("synthetic FLUID truth fixture", encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic video only")
    model_hash = "c" * 64
    origin = {
        "video_path": str(video), "video_sha256": sha(video),
        "fluid_path": str(truth), "fluid_sha256": sha(truth),
        "openvino_model_sha256": model_hash,
        "video_frame_window": [10750, 10950],
        "eligible_for_promotion": False,
    }
    provenance = batch / "source_provenance.json"
    provenance.write_text(json.dumps(origin))
    tracks = batch / "hard_nms_ios_0.30.txt"
    tracks.write_text("synthetic test tracks", encoding="utf-8")
    pixel = {
        "truth_points": 100, "predicted_points": 95, "matched_points": 85,
        "frame_offset": 1, "max_distance_px": 50.0,
        "point_recall": 0.85, "point_precision": 85/95,
        "evaluation_frame_start": 10751, "evaluation_frame_end": 10951,
    }
    identity = {
        "truth_tracks": 40, "predicted_tracks": 38, "matched_truth_tracks": 35,
        "frame_offset": 1, "max_distance_px": 50.0,
        "fraction_truth_tracks_fragmented": 0.12,
        "total_contiguous_id_switches": 5,
    }
    classes = {
        "truth_points": 100, "predicted_points": 95,
        "frame_offset": 1, "max_pixel_distance": 50.0,
        "by_class": {
            name: {"truth_points": 10, "correct_class_matches": 8,
                   "correct_class_recall": 0.8}
            for name in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
        },
    }
    policy = {
        "name": "hard_nms_ios_0.30", "tracks": tracks.name,
        "pixel": pixel, "identity": identity,
        "class_diagnostics": classes,
    }
    report = {
        "policies": [policy], "evaluated_first_frame": 10750,
        "evaluated_last_frame": 10950, "warmup_frames_processed": 90,
        "same_window_T000": None,
    }
    verified = {
        "original_source_sha256_verified": True, "report": report,
        "provenance": origin, "batch_sha256": "b" * 64,
        "policy_count": 1,
    }
    receipt = tmp_path / "readiness.json"
    receipt.write_text(json.dumps({
        "status": "W04_LOCAL_DEVELOPMENT_ASSETS_VERIFIED",
        "source_video_and_model_verified": True,
        "matching_fluid_truth_verified": True,
        "independent_holdout_pass": False,
        "eligible_for_production": False,
        "files": {
            "provenance": {"sha256": sha(provenance)},
            "original_video": {"sha256_verified": True,
                               "sha256": sha(video)},
            "fluid_truth": {"sha256_verified": True,
                            "sha256": sha(truth)},
            "openvino_model": {"sha256_verified": True,
                               "sha256": model_hash},
        },
    }))
    class FakeScorer:
        def __init__(self, bad=None):
            self.bad = bad
        def __call__(self, **kwargs):
            value = copy.deepcopy(pixel)
            if self.bad == "precision":
                value["point_precision"] += 0.01
            return value, copy.deepcopy(identity)
        def class_report(self, path):
            value = copy.deepcopy(classes)
            if self.bad == "class":
                value["by_class"]["MOTORCYCLE"]["correct_class_recall"] = 0.1
            return value

    def factory(*args, bad=None):
        return FakeScorer(bad)

    return batch, receipt, verified, factory


def test_synthetic_injection_never_claims_genuine_scoring(tmp_path):
    batch, receipt, verified, factory = fixture(tmp_path)
    result = replay(batch_dir=batch, readiness_file=receipt,
                    verified_batch=verified, scorer_factory=factory)
    assert result["status"] == "W04_SYNTHETIC_TEST_INJECTION_NOT_SOURCE_REPLAY"
    assert result["policy_count"] == 1
    assert result["scorecards"][0]["verified_metric_fields"] > 10
    assert result["same_window_t000_available"] is False
    assert result["independent_heldout_accuracy_claimed"] is False
    assert result["new_detector_inference_performed"] is False
    assert result["production_release_approved"] is False


@pytest.mark.parametrize("kind", ["precision", "class"])
def test_different_saved_and_replayed_metrics_fail_closed(tmp_path, kind):
    batch, receipt, verified, factory = fixture(tmp_path)
    with pytest.raises(ReplayError, match="differs"):
        replay(batch_dir=batch, readiness_file=receipt,
               verified_batch=verified,
               scorer_factory=lambda *a: factory(bad=kind))


def test_tampered_provenance_does_not_use_old_receipt(tmp_path):
    batch, receipt, verified, factory = fixture(tmp_path)
    (batch / "source_provenance.json").write_text('{"changed":true}')
    with pytest.raises(ReplayError, match="provenance"):
        replay(batch_dir=batch, readiness_file=receipt,
               verified_batch=verified, scorer_factory=factory)


def test_readiness_receipt_with_missing_truth_refused(tmp_path):
    batch, receipt, verified, factory = fixture(tmp_path)
    val = json.loads(receipt.read_text())
    val["matching_fluid_truth_verified"] = False
    receipt.write_text(json.dumps(val))
    with pytest.raises(ReplayError, match="Readiness receipt"):
        replay(batch_dir=batch, readiness_file=receipt,
               verified_batch=verified, scorer_factory=factory)


def test_changed_development_window_refused(tmp_path):
    batch, receipt, verified, factory = fixture(tmp_path)
    verified["report"]["evaluated_first_frame"] = 10800
    with pytest.raises(ReplayError, match="frozen 201-frame"):
        replay(batch_dir=batch, readiness_file=receipt,
               verified_batch=verified, scorer_factory=factory)


def test_receipt_source_sha_mismatch_fails(tmp_path):
    batch, receipt, verified, factory = fixture(tmp_path)
    val = json.loads(receipt.read_text())
    val["files"]["openvino_model"]["sha256"] = "f" * 64
    receipt.write_text(json.dumps(val))
    with pytest.raises(ReplayError, match="does not match"):
        replay(batch_dir=batch, readiness_file=receipt,
               verified_batch=verified, scorer_factory=factory)
