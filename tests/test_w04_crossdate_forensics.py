"""Real-source forensic audit safety contracts, using synthetic tracks/labels only."""
from __future__ import annotations

import json

import pytest

from streetlab_integration import w04_crossdate_forensics as lab
from streetlab_integration.w04_holdout_intake import digest
from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth
from streetlab_phase3.video.class_diagnostics import class_diagnostics


def pred(frame, tid, x, label):
    return GeoTraxPixelPoint(frame, tid, float(x), 100., float(x), 100., label, .85)


def gt(frame, tid, x, label):
    return FluidPixelTruth(frame, tid, float(x), 100., label)


def observations():
    # The sample intentionally models HEAVY class confusion, an actual
    # matching BUS->CAR (not BUS ground truth), and a BUS unmatched prediction.
    truth = [
        gt(61, "h1", 100, "HEAVY_VEHICLE"),
        gt(61, "m1", 200, "MOTORCYCLE"),
        gt(61, "c1", 300, "CAR"),
        gt(61, "h2", 400, "HEAVY_VEHICLE"),
        gt(62, "h3", 250, "HEAVY_VEHICLE"),
    ]
    predicted = [
        pred(60, "P1", 100, "CAR"),
        pred(60, "P2", 300, "BUS"),
        pred(60, "P3", 600, "BUS"),
        pred(61, "P4", 250, "HEAVY_VEHICLE"),
    ]
    return predicted, truth


def test_review_queue_preserves_true_vs_pred_class_counts_and_frame_indices():
    predicted, truth = observations()
    result = lab.review_candidates(predicted, truth)
    assert result["spatial_matches_in_original_fixed_cohort"] == 3
    assert result["total_candidate_events_by_category"] == {
        "BUS_PRED_MATCHED_NONBUS_TRUTH": 1,
        "BUS_PRED_UNMATCHED": 1,
        "HEAVY_TRUTH_UNMATCHED": 1,
        "HEAVY_WRONG_CLASS": 1,
        "MOTORCYCLE_TRUTH_UNMATCHED": 1,
    }
    assert result["exported_review_frames"] == 1
    assert result["frames"][0]["video_frame"] == 60
    assert result["frames"][0]["fluid_frame"] == 61
    assert all(e["annotation_status"] ==
               "FLUID_LABEL_ONLY_NOT_PHYSICALLY_ADJUDICATED" for e in result["events"])
    assert result["unmatched_BUS_predictions_are_not_proven_false_objects"] is True
    assert result["review_is_not_independent_physical_truth"] is True


def test_review_queue_is_bounded_and_deterministic():
    predicted, truth = observations()
    a = lab.review_candidates(predicted, truth, max_frames=1, max_events=2)
    b = lab.review_candidates(predicted, truth, max_frames=1, max_events=2)
    assert a == b
    assert len(a["events"]) == 2
    assert a["events"][0]["event"] == "HEAVY_WRONG_CLASS"
    assert a["events"][1]["event"] == "BUS_PRED_MATCHED_NONBUS_TRUTH"


def test_invalid_review_limits_fail_closed():
    for options in ({"max_frames": 0}, {"max_events": 0}):
        with pytest.raises(lab.AuditError, match="caps"):
            lab.review_candidates([], [], **options)


def test_nested_score_mutation_or_field_drop_detected():
    v = {"foo": [{"value": .25, "gt": None}]}
    lab._check_equal(v, {"foo": [{"value": .25, "gt": None}]}, "test")
    with pytest.raises(lab.AuditError):
        lab._check_equal(v, {"foo": [{"value": .55, "gt": None}]}, "test")
    with pytest.raises(lab.AuditError):
        lab._check_equal(v, {"foo": [{"value": .25}]}, "test")
    with pytest.raises(lab.AuditError):
        lab._check_equal(v, {"foo": [{"value": float("nan"), "gt": None}]}, "test")


def fixture(tmp_path):
    predicted, truth = observations()
    cls = class_diagnostics(predicted, truth, start_frame=60, end_frame=179)
    run = tmp_path / "original_run"
    proto = tmp_path / "original_protocol"
    run.mkdir()
    proto.mkdir()
    batch = run / "batch"
    batch.mkdir()
    tracks = batch / "hard_nms_ios_0.30.txt"
    tracks.write_text("synthetic tracks test only")
    actual_truth = tmp_path / "may25_truth.csv"
    actual_truth.write_text("synthetic annotations test only")
    lock = {
        "status": "PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT",
        "selection_scope": "DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION",
        "eligible_for_production": False,
        "selected_policy": "hard_nms_ios_0.30",
        "selected_settings": {
            "name": "hard_nms_ios_0.30", "policy": "hard_nms", "threshold": .3,
            "metric": "ios", "score_floor": .15, "sigma": .5,
            "class_agnostic": False,
        },
        "source_development_batch_sha256": "a"*64,
        "source_development_video_sha256": "b"*64,
        "source_fluid_sha256": "f"*64,
        "source_model_sha256": "d"*64,
        "predeclared_limits": {"max_cpu_seconds_per_frame": 2.},
    }
    lock_file = proto / "policy_lock.json"
    lock_file.write_text(json.dumps(lock))
    dev_classes = {
        label: {"truth_points": 100 if label != "BUS" else 0,
                "correct_class_recall": .6 if label != "BUS" else None}
        for label in lab.TRUTH_CLASSES
    }
    dev = {
        "policy": "hard_nms_ios_0.30", "precision": .8, "recall": .5,
        "f1": 2*.8*.5/(.8+.5), "contiguous_id_switches": 10,
        "class_diagnostics": dev_classes,
    }
    protocol = {
        "schema_version": 1, "status": "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED",
        "research_scope": "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY",
        "policy_lock_sha256": digest(lock_file),
        "selected_policy": "hard_nms_ios_0.30",
        "development_batch_sha256": "a"*64,
        "source_video_sha256": "b"*64,
        "source_fluid_sha256": "f"*64,
        "source_model_sha256": "d"*64,
        "selected_development_metrics": dev,
        "predeclared_numeric_gates": {"max_cpu_seconds_per_frame": 2.},
        "new_source_established_untouched": False,
        "same_window_t000_baseline_available": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }
    protocol_file = proto / "holdout_protocol.json"
    protocol_file.write_text(json.dumps(protocol))
    pixel = {"point_precision": .75, "point_recall": .6,
             "frame_offset": 1, "max_distance_px": 50.,
             "evaluation_frame_start": 61, "evaluation_frame_end": 180}
    identity = {"total_contiguous_id_switches": 2,
                "frame_offset": 1, "max_distance_px": 50.}
    p_f1 = 2*.75*.6/(.75+.6)
    summary = {
        "status": "W04_MAY25_CROSSDATE_REGRESSION_RECORDED_NOT_BLIND_HOLDOUT",
        "cohort_role": "PREVIOUSLY_USED_MAY25_REGRESSION_REFERENCE",
        "selected_policy": "hard_nms_ios_0.30",
        "source_video_sha256": "c"*64,
        "original_fluid_sha256": "e"*64,
        "frozen_w04_model_sha256": "d"*64,
        "source_window": [60, 179], "truth_frame_window": [61, 180],
        "source_video_dimensions": [3840, 2160],
        "source_frame_rate_reported": 10.,
        "decoded_frames": 150, "evaluation_frames": 120,
        "truth_label_precision": .75, "truth_label_recall": .6,
        "truth_label_f1": p_f1, "contiguous_id_switches": 2,
        "per_class_fluid_label_diagnostics": cls["by_class"],
        "detector_cpu_median_seconds_per_frame": 1.2,
        "tracker_cpu_median_seconds_per_frame": .01,
        "postprocess_cpu_median_seconds_per_frame": .02,
        "original_saved_batch_sha256": "g"*64,
        "not_a_blind_or_untouched_holdout": True,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }
    (run / "cross_date_summary.json").write_text(json.dumps(summary))
    preflight = {
        "status": "W04_MAY25_CROSSDATE_PREFLIGHT_VERIFIED_REGRESSION_ONLY",
        "policy_lock_sha256": digest(lock_file),
        "protocol_sha256": digest(protocol_file),
        "source_video_sha256": "c"*64,
        "source_truth_sha256": "e"*64,
        "model_tree_sha256": "d"*64,
        "source_video_dimensions": [3840, 2160],
        "source_frame_rate_reported": 10.,
        "fixed_source_evaluation_frames": [60, 179],
        "fixed_fluid_evaluation_frames": [61, 180],
        "tracker_warmup_frames": 30,
    }
    (run / "preflight.json").write_text(json.dumps(preflight))
    policy = {
        "name": "hard_nms_ios_0.30",
        "settings": lock["selected_settings"],
        "tracks": tracks.name,
        "pixel": pixel, "identity": identity,
        "class_diagnostics": cls,
        "tracking_median_seconds": .01,
        "postprocess_median_seconds": .02,
    }
    batch_record = {
        "evaluated_first_frame": 60, "evaluated_last_frame": 179,
        "first_decoded_frame": 30, "warmup_frames_processed": 30,
        "processed_frames": 150, "detector_median_seconds": 1.2,
        "policies": [policy],
    }
    origin = {
        "source_is_independent_holdout": False,
        "video_sha256": "c"*64, "fluid_sha256": "e"*64,
        "fluid_path": str(actual_truth),
        "openvino_model_sha256": "d"*64,
    }
    verified = {
        "original_source_sha256_verified": True,
        "report": batch_record, "provenance": origin,
        "policy_count": 1, "batch_sha256": "g"*64,
    }
    class FakeScorer:
        def __call__(self, **kwargs):
            return pixel, identity
        def class_report(self, t):
            return cls
    options = {
        "verifier": lambda path: verified,
        "score_factory": lambda path, left, right: FakeScorer(),
        "track_reader": lambda path: predicted,
        "truth_reader": lambda path: truth,
    }
    return run, proto, options


def test_injected_synthetic_audit_recomputes_deltas_but_cannot_claim_real_video(tmp_path):
    run, protocol, options = fixture(tmp_path)
    report = lab.audit(run_dir=run, protocol_folder=protocol, **options)
    assert report["status"] == "SYNTHETIC_FORENSICS_TEST_NOT_AUTHENTIC_VIDEO_AUDIT"
    assert report["original_saved_scores_recomputed_from_tracks_and_FLUID"] is False
    assert report["independent_holdout_pass"] is False
    assert report["May25"]["f1"] == pytest.approx(2*.75*.6/(.75+.6))
    assert report["difference_May25_minus_tuned_W04"]["precision_percentage_points"] == pytest.approx(-5)
    assert report["by_class_comparison"]["BUS"]["difference_percentage_points"] is None
    assert report["review"]["total_candidate_events"] == 5


@pytest.mark.parametrize("field", ["truth_label_precision", "truth_label_f1",
                                  "source_video_sha256", "independent_holdout_pass"])
def test_audit_fails_closed_on_modified_saved_summary(tmp_path, field):
    run, protocol, options = fixture(tmp_path)
    path = run / "cross_date_summary.json"
    value = json.loads(path.read_text())
    value[field] = True if field == "independent_holdout_pass" else 0.99
    path.write_text(json.dumps(value))
    with pytest.raises((lab.AuditError, ValueError)):
        lab.audit(run_dir=run, protocol_folder=protocol, **options)


def test_audit_rejects_replaced_policy_lock(tmp_path):
    run, protocol, options = fixture(tmp_path)
    lock = protocol / "policy_lock.json"
    obj = json.loads(lock.read_text())
    obj["selected_settings"]["threshold"] = .7
    lock.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        lab.audit(run_dir=run, protocol_folder=protocol, **options)


def test_real_audit_requires_saved_batch_and_true_source_sha(tmp_path):
    run, protocol, _ = fixture(tmp_path)
    with pytest.raises((OSError, ValueError)):
        lab.audit(run_dir=run, protocol_folder=protocol)


def test_cli_writes_json_and_csv_without_overwriting_previous_receipts(tmp_path, monkeypatch):
    run, proto, opts = fixture(tmp_path)
    report = lab.audit(run_dir=run, protocol_folder=proto, **opts)
    monkeypatch.setattr(lab, "audit", lambda **k: report)
    output = tmp_path / "forensic_output"
    argv = ["--run-dir", str(run), "--protocol-folder", str(proto),
            "--output-dir", str(output)]
    assert lab.main(argv) == 0
    assert (output/"regression_forensics.json").is_file()
    with (output/"review_events.csv").open(newline="") as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert len(rows) == report["review"]["exported_candidate_events"]
    with pytest.raises(FileExistsError):
        lab.main(argv)
