"""May-25 regression safety tests use synthetic video metadata; no detector run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from streetlab_integration import w04_crossdate_regression as cross
from streetlab_integration.w04_holdout_intake import digest


class Capture:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FRAME_COUNT = 3
    CAP_PROP_FPS = 4

    def __init__(self, values=None):
        self.values = values or {1: 3840, 2: 2160, 3: 10000, 4: 30.0}
        self.released = False

    def VideoCapture(self, name):
        return self

    def isOpened(self):
        return True

    def get(self, field):
        return self.values[field]

    def release(self):
        self.released = True


def files(tmp_path):
    video = tmp_path / "20250525_video.mp4"
    truth = tmp_path / "20250525_video_Traj.csv"
    model = tmp_path / "checkpoint_openvino_model"
    model.mkdir()
    video.write_bytes(b"fake regression video")
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "61,1,100,200,car\n"
        "62,2,110,250,motorcycle\n"
        "180,3,400,230,truck\n"
        "181,4,150,330,car\n", encoding="utf-8",
    )
    lock = {
        "status": "PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT",
        "selection_scope": "DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION",
        "eligible_for_production": False,
        "selected_policy": "hard_nms_ios_0.30",
        "selected_settings": {
            "name": "hard_nms_ios_0.30", "policy": "hard_nms",
            "threshold": .3, "metric": "ios", "score_floor": .15,
            "sigma": .5, "class_agnostic": False,
        },
        "source_development_batch_sha256": "a"*64,
        "source_development_video_sha256": "b"*64,
        "source_fluid_sha256": "c"*64,
        "source_model_sha256": "d"*64,
        "predeclared_limits": {"max_cpu_seconds_per_frame": 2.0},
    }
    lp = tmp_path / "policy_lock.json"
    lp.write_text(json.dumps(lock))
    protocol = {
        "schema_version": 1, "status": "W04_DEVELOPMENT_POLICY_PREREGISTERED_NOT_HOLDOUT_VALIDATED",
        "research_scope": "FROZEN_W04_DEVELOPMENT_SELECTION_ONLY",
        "policy_lock_sha256": digest(lp),
        "selected_policy": "hard_nms_ios_0.30",
        "development_batch_sha256": "a"*64,
        "source_video_sha256": "b"*64,
        "source_fluid_sha256": "c"*64,
        "source_model_sha256": "d"*64,
        "predeclared_numeric_gates": {"max_cpu_seconds_per_frame": 2.0},
        "new_source_established_untouched": False,
        "same_window_t000_baseline_available": False,
        "independent_holdout_pass": False,
        "production_release_approved": False,
    }
    pp = tmp_path / "holdout_protocol.json"
    pp.write_text(json.dumps(protocol))
    return pp, lp, video, truth, model


def ready(paths, *, capture=None, hasher=None):
    pp, lp, video, truth, model = paths
    return cross.check_inputs(
        protocol_file=pp, lock_file=lp, video=video,
        truth=truth, model_dir=model,
        capture=capture if capture is not None else Capture(),
        hasher=hasher if hasher is not None else lambda d: "d"*64,
    )


def test_fixed_window_regression_preflight(tmp_path):
    paths = files(tmp_path)
    capture = Capture()
    report = ready(paths, capture=capture)
    assert capture.released is True
    assert report["fixed_source_evaluation_frames"] == [60, 179]
    assert report["fixed_fluid_evaluation_frames"] == [61, 180]
    assert report["tracker_warmup_frames"] == 30
    assert report["first_decoded_source_frame"] == 30
    assert report["supported_fluid_truth_points"] == 3
    assert report["supported_fluid_labeled_frames"] == 3
    assert report["selected_policy"] == "hard_nms_ios_0.30"
    assert report["independent_holdout_pass"] is False
    assert report["production_release_approved"] is False


def test_original_fluid_blank_coordinates_are_skipped_but_reported(tmp_path):
    paths = files(tmp_path)
    pp, lp, video, truth, model = paths
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "61,1,100,200,car\n"
        "61,2,,150,motorcycle\n"
        "62,3,99,,truck\n"
        "62,4,,,bus\n"
        "63,5,30,40,moped\n"
        "64,6,15,17,lorry\n"
        "65,7,22,50,pedestrian\n"
        "180,8,45,60,motorcycle\n",
        encoding="utf-8",
    )
    before = digest(truth)
    report = ready(paths)
    assert report["supported_fluid_rows_before_blank_coordinate_filter"] == 6
    assert report["skipped_fluid_rows_with_blank_coordinates"] == 3
    assert report["supported_fluid_truth_points"] == 3
    assert report["supported_fluid_labeled_frames"] == 3
    assert report["original_fluid_label_types"] == {
        "car": 1, "moped": 1, "motorcycle": 1,
    }
    assert digest(truth) == before
    assert report["independent_holdout_pass"] is False


def test_all_original_fluid_rows_without_centers_fail_without_fake_scores(tmp_path):
    paths = files(tmp_path)
    truth = paths[3]
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "61,1,,100,car\n"
        "62,2,200,,motorcycle\n"
        "63,3,,,truck\n",
    )
    before = digest(truth)
    with pytest.raises(cross.CrossDateError, match="3 skipped blank coordinate rows"):
        ready(paths)
    assert digest(truth) == before


@pytest.mark.parametrize("bad_x", ["not-a-number", "NaN", "Infinity", " "])
def test_malformed_nonempty_centers_still_fail_closed(tmp_path, bad_x):
    paths = files(tmp_path)
    paths[3].write_text(
        "frame,id,cx,cy,type\n"
        f"61,1,{bad_x},100,car\n"
        "62,2,5,30,motorcycle\n",
    )
    with pytest.raises(cross.CrossDateError, match="Invalid original FLUID"):
        ready(paths)


def test_frozen_openvino_model_hash_required(tmp_path):
    paths = files(tmp_path)
    with pytest.raises(cross.CrossDateError, match="OpenVINO model"):
        ready(paths, hasher=lambda d: "b"*64)


def test_missing_truth_window_rejected(tmp_path):
    paths = files(tmp_path)
    paths[3].write_text("frame,id,cx,cy,type\n500,1,2,3,car\n")
    with pytest.raises(cross.CrossDateError, match="No supported FLUID truth"):
        ready(paths)


@pytest.mark.parametrize("index,value", [(1, 1920), (2, 1080), (3, 50), (4, 25)])
def test_wrong_video_dimensions_or_frame_rate_rejected(tmp_path, index, value):
    paths = files(tmp_path)
    values = Capture().values.copy()
    values[index] = value
    with pytest.raises(cross.CrossDateError, match="dimensions/FPS"):
        ready(paths, capture=Capture(values))


def test_video_and_truth_must_be_exact_historical_pair_names(tmp_path):
    pp, lp, video, truth, model = files(tmp_path)
    new_video = tmp_path / "renamed_video.mp4"
    new_video.write_bytes(video.read_bytes())
    with pytest.raises(cross.CrossDateError, match="original May-25"):
        cross.check_inputs(
            protocol_file=pp, lock_file=lp, video=new_video, truth=truth,
            model_dir=model, capture=Capture(), hasher=lambda d: "d"*64,
        )


def test_reused_W04_sha_is_blocked(tmp_path):
    pp, lp, video, truth, model = files(tmp_path)
    changed = json.loads(lp.read_text())
    changed["source_development_video_sha256"] = digest(video)
    lp.write_text(json.dumps(changed))
    protocol = json.loads(pp.read_text())
    protocol["source_video_sha256"] = digest(video)
    protocol["policy_lock_sha256"] = digest(lp)
    pp.write_text(json.dumps(protocol))
    with pytest.raises(cross.CrossDateError, match="reused exact tuned W04"):
        ready((pp, lp, video, truth, model))


def test_mutated_lock_refused(tmp_path):
    pp, lp, video, truth, model = files(tmp_path)
    obj = json.loads(lp.read_text())
    obj["selected_settings"]["threshold"] = .70
    lp.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="original aligned"):
        ready((pp, lp, video, truth, model))


def test_cli_preflight_exclusive_receipt_and_no_real_inference(tmp_path, monkeypatch):
    pp, lp, video, truth, model = files(tmp_path)
    planned = ready((pp, lp, video, truth, model))
    monkeypatch.setattr(cross, "check_inputs", lambda **kwargs: planned)
    output = tmp_path / "new_run"
    args = ["--protocol", str(pp), "--policy-lock", str(lp),
            "--video", str(video), "--truth", str(truth),
            "--model-dir", str(model), "--output-dir", str(output)]
    assert cross.main(args) == 0
    receipt = json.loads((output/"preflight.json").read_text())
    assert receipt["independent_holdout_pass"] is False
    assert not (output/"batch").exists()
    with pytest.raises(FileExistsError):
        cross.main(args)


def test_cli_injected_result_never_changes_holdout_status(tmp_path, monkeypatch):
    pp, lp, video, truth, model = files(tmp_path)
    planned = ready((pp, lp, video, truth, model))
    monkeypatch.setattr(cross, "check_inputs", lambda **kwargs: planned)
    def stubbed_execution(**kwargs):
        assert kwargs["preflight"] == planned
        return {
            "status": cross.STATUS, "truth_label_precision": .8,
            "truth_label_recall": .9, "truth_label_f1": 2*.8*.9/1.7,
            "independent_holdout_pass": False, "production_release_approved": False,
        }
    monkeypatch.setattr(cross, "execute", stubbed_execution)
    out = tmp_path / "another_run"
    assert cross.main(["--protocol", str(pp), "--policy-lock", str(lp),
                       "--video", str(video), "--truth", str(truth),
                       "--model-dir", str(model), "--output-dir", str(out),
                       "--run-inference"]) == 0
    report = json.loads((out/"cross_date_summary.json").read_text())
    assert report["independent_holdout_pass"] is False
    assert report["production_release_approved"] is False
