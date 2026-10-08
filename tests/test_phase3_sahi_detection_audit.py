from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.identity_benchmark import IdentityBenchmark
from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark
from streetlab_phase3.video.sahi_detection_audit import (
    Detection, DetectorAudit, _accepted_matches, convert_sahi_predictions,
    detector_gate, run_detector_audit, score_detections,
)


def truth(frame, id, x, category):
    return FluidPixelTruth(frame, id, x, 20.0, category)


def pred(frame, id, x, category="MOTORCYCLE"):
    return GeoTraxPixelPoint(frame, id, x, 20, x, 20, category, .95)


def test_fixed_window_counts_missing_first_last_and_entirely_empty_output():
    records = [truth(f, "T", 100, "MOTORCYCLE") for f in (1, 2, 3, 4)]
    metric = PixelBenchmark(max_distance_px=50.0)
    report = metric.evaluate([pred(1, "P", 100)], records, frame_offsets=(1,),
                             evaluation_frame_range=(1, 4))
    assert report.matched_points == 1
    assert report.truth_points == 4
    assert report.evaluation_frame_start == 1
    assert report.evaluation_frame_end == 4
    assert report.point_recall == .25
    empty = metric.evaluate([], records, frame_offsets=(1,),
                            evaluation_frame_range=(1, 4))
    assert empty.truth_points == 4
    assert empty.point_recall == 0
    assert empty.track_coverage == 0
    assert empty.predicted_points == 0
    identity = IdentityBenchmark().evaluate([pred(1, "P", 100)], records,
        frame_offsets=(1,), evaluation_frame_range=(1, 4))
    assert identity.scorecard.mean_truth_track_matched_fraction == .25


def test_fixed_window_rejects_invalid_intervals():
    with pytest.raises(ValueError, match="evaluation_frame_range"):
        PixelBenchmark().evaluate([], [], frame_offsets=(1,), evaluation_frame_range=(3, 1))


def sahi_object(name, category_id, x, confidence=.85):
    return SimpleNamespace(
        category=SimpleNamespace(name=name, id=category_id),
        bbox=SimpleNamespace(to_xyxy=lambda: (x - 5, 15, x + 5, 25)),
        score=SimpleNamespace(value=confidence))


def test_sahi_class_ontology_and_box_centers():
    from streetlab_phase3.video.engine_shootout import load_class_map
    result = convert_sahi_predictions([
        sahi_object("motorcycle", 3, 50),
        sahi_object("bicycle", 1, 80),
        sahi_object("car", 2, 30)], 10, load_class_map(None))
    assert [(p.vehicle_class, p.x_px, p.frame) for p in result] == [
        ("MOTORCYCLE", 50, 10), ("CAR", 30, 10)]


def test_sahi_custom_numeric_mapping_is_explicit_not_default():
    from streetlab_phase3.video.engine_shootout import load_class_map
    assert convert_sahi_predictions([sahi_object("person", 0, 50)], 0,
                                    load_class_map(None)) == []
    assert convert_sahi_predictions([sahi_object("class0", 0, 50)], 0,
                                    {"0": 0})[0].vehicle_class == "CAR"


def test_sahi_fails_on_malformed_predictions():
    invalid = sahi_object("motorcycle", 3, 50, float("nan"))
    with pytest.raises(ValueError, match="Invalid SAHI"):
        convert_sahi_predictions([invalid], 0, {"motorcycle": 3})


def test_matching_is_class_aware_and_max_cardinality():
    # Car at same center must NOT rescue absent motorcycle.
    matched = _accepted_matches([Detection(1, 100, 20, "CAR", .9)],
                                [truth(2, "t", 100, "MOTORCYCLE")], 50)
    # Caller separates classes; individual matcher handles one class only.
    assert matched == [0.0]
    points = [Detection(0, 100, 20, "CAR", .9)]
    labels = [truth(1, "m", 100, "MOTORCYCLE")]
    report = score_detections({0: points}, labels, frames=[0])
    assert report["matched_points"] == 0
    assert report["recall"] == 0
    assert report["precision"] == 0


def test_empty_detection_frames_cannot_improve_recall_artificially():
    labels = [truth(i + 1, str(i), 100, "MOTORCYCLE") for i in range(5)]
    report = score_detections(
        {2: [Detection(2, 100, 20, "MOTORCYCLE", .9)]},
        labels, frames=[0, 1, 2, 3, 4])
    assert report["sampled_frames"] == 5
    assert report["truth_points"] == 5
    assert report["matched_points"] == 1
    assert report["recall"] == pytest.approx(.2)
    assert report["per_class"]["MOTORCYCLE"]["misses"] == 4


def test_candidate_gate_rejects_absent_positives_and_precision_regression():
    base = score_detections({0: [Detection(0, 100, 20, "MOTORCYCLE", .8)]},
                            [truth(1, "m", 100, "MOTORCYCLE")], [0])
    sliced = score_detections({0: [Detection(0, 100, 20, "MOTORCYCLE", .8)]},
                              [truth(1, "m", 100, "MOTORCYCLE")], [0])
    base["latency_median_s"] = .1
    sliced["latency_median_s"] = .4
    assert not detector_gate(base, sliced)["eligible_for_tracking_trial"]
    assert not detector_gate(base, sliced)["eligible_for_production"]


class FakeCapture:
    def __init__(self, path):
        self.frame = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, frame):
        self.frame = int(frame)
        return True

    def get(self, prop):
        return self.frame

    def read(self):
        self.frame += 1
        return True, np.zeros((6, 6, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_POS_FRAMES = 1
    COLOR_BGR2RGB = 4
    VideoCapture = FakeCapture
    convert_calls = 0

    @staticmethod
    def cvtColor(image, code):
        assert code == FakeCV2.COLOR_BGR2RGB
        FakeCV2.convert_calls += 1
        return image


def fake_standard(image, model, **kwargs):
    return SimpleNamespace(object_prediction_list=[sahi_object("car", 0, 10)])


def fake_sliced(image, model, **kwargs):
    assert kwargs["perform_standard_pred"] is False
    assert kwargs["postprocess_type"] == "NMS"
    return SimpleNamespace(object_prediction_list=[
        sahi_object("car", 0, 10), sahi_object("motorcycle", 3, 100)])


class FakeTimer:
    n = 0
    def __call__(self):
        self.n += 1
        return self.n / 10


def test_sahi_pipeline_exports_detection_only_without_pretending_tracking(tmp_path):
    video = tmp_path / "video.mp4"
    weights = tmp_path / "model.pt"
    csv_file = tmp_path / "fluid.csv"
    video.write_bytes(b"test frames placeholder")
    weights.write_bytes(b"test checkpoint placeholder")
    csv_file.write_text("frame,id,cx,cy,type\n"
                        "1,1,10,20,car\n1,2,100,20,moped\n"
                        "2,1,10,20,car\n2,2,100,20,moped\n")
    out = tmp_path / "audit"
    FakeCV2.convert_calls = 0
    report = run_detector_audit(DetectorAudit(
        str(video), str(csv_file), str(weights), str(out), 0, 1, sample_step=1),
        model_loader=lambda *a: object(),
        predictors=(fake_standard, fake_sliced),
        cv2_module=FakeCV2, timer=FakeTimer())
    assert report["sample_frames"] == [0, 1]
    assert report["standard"]["per_class"]["MOTORCYCLE"]["recall"] == 0
    assert report["sliced"]["per_class"]["MOTORCYCLE"]["recall"] == 1
    assert report["standard"]["per_class"]["CAR"]["recall"] == 1
    assert report["gate"]["eligible_for_tracking_trial"]
    assert report["gate"]["eligible_for_production"] is False
    assert FakeCV2.convert_calls == 2
    assert (out / "report.json").exists()
    with (out / "sliced_detections.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "track_id" not in rows[0]
    assert json.loads((out / "report.json").read_text())["status"] == "EXPERIMENTAL_DETECTION_ONLY"
    with pytest.raises(FileExistsError):
        run_detector_audit(DetectorAudit(str(video), str(csv_file), str(weights),
                                         str(out), 0, 1), model_loader=lambda *a: object(),
                          predictors=(fake_standard, fake_sliced), cv2_module=FakeCV2)


def test_sahi_rejects_windows_with_no_annotated_motorcycles(tmp_path):
    video = tmp_path / "v.mp4"
    weights = tmp_path / "w.pt"
    labels = tmp_path / "gt.csv"
    video.write_bytes(b"a")
    weights.write_bytes(b"b")
    labels.write_text("frame,id,cx,cy,type\n1,1,10,20,car\n")
    with pytest.raises(ValueError, match="No annotated motorcycles"):
        run_detector_audit(DetectorAudit(str(video), str(labels), str(weights),
                           str(tmp_path / "out"), 0, 0),
                           model_loader=lambda *a: object(),
                           predictors=(fake_standard, fake_sliced),
                           cv2_module=FakeCV2)
    assert not (tmp_path / "out").exists()
