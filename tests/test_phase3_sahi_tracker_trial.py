from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
from streetlab_phase3.video.sahi_tracker_trial import (
    SahiTrackingTrial, rows_from_tracks, run_sahi_tracking, sahi_to_detections,
)


def object_prediction(name: str, cls_id: int, x: float, score: float = .8):
    return SimpleNamespace(
        category=SimpleNamespace(name=name, id=cls_id),
        bbox=SimpleNamespace(to_xyxy=lambda: [x - 10, 15, x + 10, 25]),
        score=SimpleNamespace(value=score))


class FakeDetections:
    def __init__(self, *, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = None


class FakeTracker:
    instances = []

    def __init__(self, **settings):
        self.settings = settings
        self.updates = []
        FakeTracker.instances.append(self)

    def update(self, detections):
        self.updates.append(detections)
        detections.tracker_id = np.array(
            [1 if klass == 3 else -1 for klass in detections.class_id], dtype=int)
        return detections


class FakeCapture:
    def __init__(self, path):
        self.current = 0
        self.released = False
        FakeCV.last_capture = self

    def isOpened(self):
        return True

    def set(self, prop, idx):
        self.current = int(idx)
        return True

    def read(self):
        if self.current == 7:
            return False, None
        now = self.current
        self.current += 1
        return True, now

    def get(self, prop):
        return self.current

    def release(self):
        self.released = True


class FakeCV:
    CAP_PROP_POS_FRAMES = 1
    COLOR_BGR2RGB = 2
    VideoCapture = FakeCapture
    last_capture = None

    @staticmethod
    def cvtColor(img, code):
        assert code == FakeCV.COLOR_BGR2RGB
        return img


def predict(img, model, **kwargs):
    # A false positive (bicycle) is never exported; a car without tracker ID
    # is ignored, and the motorcycle retains a real persistent tracker ID.
    return SimpleNamespace(object_prediction_list=[
        object_prediction("motorcycle", 3, float(img + 100)),
        object_prediction("car", 0, float(img + 200)),
        object_prediction("bicycle", 1, float(img + 300))])


def inputs(tmp_path, start=3, end=5, mode="sliced"):
    video, weights, outfile = [tmp_path / name for name in ("video.mp4", "model.pt", "run.txt")]
    video.write_bytes(b"video fixture")
    weights.write_bytes(b"model fixture")
    return SahiTrackingTrial(
        str(video), str(weights), str(outfile), start_frame=start, end_frame=end,
        warmup_frames=2, mode=mode), outfile


def run_mock(trial):
    return run_sahi_tracking(
        trial, model_loader=lambda *args: object(),
        predictors=(predict, predict),
        tracker_factory=FakeTracker,
        detection_class=FakeDetections,
        cv2_module=FakeCV,
    )


def test_sahi_detections_ignore_bicycles_and_have_finite_boxes():
    boxes = sahi_to_detections([
        object_prediction("motorcycle", 3, 50),
        object_prediction("bicycle", 1, 75),
        object_prediction("car", 2, 100)], {"car": 0, "motorcycle": 3},
        FakeDetections)
    assert boxes.xyxy.shape == (2, 4)
    assert list(boxes.class_id) == [3, 0]
    assert len(boxes.confidence) == 2
    assert sahi_to_detections([], {"car": 0}, FakeDetections).xyxy.shape == (0, 4)


def test_rows_exclude_unconfirmed_and_reject_duplicate_ids():
    d = FakeDetections(xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]]),
                       confidence=np.array([.8, .7]),
                       class_id=np.array([3, 0]))
    d.tracker_id = np.array([5, -1])
    rows, ignored = rows_from_tracks(d, 42)
    assert ignored == 1
    assert rows == [[42, 5, 20., 30., 20., 20., 20., 30., 20., 20.,
                     3, .8, 20., 20.]]
    d.tracker_id = np.array([5, 5])
    with pytest.raises(ValueError, match="duplicate"):
        rows_from_tracks(d, 42)


@pytest.mark.parametrize("mode", ["standard", "sliced"])
def test_pipeline_emits_absolute_frames_90_warmup_contract_and_real_ids(tmp_path, mode):
    FakeTracker.instances.clear()
    trial, output = inputs(tmp_path, mode=mode)
    report = run_mock(trial)
    assert report["status"] == "EXPERIMENTAL_NOT_PROMOTED"
    assert report["first_decoded_frame"] == 1
    assert report["processed_frames"] == 5
    assert report["evaluation_frames"] == 3
    assert report["evaluation_raw_detections"] == 6
    assert report["evaluation_confirmed_tracks"] == 3
    assert report["evaluation_unconfirmed_detections"] == 3
    assert FakeCV.last_capture.released
    assert len(FakeTracker.instances[-1].updates) == 5
    tracks = load_geotrax_pixel_tracks(output)
    assert [track.frame for track in tracks] == [3, 4, 5]
    assert all(track.track_id == "1" for track in tracks)
    assert all(track.vehicle_class == "MOTORCYCLE" for track in tracks)
    assert all(track.x_px == float(track.frame + 100) for track in tracks)
    assert output.with_suffix(".manifest.json").is_file()
    with pytest.raises(FileExistsError):
        run_mock(trial)


def test_decode_shift_causes_explicit_trial_failure(tmp_path):
    class MisalignedCapture(FakeCapture):
        def get(self, prop):
            return self.current + 1

    class MisalignedCV(FakeCV):
        VideoCapture = MisalignedCapture

    trial, output = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="decode/frame alignment"):
        run_sahi_tracking(
            trial, model_loader=lambda *args: object(),
            predictors=(predict, predict), tracker_factory=FakeTracker,
            detection_class=FakeDetections, cv2_module=MisalignedCV)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_interrupted_trial_leaves_no_truncated_artifact(tmp_path):
    trial, output = inputs(tmp_path, start=4, end=8)
    with pytest.raises(RuntimeError, match="Unexpected video EOF"):
        run_mock(trial)
    assert FakeCV.last_capture.released
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert list(tmp_path.glob(".sahi-trial-*")) == []


def test_bad_tracker_thresholds_fail_before_model_loading(tmp_path):
    trial, _ = inputs(tmp_path)
    from dataclasses import replace
    with pytest.raises(ValueError, match="thresholds"):
        run_mock(replace(trial, high_conf_det_threshold=.30,
                         track_activation_threshold=.20))
    with pytest.raises(ValueError, match="Detector floor"):
        run_mock(replace(trial, confidence=.19,
                         high_conf_det_threshold=.15))


def test_zero_confirmed_tracks_are_rejected(tmp_path):
    class AllUnconfirmed(FakeTracker):
        def update(self, d):
            d.tracker_id = np.full(len(d.xyxy), -1, dtype=int)
            return d
    trial, output = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="No confirmed vehicle tracks"):
        run_sahi_tracking(
            trial, model_loader=lambda *args: object(),
            predictors=(predict, predict), tracker_factory=AllUnconfirmed,
            detection_class=FakeDetections, cv2_module=FakeCV)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
