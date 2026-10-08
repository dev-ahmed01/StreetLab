from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from streetlab_phase3.video.engine_shootout import (
    Trial, load_class_map, model_class_ids, points_from_boxes,
    promotion_gate, run_trial,
)


class Tensor:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


def fake_boxes():
    return SimpleNamespace(
        xywh=Tensor([[100.0, 200.0, 20.0, 10.0],
                     [300.0, 400.0, 30.0, 12.0]]),
        id=Tensor([41.0, 99.0]),
        cls=Tensor([0.0, 2.0]),  # 2 is bicycle; MUST NOT become motorcycle
        conf=Tensor([0.92, 0.87]),
    )


def test_class_taxonomy_is_explicit_and_bicycle_is_not_a_motorcycle():
    ids = model_class_ids(
        {0: "car", 1: "motorcycle", 2: "bicycle", 3: "truck",
         4: "bus"}, load_class_map(None))
    assert ids == {0: 0, 1: 3, 3: 2, 4: 1}
    rows = points_from_boxes(fake_boxes(), 37, ids)
    assert len(rows) == 1
    assert rows[0][0] == 37
    assert rows[0][1] == 41
    assert rows[0][2:6] == [100.0, 200.0, 20.0, 10.0]
    assert rows[0][10] == 0
    assert len(rows[0]) == 14


def test_untracked_detections_do_not_get_fabricated_ids():
    assert points_from_boxes(SimpleNamespace(id=None), 5, {0: 0}) == []


def test_custom_model_class_mapping_is_validated(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"custom moped": 3, "car": 0}))
    assert load_class_map(str(path)) == {"custom moped": 3, "car": 0}
    path.write_text(json.dumps({"bicycle": 6}))
    with pytest.raises(ValueError, match="Invalid class mapping"):
        load_class_map(str(path))
    path.write_text(json.dumps({"car": True}))
    with pytest.raises(ValueError, match="Invalid class mapping"):
        load_class_map(str(path))


def test_unknown_detector_taxonomy_fails_closed():
    with pytest.raises(ValueError, match="No model classes map"):
        model_class_ids({0: "bicycle", 1: "person"}, load_class_map(None))


class FakeCapture:
    def __init__(self, path):
        self.current = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.current = int(value)
        return True

    def read(self):
        if self.current >= 10:
            return False, None
        value = self.current
        self.current += 1
        return True, value

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_POS_FRAMES = 1
    VideoCapture = FakeCapture


class FakeYOLO:
    instances = []

    def __init__(self, weights):
        self.names = {0: "car", 1: "motorcycle", 2: "bicycle"}
        self.calls = []
        FakeYOLO.instances.append(self)

    def track(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return [SimpleNamespace(boxes=fake_boxes())]


def test_window_runner_warms_tracker_exports_absolute_frame_ids_and_roundtrips(tmp_path):
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "candidate.txt"
    FakeYOLO.instances.clear()
    trial = Trial(str(video), "fake-model.pt", "bytetrack.yaml", str(output),
                  start_frame=3, end_frame=5, warmup_frames=2)
    result = run_trial(trial, model_factory=FakeYOLO, cv2_module=FakeCV2)
    assert result["first_processed_frame"] == 1
    assert result["processed_frames"] == 5
    assert result["exported_points"] == 3
    assert result["status"] == "EXPERIMENTAL_NOT_PROMOTED"
    with output.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert [int(row[0]) for row in rows] == [3, 4, 5]
    assert all(len(row) == 14 for row in rows)
    assert all(row[10] == "0" for row in rows)
    assert FakeYOLO.instances[0].calls[0][0] == 1
    assert all(kw["persist"] is True for _, kw in FakeYOLO.instances[0].calls)
    loaded = load_geotrax_pixel_tracks(output)
    assert all(p.track_id == "41" and p.vehicle_class == "CAR" for p in loaded)
    assert output.with_suffix(".manifest.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_trial(trial, model_factory=FakeYOLO, cv2_module=FakeCV2)


def test_failure_keeps_no_partial_track_files(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "missing.txt"
    trial = Trial(str(video), "fake-model.pt", "bytetrack.yaml", str(out),
                  start_frame=8, end_frame=12, warmup_frames=1)
    with pytest.raises(RuntimeError, match="Unexpected EOF"):
        run_trial(trial, model_factory=FakeYOLO, cv2_module=FakeCV2)
    assert not out.exists()
    assert not out.with_suffix(".manifest.json").exists()
    assert list(tmp_path.glob(".trial-*")) == []


def baseline_pixel():
    return {"truth_points": 100, "truth_tracks": 10, "evaluation_frame_start": 3,
            "evaluation_frame_end": 100, "max_distance_px": 50.0,
            "frame_offset": 1, "point_recall": .60, "point_precision": .95,
            "track_coverage": .80, "class_agreement": .93, "pixel_mae": 2.0}


def test_promotion_requires_matching_cohort_gains_and_identity():
    base = baseline_pixel()
    improved = dict(base, point_recall=.63, point_precision=.945)
    identity0 = {"truth_tracks": 10, "max_distance_px": 50.0,
                 "frame_offset": 1, "fraction_truth_tracks_fragmented": .25}
    identity1 = dict(identity0, fraction_truth_tracks_fragmented=.23)
    yes = promotion_gate(base, improved, identity0, identity1)
    assert yes["eligible_for_promotion"] is True
    bad_cohort = promotion_gate(base, dict(improved, truth_points=99), identity0, identity1)
    assert bad_cohort["eligible_for_promotion"] is False
    assert any("cohort" in x for x in bad_cohort["reasons"])
    bad_precision = promotion_gate(base, dict(improved, point_precision=.90), identity0, identity1)
    assert bad_precision["eligible_for_promotion"] is False
    assert not promotion_gate(base, improved)["eligible_for_promotion"]
    assert not promotion_gate(base, dict(base), identity0, identity0)["eligible_for_promotion"]
