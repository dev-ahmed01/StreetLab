"""Small, fully synthetic regression cases for the W04 cached-class diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase3_offline_class_flip_attribution.py"
SPEC = importlib.util.spec_from_file_location("phase3_class_flip_attribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_box_overlap_and_dual_raw_class_support():
    box = (0.0, 0.0, 10.0, 10.0)
    assert mod.overlap_iou(box, box) == 1
    assert mod.overlap_iou(box, (20.0, 20.0, 30.0, 30.0)) == 0
    observations = {2: [
        dict(x1=0, y1=0, x2=10, y2=10, vehicle_class="CAR", confidence=.8),
        dict(x1=0, y1=0, x2=10, y2=10, vehicle_class="BUS", confidence=.6),
    ]}
    assert set(mod.raw_box_support(observations, 2, box, .5)) == {"CAR", "BUS"}


def test_class_transition_preserves_both_hypotheses_and_previous_frame():
    box = (0.0, 0.0, 10.0, 10.0)
    tracks = {1: [(1, "CAR", box, .9), (2, "BUS", box, .6)]}
    observations = {
        1: [dict(x1=0, y1=0, x2=10, y2=10, vehicle_class="CAR", confidence=.8)],
        2: [
            dict(x1=0, y1=0, x2=10, y2=10, vehicle_class="BUS", confidence=.6),
            dict(x1=0, y1=0, x2=10, y2=10, vehicle_class="CAR", confidence=.5),
        ],
    }
    events, stats = mod.flip_events("test_policy", tracks, observations, .5)
    assert len(events) == 1
    assert events[0]["classification"] == "both_classes_in_current_raw"
    assert events[0]["prior_class_on_prior_frame"] is True
    assert stats["tracker_ids_with_class_change"] == 1
    assert stats["contiguous_class_change_events"] == 1


def test_duplicate_tracker_id_same_frame_rejected():
    row = "10750,1,1,1,2,2,1,1,2,2,0,0.9,2,2\n"
    with pytest.raises(ValueError, match="duplicate"):
        mod.read_track_rows((row + row).encode(), "test_policy")


def test_reject_invalid_original_evidence_status(tmp_path):
    (tmp_path / "batch_report.json").write_text('{"status":"FAKE"}', encoding="utf-8")
    evidence = mod.Evidence(archive=None, batch_dir=tmp_path)
    with pytest.raises(ValueError, match="Unexpected W04 evidence cohort"):
        mod.analyze(evidence)
