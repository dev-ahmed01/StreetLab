"""Synthetic controls for W04 hybrid merger; no physical performance claim."""
from __future__ import annotations

import pytest
from streetlab_phase3.video.integrated_box_lab import RawBox, merge_boxes
from streetlab_phase3.video.w04_hybrid_rare_suppression import (
    merge_rare_preserving, MODES,
)


def box(frame, x0, y0, x1, y1, label, tile, idx, conf=.85):
    return RawBox(frame, x0, y0, x1, y1, conf, label, tile, idx)


def test_hybrid_retains_overlapping_distinct_motorcycle_hypotheses():
    # Box overlap: IoS 0.5 but IoU 1/3; old aggressive IoS suppresses one.
    raw = [
        box(10, 0, 0, 10, 10, "MOTORCYCLE", 0, 0, .9),
        box(10, 5, 0, 15, 10, "MOTORCYCLE", 1, 1, .8),
    ]
    original = merge_boxes(raw, policy="hard_nms", threshold=.3,
                           metric="ios", score_floor=.15)
    hybrid = merge_rare_preserving(raw, mode="rare_iou05")
    assert len(original) == 1
    assert len(hybrid) == 2
    assert [p.vehicle_class for p in hybrid] == ["MOTORCYCLE"]*2


def test_cars_remain_original_ios_nms_and_no_cross_class_label_mutation():
    raw = [
        box(2, 0, 0, 10, 10, "CAR", 0, 0, .9),
        box(2, 5, 0, 15, 10, "CAR", 1, 1, .7),
        box(2, 5, 0, 15, 10, "MOTORCYCLE", 1, 2, .7),
    ]
    expected = merge_boxes(raw, policy="hard_nms", threshold=.3,
                           metric="ios", score_floor=.15)
    result = merge_rare_preserving(raw)
    assert sum(p.vehicle_class=="CAR" for p in result)==1
    assert sum(p.vehicle_class=="MOTORCYCLE" for p in result)==1
    assert len(result)==len(expected)
    assert all(p.frame==2 for p in result)


def test_center_guard_distinguishes_adjacent_and_near_identical_boxes():
    adjacent = [
        box(2, 0, 0, 10, 10, "MOTORCYCLE", 0, 0, .9),
        box(2, 5, 0, 15, 10, "MOTORCYCLE", 1, 1, .8),
    ]
    identical = [
        box(2, 0, 0, 10, 10, "MOTORCYCLE", 0, 0, .9),
        box(2, 0, 0, 10, 10, "MOTORCYCLE", 1, 1, .8),
    ]
    assert len(merge_rare_preserving(adjacent,mode="rare_center_gate"))==2
    assert len(merge_rare_preserving(identical,mode="rare_center_gate"))==1


def test_cross_tile_guard_does_not_delete_same_tile_candidates():
    same = [
        box(3, 0, 0, 10, 10, "HEAVY_VEHICLE", 1, 0, .9),
        box(3, 0, 0, 10, 10, "HEAVY_VEHICLE", 1, 1, .7),
    ]
    assert len(merge_rare_preserving(same,mode="rare_center_gate"))==1
    assert len(merge_rare_preserving(same,mode="rare_cross_tile_center_gate"))==2


def test_frozen_frames_and_classes_and_fail_closed_modes():
    boxes=[
        box(10,0,0,10,10,"BUS",0,0),
        box(11,0,0,10,10,"MOTORCYCLE",0,0),
    ]
    for mode in MODES:
        returned=merge_rare_preserving(boxes,mode=mode)
        assert {(p.frame,p.vehicle_class) for p in returned}=={
            (10,"BUS"),(11,"MOTORCYCLE")
        }
    with pytest.raises(ValueError):
        merge_rare_preserving(boxes,mode="invented")
