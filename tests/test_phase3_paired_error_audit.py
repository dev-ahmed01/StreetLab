"""Paired vehicle error forensics; matches frozen spatial benchmark semantics."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth
from streetlab_phase3.video.paired_error_audit import paired_error_audit


def gt(frame, tid, x, cls):
    return FluidPixelTruth(frame=frame, track_id=tid,
                           x_px=float(x), y_px=20., vehicle_class=cls)


def det(frame, tid, x, cls, confidence=.9):
    return GeoTraxPixelPoint(
        frame=frame, track_id=tid, x_px=float(x), y_px=20.,
        x_stabilized_px=float(x), y_stabilized_px=20.,
        vehicle_class=cls, confidence=confidence)


def test_rescued_motorcycle_and_fp_confidence_are_paired_by_observation():
    labels = [
        gt(1, "m1", 10, "MOTORCYCLE"),
        gt(1, "car1", 100, "CAR"),
        gt(2, "m1", 12, "MOTORCYCLE"),
        gt(2, "car1", 100, "CAR"),
    ]
    standard = [
        det(0, "mA", 10, "MOTORCYCLE", .8),
        det(0, "cA", 100, "CAR", .8),
        det(1, "cA", 100, "CAR", .9),
    ]
    sliced = [
        det(0, "mB", 10, "MOTORCYCLE", .9),
        det(0, "cB", 100, "CAR", .9),
        det(1, "mB", 12, "MOTORCYCLE", .9),
        det(1, "cB", 100, "CAR", .9),
        det(1, "false_m", 999, "MOTORCYCLE", .18),
        det(1, "false_c", 900, "CAR", .58),
    ]
    audit = paired_error_audit(standard, sliced, labels,
                               start_frame=0, end_frame=1)
    m = audit["by_truth_class"]["MOTORCYCLE"]
    assert audit["eligible_for_promotion"] is False
    assert m["truth_points"] == 2
    assert m["spatial"] == {"both": 1, "standard_only": 0,
                            "sliced_only": 1, "neither": 0}
    assert m["correct_class"] == m["spatial"]
    assert m["spatially_rescued_by_sliced"] == [
        {"video_frame": 1, "fluid_track_id": "m1"}]
    assert audit["sliced_unmatched"]["MOTORCYCLE"]["unmatched_point_count"] == 1
    assert audit["sliced_unmatched"]["MOTORCYCLE"]["unmatched_confidence_bands"]["[0,0.25)"] == 1
    assert audit["sliced_unmatched"]["MOTORCYCLE"]["unmatched_points_from_one_frame_ids"] == 1
    assert audit["sliced_unmatched"]["CAR"]["unmatched_confidence_bands"]["[0.50,1.00]"] == 1
    assert audit["standard_unmatched"]["MOTORCYCLE"]["unmatched_point_count"] == 0
    assert audit["sliced_unmatched"]["MOTORCYCLE"]["matched_mean_confidence"] == pytest.approx(.9)


def test_spatial_rescue_without_correct_class_credit():
    labels = [gt(1, "m", 10, "MOTORCYCLE")]
    audit = paired_error_audit([], [det(0, "t", 10, "CAR")], labels,
                               start_frame=0, end_frame=0)
    m = audit["by_truth_class"]["MOTORCYCLE"]
    assert m["spatial"]["sliced_only"] == 1
    assert m["correct_class"]["neither"] == 1
    assert m["wrong_class_spatial_matches"] == [
        {"video_frame": 0, "fluid_track_id": "m",
         "engine": "sliced", "predicted_class": "CAR"}]


def test_duplicate_ground_truth_identity_fails_closed():
    labels = [gt(1, "m", 10, "MOTORCYCLE"),
              gt(1, "m", 12, "MOTORCYCLE")]
    with pytest.raises(ValueError, match="Duplicate"):
        paired_error_audit([], [], labels, start_frame=0, end_frame=0)


def test_out_of_window_predictions_cannot_inflate_fp_or_matches():
    labels = [gt(1, "m", 10, "MOTORCYCLE")]
    standard = [det(100, "spur", 999, "MOTORCYCLE", .2)]
    report = paired_error_audit(standard, [], labels,
                                start_frame=0, end_frame=0)
    assert report["standard_unmatched"]["MOTORCYCLE"]["unmatched_point_count"] == 0
    assert report["by_truth_class"]["MOTORCYCLE"]["spatial"]["neither"] == 1


def test_confidence_outside_range_rejected():
    with pytest.raises(ValueError, match="Invalid prediction confidence"):
        paired_error_audit([det(0, "bad", 50, "CAR", 1.2)], [], [],
                           start_frame=0, end_frame=0)
