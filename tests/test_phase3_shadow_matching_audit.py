"""Regression: original pixel scores are frozen, shadow matcher diagnoses crowded failures."""
from __future__ import annotations

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth, PixelBenchmark
from streetlab_phase3.video.shadow_matching_audit import (
    maximum_cardinality_gate_matches, shadow_matching_audit
)


def p(frame, tid, x, y, label="CAR", confidence=.9):
    return GeoTraxPixelPoint(frame, tid, x, y, x, y, label, confidence)


def t(frame, tid, x, y, label="CAR"):
    return FluidPixelTruth(frame, tid, x, y, label)


def test_crowded_geometry_can_lose_a_valid_pair_under_post_gate_hungarian():
    # Truth at (0,0) and (40,0); second detection ~40px from first
    # truth, ~70px from second truth. Euclidean positions are valid.
    labels = [t(1, "1", 0., 0.), t(1, "2", 40., 0.)]
    detections = [
        p(0, "near", 0., 0.),
        p(0, "side", -21.25, 33.89, confidence=.8),
    ]
    frozen = PixelBenchmark(50.).match_points(detections, labels, frame_offset=1)
    shadow = maximum_cardinality_gate_matches(detections, labels)
    assert len(frozen) == 1
    assert len(shadow) == 2
    assert {m[1].track_id for m in shadow} == {"1", "2"}
    report = shadow_matching_audit(detections, detections, labels,
                                   start_frame=0, end_frame=0)
    assert report["eligible_for_promotion"] is False
    assert report["sliced"]["frozen_p3b_hungarian_matches"] == 1
    assert report["sliced"]["shadow_max_cardinality_matches"] == 2
    assert report["sliced"]["additional_in_gate_matches"] == 1
    assert report["sliced"]["previously_unmatched_high_confidence_car_predictions_now_assigned"] == 1
    assert len(report["sliced"]["newly_matched_truth_examples"]) == 1


def test_shadow_same_as_frozen_for_simple_distinct_targets():
    labels = [t(1, "m", 10., 0., "MOTORCYCLE"),
              t(1, "car", 100., 0.)]
    detections = [p(0, "m1", 10., 0., "MOTORCYCLE"),
                  p(0, "c1", 100., 0.),
                  p(0, "false", 900., 0.)]
    report = shadow_matching_audit(detections, detections, labels,
                                   start_frame=0, end_frame=0)
    for mode in ("standard", "sliced"):
        assert report[mode]["frozen_p3b_hungarian_matches"] == 2
        assert report[mode]["shadow_max_cardinality_matches"] == 2
        assert report[mode]["additional_in_gate_matches"] == 0
        assert report[mode]["shadow_unmatched_predictions"] == 1


def test_shadow_uses_same_fixed_window_and_does_not_match_outside():
    truth = [t(10, "m", 10., 0., "MOTORCYCLE")]
    detections = [p(0, "outside", 10., 0., "MOTORCYCLE")]
    report = shadow_matching_audit(detections, [], truth,
                                   start_frame=9, end_frame=9)
    assert report["standard"]["truth_points"] == 1
    assert report["standard"]["predicted_points"] == 0
    assert report["standard"]["shadow_max_cardinality_matches"] == 0
