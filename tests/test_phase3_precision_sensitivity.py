"""Pure offline confidence counterfactuals: never rerun nor mutate a tracker."""
from __future__ import annotations

from dataclasses import replace

import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth
from streetlab_phase3.video.precision_sensitivity import (
    THRESHOLDS, confidence_sensitivity, unmatched_proximity_audit,
)


def t(frame: int, tid: str, x: float, cls: str) -> FluidPixelTruth:
    return FluidPixelTruth(frame, tid, x, 20., cls)


def p(frame: int, tid: str, x: float, cls: str, confidence: float) -> GeoTraxPixelPoint:
    return GeoTraxPixelPoint(frame, tid, x, 20., x, 20., cls, confidence)


def corpus():
    truth = [
        t(1, "motor", 10, "MOTORCYCLE"),
        t(2, "motor", 12, "MOTORCYCLE"),
        t(1, "car", 100, "CAR"),
        t(2, "car", 100, "CAR"),
    ]
    prediction = [
        p(0, "m", 10, "MOTORCYCLE", .60),
        p(1, "m", 12, "MOTORCYCLE", .36),
        p(1, "ghost_m", 15, "MOTORCYCLE", .22),
        p(0, "c", 100, "CAR", .92),
        p(1, "c", 100, "CAR", .81),
        p(1, "ghost_c_near", 110, "CAR", .70),
        p(1, "ghost_c_far", 900, "CAR", .52),
        p(100, "outside", 1500, "CAR", 1.0),
    ]
    return truth, prediction


def test_predeclared_thresholds_report_recall_loss_after_post_track_cut():
    truth, sliced = corpus()
    source = list(sliced)
    report = confidence_sensitivity(
        sliced, truth, start_frame=0, end_frame=1,
        thresholds=(.15, .30, .50, .95))
    assert report["eligible_for_promotion"] is False
    assert report["thresholds_predeclared"] == [.15, .30, .50, .95]
    assert len(report["results"]) == 8
    global_rows = report["results"][:4]
    assert [x["predictions_kept"] for x in global_rows] == [7, 6, 5, 0]
    assert [x["matched_points"] for x in global_rows] == [4, 4, 3, 0]
    assert [x["unmatched_predictions"] for x in global_rows] == [3, 2, 2, 0]
    assert [x["motorcycle_correct_class_recall"] for x in global_rows] == [1, 1, .5, 0]
    assert global_rows[0]["truth_points"] == 4
    assert global_rows[3]["overall_point_precision"] is None
    assert report["results"][4]["policy"] == "motorcycle_only_post_track_cutoff"
    assert report["results"][6]["predictions_kept"] == 5
    assert list(sliced) == source  # Immutable original evidence


def test_unmatched_proximity_separates_close_centers_from_far_errors():
    truth, pred = corpus()
    report = unmatched_proximity_audit(pred, truth, start_frame=0, end_frame=1)
    assert report["unmatched_total"] == 3
    assert report["by_class"]["MOTORCYCLE"]["unmatched"] == 1
    assert report["by_class"]["MOTORCYCLE"]["near_matched_prediction_count"] == 1
    assert report["by_class"]["CAR"]["unmatched"] == 2
    assert report["by_class"]["CAR"]["near_matched_prediction_count"] == 1
    assert report["by_class"]["CAR"]["high_confidence_at_least_0_5"] == 2
    assert report["by_class"]["CAR"]["high_confidence_near_matched_prediction"] == 1
    assert report["by_class"]["CAR"]["near_any_truth_within_50px"] == 1
    far = next(x for x in report["examples"] if x["predicted_track_id"] == "ghost_c_far")
    assert far["near_matched_prediction"] is False
    assert far["near_truth_point"] is False
    assert not any(x["predicted_track_id"] == "outside" for x in report["examples"])


def test_missing_or_invalid_confidences_are_fail_closed():
    truth, pred = corpus()
    for bad in (None, float("nan"), float("inf"), 1.5, -.2):
        altered = [replace(pred[0], confidence=bad), *pred[1:]]
        with pytest.raises(ValueError, match="confidence"):
            confidence_sensitivity(altered, truth,
                                   start_frame=0, end_frame=1)
        with pytest.raises(ValueError, match="confidence"):
            unmatched_proximity_audit(altered, truth,
                                      start_frame=0, end_frame=1)


def test_rejects_threshold_search_order_and_missing_motorcycle_truth():
    truth, pred = corpus()
    for thresholds in ((), (.5, .25), (.25, .25), (.0,), (1.2,)):
        with pytest.raises(ValueError, match="Threshold"):
            confidence_sensitivity(pred, truth, start_frame=0, end_frame=1,
                                   thresholds=thresholds)
    with pytest.raises(ValueError, match="Motorcycle truth"):
        confidence_sensitivity(pred, [truth[2], truth[3]],
                               start_frame=0, end_frame=1)
    with pytest.raises(ValueError, match="Proximity"):
        unmatched_proximity_audit(pred, truth,
                                  start_frame=0, end_frame=1, proximity_px=-2.)


def test_default_thresholds_are_fixed_and_nonempty():
    assert THRESHOLDS == (.15, .25, .35, .45, .50, .60, .75)
