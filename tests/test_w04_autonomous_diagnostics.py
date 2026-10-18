"""Synthetic-only autonomous May25 full-cohort diagnostics regression tests.

Tests exercise the exact original PixelBenchmark matcher, without pretending
the synthetic points are the user's real 505MB video or new field evidence.
"""
from __future__ import annotations

import math
import pytest

from streetlab_phase3.geotrax_pixel import GeoTraxPixelPoint
from streetlab_phase3.pixel_benchmark import FluidPixelTruth
from streetlab_integration.w04_autonomous_diagnostics import (
    STATUS, AuditError, analyze_points, findings_markdown,
)


def pred(frame, ident, x, cls, *, y=100.):
    return GeoTraxPixelPoint(
        frame=frame, track_id=ident, x_px=x, y_px=y,
        x_stabilized_px=x, y_stabilized_px=y,
        vehicle_class=cls, confidence=.81,
    )


def gt(frame, ident, x, cls, *, y=100.):
    return FluidPixelTruth(
        frame=frame, track_id=ident, x_px=x, y_px=y, vehicle_class=cls)


def cohort():
    observations, originals = [], []
    for frame in range(60, 68):
        originals.extend([
            gt(frame+1, "7", 100.+frame-60, "HEAVY_VEHICLE"),
            gt(frame+1, "36", 200., "HEAVY_VEHICLE"),
            gt(frame+1, "c", 400., "CAR"),
        ])
        observations.extend([
            pred(frame, "26", 100.+frame-60, "CAR"),
            pred(frame, "29", 200., "CAR"),
            pred(frame, "car", 400., "CAR"),
        ])
    observations.extend([pred(frame, "bus_alpha", 500., "BUS")
                         for frame in range(60, 68)])
    return observations, originals


def test_repeated_frame_disagreements_group_into_original_truth_tracks():
    predictions, truth = cohort()
    result = analyze_points(predictions, truth, width=640, height=360)
    assert result["status"] == STATUS
    assert result["HEAVY_truth_unique_tracks"] == 2
    assert result["HEAVY_truth_tracks_with_class_disagreement"] == 2
    assert result["HEAVY_wrong_class_matched_points"] == 16
    assert result["HEAVY_error_share_top_two_FLUID_tracks"] == 1.0
    assert result["HEAVY_error_share_top_five_FLUID_tracks"] == 1.0
    assert result["unique_FLUID_truth_tracks"] == 3
    assert result["matched_points"] == 24
    assert result["truth_points"] == 24
    assert result["BUS_FLUID_truth_points"] == 0
    assert result["BUS_unique_predicted_track_ids"] == 1
    assert result["BUS_predicted_track_diagnostics"][0]["BUS_unmatched_FLUID_frames"] == 8
    assert all(item["dominant_predicted_class_fraction"] == 1.0
               for item in result["truth_track_diagnostics"])
    assert result["physical_object_labels_independently_adjudicated"] is False
    assert result["independent_holdout_pass"] is False
    assert result["production_release_approved"] is False


def test_full_cohort_reproduces_hungarian_class_confusion_not_class_aware_rematch():
    p = [pred(60,"P1",100.,"CAR"), pred(60,"P2",110.,"HEAVY_VEHICLE")]
    t = [gt(61,"T1",100.,"HEAVY_VEHICLE"),gt(61,"T2",110.,"CAR")]
    result=analyze_points(p,t,width=640,height=360)
    assert result["matched_points"] == 2
    assert result["HEAVY_wrong_class_matched_points"] == 1
    assert result["HEAVY_wrong_matches_with_close_competing_association"] == 1
    assert result["total_matched_points_with_close_competing_association"] == 2
    assert result["recomputed_class_diagnostics"]["confusion_truth_to_predicted"] == [
        {"truth":"CAR","predicted":"HEAVY_VEHICLE","points":1},
        {"truth":"HEAVY_VEHICLE","predicted":"CAR","points":1},
    ]


def test_track_id_fragmentation_is_exposed_without_repairing_history():
    predictions,truth=cohort()
    for i,frame in enumerate(range(64,68)):
        original=next(p for p in predictions if p.frame==frame and p.track_id=="26")
        predictions.remove(original)
        predictions.append(pred(frame,"26_second_ID", original.x_px, "CAR"))
    result=analyze_points(predictions,truth,width=640,height=360)
    row=next(x for x in result["truth_track_diagnostics"] if x["truth_track_id"]=="7")
    assert row["matched_predicted_track_ids"] == {"26":4,"26_second_ID":4}
    assert row["wrong_class_matches"] == 8
    assert result["HEAVY_truth_tracks_with_class_disagreement"] == 2


def test_persistent_bus_track_cannot_be_treated_as_physically_false_bus():
    pred_only=[pred(60,"BUS_P",200.,"BUS"),pred(61,"BUS_P",201.,"BUS")]
    truth=[gt(61,"CAR",200.,"CAR")]
    result=analyze_points(pred_only,truth,width=640,height=360)
    assert result["BUS_unique_predicted_track_ids"]==1
    assert result["BUS_predicted_track_diagnostics"][0]["BUS_matched_any_FLUID_class"]==1
    assert result["BUS_predicted_track_diagnostics"][0]["BUS_unmatched_FLUID_frames"]==1
    assert result["recomputed_class_diagnostics"]["by_class"]["BUS"]["truth_points"]==0
    assert result["recomputed_class_diagnostics"]["by_class"]["BUS"]["correct_class_recall"] is None


@pytest.mark.parametrize("kwargs",[
    {"start":61}, {"end":180}, {"offset":0}, {"gate":30.},
    {"width":0}, {"height":0},
])
def test_no_cohort_or_distance_gate_retuning_allowed(kwargs):
    predictions,truth=cohort()
    args={"width":640,"height":360}
    args.update(kwargs)
    with pytest.raises(AuditError):
        analyze_points(predictions,truth,**args)


def test_out_of_frame_or_missing_original_fails_closed():
    predictions,truth=cohort()
    predictions[0]=pred(60,"26",1000.,"CAR")
    with pytest.raises(AuditError, match="outside"):
        analyze_points(predictions,truth,width=640,height=360)
    with pytest.raises(AuditError,match="Missing"):
        analyze_points([],truth,width=640,height=360)


def test_generated_report_explicitly_blocks_production_and_physical_claims():
    predictions,truth=cohort()
    result=analyze_points(predictions,truth,width=640,height=360)
    result["original_saved_FLUID_label_F1"]=0.9142034068
    summary=findings_markdown(result)
    assert "16" in summary
    assert "track consistency" in summary
    assert "not a blind holdout" in summary.lower()
    assert "No manual review is required" in summary
    assert "No independent" not in summary or "not passed" in summary.lower()
