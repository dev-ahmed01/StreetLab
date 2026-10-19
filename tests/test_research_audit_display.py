"""The research UI must never display unverified or promoted video evidence."""
import json
from copy import deepcopy
import pytest
from streetlab_integration.research_audit_display import read_audit, AuditDisplayError

VIDEO_SHA = "7ba62d051a0fdd1dd8cb0bb91d32a37fef1ee9f16c3aeadd70e1b820dfa88c82"
TRUTH_SHA = "ff845d13053065b01c2954a2cdf586fc070214cf896b208734a703692c74c643"

def fixture():
    return {
        "status":"W04_MAY25_AUTOMATED_LABEL_RELATIVE_OBJECT_AUDIT_COMPLETE",
        "source_video_frame_window":[60,179], "FLUID_frame_window":[61,180],
        "physical_object_labels_independently_adjudicated":False,
        "independent_holdout_pass":False, "production_release_approved":False,
        "no_new_detector_inference":True, "no_changes_to_model_or_tracking_policy":True,
        "no_FLUID_relabeling":True,
        "original_video_and_FLUID_SHA":{"original_video_sha256":VIDEO_SHA,"original_truth_sha256":TRUTH_SHA},
        "recomputed_class_diagnostics":{"truth_points":3,"matched_points":2,"predicted_points":2,
            "by_class":{"CAR":{},"MOTORCYCLE":{},"BUS":{"truth_points":0},
                "HEAVY_VEHICLE":{"truth_points":3,"spatial_matches":2,"correct_class_matches":1,
                    "class_errors_on_spatial_matches":1}}},
        "HEAVY_wrong_class_matched_points":1,
        "HEAVY_truth_tracks_with_class_disagreement":1,
        "truth_track_diagnostics":[{"truth_track_id":"7","truth_class":"HEAVY_VEHICLE",
            "wrong_class_matches":1}],
        "predicted_track_class_stability":[{"predicted_track_id":"26"}],
        "BUS_predicted_track_diagnostics":[],
        "BUS_unique_predicted_track_ids":0,
        "BUS_FLUID_truth_points":0,
        "unique_FLUID_truth_tracks":1, "unique_predicted_tracks":1,
        "truth_points":3,"matched_points":2,"predicted_points":2,
        "original_saved_FLUID_label_F1":0.8
    }

def test_verified_label_relative_payload(tmp_path):
    p=tmp_path/"audit.json"
    p.write_text(json.dumps(fixture()),encoding="utf-8")
    x=read_audit(p)
    assert x["heavy_truth_track_ids_with_disagreement"] == 1
    assert x["spatial_f1_pct"] == 80.0
    assert x["bus_label_accuracy_measurable"] is False
    assert x["production_release_approved"] is False

@pytest.mark.parametrize("change",[
    ("production_release_approved",True),
    ("independent_holdout_pass",True),
    ("no_new_detector_inference",False),
    ("HEAVY_wrong_class_matched_points",2),
    ("BUS_unique_predicted_track_ids",1),
    ("source_video_frame_window",[0,119]),
    ("truth_points",4),
])
def test_rejects_unverified_or_inconsistent_source(tmp_path,change):
    data=fixture()
    data[change[0]]=change[1]
    p=tmp_path/"audit.json"
    p.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(AuditDisplayError):
        read_audit(p)

def test_missing_research_is_not_verified(tmp_path):
    with pytest.raises(AuditDisplayError):
        read_audit(tmp_path/"missing.json")

def test_refuses_symlink(tmp_path):
    src=tmp_path/"real.json"
    src.write_text(json.dumps(fixture()),encoding="utf-8")
    alias=tmp_path/"alias.json"
    alias.symlink_to(src)
    with pytest.raises(AuditDisplayError):
        read_audit(alias)
