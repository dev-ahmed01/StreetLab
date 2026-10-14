"""Test geometry and identity invariants, including counterexamples.

Tests do NOT treat overlap as physical ground truth and do not train/tune using
the seven W04 human review decisions.
"""
from __future__ import annotations

from pathlib import Path
import pytest
from streetlab_phase3.video.physical_object_correspondence import (
    overlap_evidence,compare_shadow_to_primary,IOS_CONTAINED,
    MAX_PART_AREA_RATIO,
)


def box(frame,tid,klass,x1=0.,y1=0.,x2=10.,y2=10.,shadow=False):
    return {"frame":frame,
            "shadow_id" if shadow else "primary_id":tid,
            "vehicle_class":klass,
            "x1":x1,"y1":y1,"x2":x2,"y2":y2}


def test_part_inside_larger_different_class_does_not_become_new_vehicle():
    shadow=box(1,77,"HEAVY_VEHICLE",10.,10.,30.,30.,True)
    primary=box(1,2,"CAR",0.,0.,100.,60.)
    evidence=overlap_evidence(shadow,primary)
    assert evidence["ios"]==1.0
    assert evidence["iou"]<.5
    assert evidence["relation"]=="PART_WHOLE_OVERLAP"
    assert not evidence["class_agrees"]
    assert evidence["same_physical_object_proven"] is False


def test_similar_extent_and_adjacent_boxes_not_silently_merged():
    a=box(1,2,"MOTORCYCLE",0.,0.,20.,20.,True)
    same=box(1,3,"MOTORCYCLE",1.,1.,21.,21.)
    adjacent=box(1,4,"CAR",18.,0.,38.,20.)
    assert overlap_evidence(a,same)["relation"]=="SAME_EXTENT_OVERLAP"
    assert overlap_evidence(a,adjacent)["relation"]=="NO_STRONG_GEOMETRIC_RELATION"


def test_close_occluded_neighbours_are_only_hypotheses():
    a=box(5,4,"HEAVY_VEHICLE",20.,20.,40.,40.,True)
    p=box(5,8,"CAR",0.,0.,50.,50.)
    q=box(5,9,"BUS",10.,10.,60.,60.)
    obs,pairs,tracklets=compare_shadow_to_primary([a],{5:[p,q]})
    assert obs[0]["plausible_primary_count"]==2
    assert tracklets[0]["repeated_correspondence_status"]=="WEAK_OR_DISCONTINUOUS_SPATIAL_CORRESPONDENCE"
    assert tracklets[0]["physical_object_duplicate_verified"] is False
    assert all(not x["physical_object_confirmed"] for x in pairs)


def test_continuous_multi_frame_support_does_not_edit_shadow_track():
    shadows=[box(i,101,"MOTORCYCLE",10.,10.,25.,25.,True)
             for i in range(100,105)]
    primary={i:[box(i,3,"CAR",0.,0.,50.,50.)] for i in range(100,105)}
    original=[dict(s) for s in shadows]
    obs,pairs,tracklets=compare_shadow_to_primary(shadows,primary)
    assert shadows==original
    assert len(obs)==5 and len(pairs)==5
    item=tracklets[0]
    assert item["primary_id_hypothesis"]==3
    assert item["primary_support_frames"]==5
    assert item["primary_consecutive_support"]==5
    assert item["repeated_correspondence_status"]=="REPEATED_GEOMETRIC_PRIMARY_CORRESPONDENCE"
    assert not item["physical_object_duplicate_verified"]
    assert not item["eligible_for_count"]


def test_two_consistent_primary_tracks_remain_explicitly_ambiguous():
    shadow=[box(i,101,"MOTORCYCLE",15.,15.,25.,25.,True)
            for i in range(10,14)]
    primary={i:[
        box(i,1,"CAR",0.,0.,40.,40.),
        box(i,2,"HEAVY_VEHICLE",10.,10.,30.,30.),
    ] for i in range(10,14)}
    _,_,tracklets=compare_shadow_to_primary(shadow,primary)
    assert tracklets[0]["repeated_correspondence_status"]=="MULTIPLE_PRIMARY_OBJECT_HYPOTHESES"
    assert tracklets[0]["independent_alternative_primary_hypotheses"]==1


def test_temporal_support_does_not_stitch_different_primary_ids():
    shadow=[box(i,51,"HEAVY_VEHICLE",15.,15.,25.,25.,True)
            for i in range(200,204)]
    primary={200:[box(200,1,"CAR",0.,0.,40.,40.)],
             201:[box(201,1,"CAR",0.,0.,40.,40.)],
             202:[box(202,2,"CAR",0.,0.,40.,40.)],
             203:[box(203,2,"CAR",0.,0.,40.,40.)]}
    _,_,tracklets=compare_shadow_to_primary(shadow,primary)
    assert tracklets[0]["repeated_correspondence_status"]=="WEAK_OR_DISCONTINUOUS_SPATIAL_CORRESPONDENCE"
    assert not tracklets[0]["eligible_for_count"]


def test_no_overlap_preserves_unmatched_for_future_review():
    shadow=[box(10,42,"MOTORCYCLE",200.,200.,220.,220.,True)]
    primary={10:[box(10,8,"MOTORCYCLE",0.,0.,20.,20.)]}
    obs,pairs,tracklets=compare_shadow_to_primary(shadow,primary)
    assert obs[0]["best_spatial_relation"]=="NO_SPATIAL_PAIR"
    assert obs[0]["best_primary_id_spatial_only"] is None
    assert not pairs
    assert tracklets[0]["repeated_correspondence_status"]=="NO_SPATIAL_CORRESPONDENCE"


def test_rejects_duplicate_ids_invalid_boxes_and_mixed_frames():
    s=box(11,5,"CAR",2.,2.,5.,5.,True)
    p=box(11,8,"CAR",0.,0.,20.,20.)
    with pytest.raises(ValueError,match="Repeated shadow"):
        compare_shadow_to_primary([s,s],{11:[p]})
    with pytest.raises(ValueError,match="another source frame"):
        compare_shadow_to_primary([s],{11:[box(12,8,"CAR",0.,0.,20.,20.)]})
    with pytest.raises(ValueError,match="Duplicated original"):
        compare_shadow_to_primary([s],{11:[p,p]})
    bad=dict(s,x2=2.)
    with pytest.raises(ValueError,match="positive pixel"):
        overlap_evidence(bad,p)
