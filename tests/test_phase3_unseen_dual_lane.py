"""Synthetic dual-lane integration: genuine per-frame updates, never physical scores."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from streetlab_phase3.video.integrated_box_lab import RawBox
from streetlab_phase3.video.unseen_dual_lane import (
    dual_lane_stream, policy_merge, MIN_EVAL, MIN_WARMUP,
    STATUS, PRIMARY_NAME, SHADOW_NAME,
)


class Detections:
    def __init__(self,*,xyxy,confidence,class_id,tracker_id=None):
        self.xyxy=np.asarray(xyxy)
        self.confidence=np.asarray(confidence)
        self.class_id=np.asarray(class_id)
        self.tracker_id=tracker_id

class Tracker:
    instances=[]
    def __init__(self,**kwargs):
        self.updates=[]
        self.params=kwargs
        Tracker.instances.append(self)
    def update(self,det):
        self.updates.append(len(det.xyxy))
        return Detections(xyxy=det.xyxy,confidence=det.confidence,
                          class_id=det.class_id,
                          tracker_id=np.arange(len(det.xyxy),dtype=int)+1)

def converter(det,frame):
    out=[]
    for bb,cls,conf,tid in zip(det.xyxy,det.class_id,
                               det.confidence,det.tracker_id):
        x1,y1,x2,y2=map(float,bb)
        cx,cy=(x1+x2)/2,(y1+y2)/2
        w,h=x2-x1,y2-y1
        out.append([frame,int(tid),cx,cy,w,h,cx,cy,w,h,int(cls),
                    float(conf),w,h])
    return out,0

def raw(frame):
    return [
        RawBox(frame,0,0,10,10,.9,"CAR",0,0),
        RawBox(frame,100,0,110,10,.9,"MOTORCYCLE",0,1),
        RawBox(frame,105,0,115,10,.8,"MOTORCYCLE",1,2),
    ]

def sequence(*,skip_at=None):
    for frame in range(150):
        if frame==skip_at:
            continue
        yield frame,raw(frame),.001


def test_one_inference_stream_both_actual_tracker_updates_every_frame(tmp_path):
    Tracker.instances=[]
    out=tmp_path/"dual_lanes"
    result=dual_lane_stream(
        sequence(),first=0,start=30,end=149,out=out,fps=30.,
        source_provenance={"eligible_for_production":False,"status":"SYNTHETIC_CI"},
        tracker_factory=Tracker,detection_class=Detections,
        rows_converter=converter)
    assert result["status"]==STATUS and result["eligible_for_production"] is False
    assert result["inference_passes_per_frame"]==1
    assert result["tracking_updates_per_policy"]==150
    assert result["evaluation_frames"]==120
    assert result["raw_box_count"]==450
    assert len(Tracker.instances)==2
    assert all(len(x.updates)==150 for x in Tracker.instances)
    assert Tracker.instances[0].updates==[2]*150
    assert Tracker.instances[1].updates==[3]*150
    for key,expected in (("primary",240),("shadow",360)):
        item=result["tracking"][key]
        with (out/item["filename"]).open(newline="") as stream:
            rows=list(csv.reader(stream))
        assert len(rows)==expected and len(rows[0])==14
        assert int(rows[0][0])==30 and int(rows[-1][0])==149
        assert item["sha256"]
    assert result["two_lane_eval_wall_p95_seconds"]>=result["detector_eval_wall_p95_seconds"]
    assert (out/"per_frame_timing.csv").is_file()
    assert (out/"original_pre_global_merge_boxes.jsonl").is_file()
    with pytest.raises(FileExistsError):
        dual_lane_stream(sequence(),first=0,start=30,end=149,out=out,fps=30.,
                         source_provenance={"eligible_for_production":False},
                         tracker_factory=Tracker,detection_class=Detections,
                         rows_converter=converter)


def test_nonconsecutive_frame_detection_fails_closed_without_publication(tmp_path):
    path=tmp_path/"gapped"
    with pytest.raises(ValueError,match="consecutive"):
        dual_lane_stream(sequence(skip_at=20),first=0,start=30,end=149,
                         out=path,fps=30.,
                         source_provenance={"eligible_for_production":False},
                         tracker_factory=Tracker,detection_class=Detections,
                         rows_converter=converter)
    assert not path.exists()


def test_rejects_fabricated_original_id_and_invalid_provenance(tmp_path):
    def invalid_rows(det,frame):
        rows,_=converter(det,frame)
        return rows+rows,0
    with pytest.raises(ValueError,match="confirmed IDs"):
        dual_lane_stream(sequence(),first=0,start=30,end=149,
                         out=tmp_path/"bad",fps=30.,
                         source_provenance={"eligible_for_production":False},
                         tracker_factory=Tracker,detection_class=Detections,
                         rows_converter=invalid_rows)
    assert not (tmp_path/"bad").exists()
    with pytest.raises(ValueError,match="forbid promotion"):
        dual_lane_stream(sequence(),first=0,start=30,end=149,
                         out=tmp_path/"bad_origin",fps=30.,
                         source_provenance={"eligible_for_production":True},
                         tracker_factory=Tracker,detection_class=Detections)


def test_policy_preserves_original_merge_and_rare_geometry():
    primary=policy_merge(raw(7),"primary")
    shadow=policy_merge(raw(7),"shadow")
    assert len(primary)==2 and len(shadow)==3
    assert sum(x.vehicle_class=="CAR" for x in primary)==1
    assert sum(x.vehicle_class=="CAR" for x in shadow)==1
    with pytest.raises(ValueError):
        policy_merge(raw(7),"other")


def test_frozen_bounds_for_unseen_evaluation(tmp_path):
    with pytest.raises(ValueError,match="bounds"):
        dual_lane_stream(sequence(),first=0,start=29,end=148,
                         out=tmp_path/"shortwarm",fps=30.,
                         source_provenance={"eligible_for_production":False},
                         tracker_factory=Tracker,detection_class=Detections)
    with pytest.raises(ValueError,match="bounds"):
        dual_lane_stream(sequence(),first=0,start=30,end=120,
                         out=tmp_path/"shorteval",fps=30.,
                         source_provenance={"eligible_for_production":False},
                         tracker_factory=Tracker,detection_class=Detections)
