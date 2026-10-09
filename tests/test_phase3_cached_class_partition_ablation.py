"""Synthetic invariants only; actual W04 replay is run on the user's Windows evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

RUNNER = Path(__file__).resolve().parents[1]/'scripts'/'phase3_cached_class_partition_ablation.py'
if not RUNNER.exists():
    pytest.skip('Class-partition runner intentionally omitted by sparse checkout',
                allow_module_level=True)
spec = importlib.util.spec_from_file_location('w04_partition',RUNNER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Detections:
    def __init__(self,*,xyxy,confidence,class_id,tracker_id=None):
        self.xyxy=xyxy
        self.confidence=confidence
        self.class_id=class_id
        self.tracker_id=tracker_id


class Tracker:
    def __init__(self):
        self.updates=[]
    def update(self,det):
        self.updates.append(det.class_id.tolist())
        return Detections(xyxy=det.xyxy,confidence=det.confidence,
                          class_id=det.class_id,
                          tracker_id=np.arange(len(det.xyxy),dtype=int)+1)


def converter(detection,frame):
    result=[]
    for bounds,c,conf,tid in zip(detection.xyxy,detection.class_id,
                                  detection.confidence,detection.tracker_id):
        x1,y1,x2,y2=map(float,bounds)
        w,h=x2-x1,y2-y1
        cx,cy=x1+w/2,y1+h/2
        result.append([frame,int(tid),cx,cy,w,h,cx,cy,w,h,int(c),
                       float(conf),w,h])
    return result,0


def test_class_scoped_ids_do_not_collide():
    assert len({mod.encode_class_identity(cls,tid)
                for cls in mod.CLASSES for tid in (0,1,3)})==12
    with pytest.raises(ValueError):
        mod.encode_class_identity(4,1)
    with pytest.raises(ValueError):
        mod.encode_class_identity(0,-1)


def test_separate_trackers_update_on_empty_frames_and_preserve_class():
    class_names=('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE')
    boxes=[SimpleNamespace(x1=0,y1=0,x2=10,y2=20,
                           confidence=.9,vehicle_class=label)
           for label in class_names]
    states={cls:Tracker() for cls in mod.CLASSES}
    rows,_=mod.step_partitioned(states,boxes,10750,Detections,converter)
    assert len(rows)==4
    assert len({row[1] for row in rows})==4
    assert {r[10] for r in rows}==set(mod.CLASSES)
    empty,unconfirmed=mod.step_partitioned(states,[],10751,Detections,converter)
    assert empty==[] and unconfirmed==0
    assert all(len(track.updates)==2 and track.updates[1]==[]
               for track in states.values())


def test_wrong_class_in_tracker_output_is_rejected():
    states={cls:Tracker() for cls in mod.CLASSES}
    bad=[SimpleNamespace(x1=0,y1=0,x2=10,y2=10,
                         confidence=.8,vehicle_class='BUS')]
    def wrong_class(tracked,frame):
        rows,_=converter(tracked,frame)
        for row in rows:row[10]=0
        return rows,0
    with pytest.raises(ValueError,match='wrong class'):
        mod.step_partitioned(states,bad,10750,Detections,wrong_class)


def test_class_change_count_separates_original_from_partitioned(tmp_path):
    p=tmp_path/'tracks.txt'
    first=[10750,4,1,1,4,4,1,1,4,4,0,.8,4,4]
    second=[10751,4,1,1,4,4,1,1,4,4,1,.8,4,4]
    p.write_text(','.join(map(str,first))+'\n'+','.join(map(str,second))+'\n')
    assert mod.count_class_flip_events(p)==(1,1)


def test_reject_existing_ablation_destination(tmp_path):
    dest=tmp_path/'output';dest.mkdir()
    with pytest.raises(FileExistsError):
        mod.execute(tmp_path,dest)
