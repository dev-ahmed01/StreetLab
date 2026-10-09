"""Adversarial geometry / real-bundle integration tests (no OpenVINO required)."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streetlab_phase3.video.integrated_box_lab import (
    RawBox, merge_boxes, overlap, candidate_matrix, score_centers,
    evaluate_candidates,
)
from streetlab_phase3.video.premerge_box_capture import (
    bundle_data, capture_boxes, run_integrated_lab,
)


def b(index,x1,y1,x2,y2,score=.8,klass='MOTORCYCLE',tile=0,frame=10):
    return RawBox(frame,x1,y1,x2,y2,score,klass,tile,index)


def test_overlap_iou_and_ios_and_validation():
    a,bx=b(1,0,0,100,100),b(2,10,10,50,50)
    assert overlap(a,a)==1.
    assert overlap(a,bx,'ios')==1.
    assert 0<overlap(a,bx,'iou')<.25
    with pytest.raises(ValueError,match='metric'):
        overlap(a,bx,'invalid')
    with pytest.raises(ValueError,match='Invalid original raw box'):
        b(4,40,40,20,20)


def test_hard_nms_dedup_and_preserve_nearby_real_motorcycles():
    # Identical competing boxes should merge; two genuine motorcycles
    # 14px apart should NOT be blindly removed by center distance.
    raw=[b(0,0,0,12,12,.93),b(1,1,0,13,12,.81,tile=1),
         b(2,14,0,26,12,.90,tile=2),
         b(3,0,0,12,12,.89,'CAR',tile=3)]
    out=merge_boxes(raw,policy='hard_nms',threshold=.50)
    assert len(out)==3
    assert sum(x.vehicle_class=='MOTORCYCLE' for x in out)==2
    assert any(x.vehicle_class=='CAR' for x in out)
    # Cross-class hard NMS is exploratory and explicitly not default.
    aggressive=merge_boxes(raw,policy='hard_nms',threshold=.50,class_agnostic=True)
    assert len(aggressive)==2
    assert len(merge_boxes(raw,policy='none'))==4


def test_soft_nms_and_weighted_fusion_finite_and_deterministic():
    raw=[b(0,0,0,100,100,.9),b(1,8,0,108,100,.8,tile=1),
         b(2,220,0,260,30,.7,tile=2)]
    for policy in ('soft_linear','soft_gaussian','weighted_fusion'):
        x=merge_boxes(raw,policy=policy,threshold=.5,score_floor=.15)
        y=merge_boxes(list(reversed(raw)),policy=policy,threshold=.5,score_floor=.15)
        assert x==y
        assert all(0<=p.confidence<=1 for p in x)
        assert sum(p.vehicle_class=='MOTORCYCLE' for p in x)>=2
    f=merge_boxes(raw,policy='weighted_fusion',threshold=.5)
    fused=[p for p in f if len(p.contributors)>1]
    assert len(fused)==1
    assert fused[0].confidence==.9
    assert fused[0].contributors==(0,1)
    with pytest.raises(ValueError,match='undefined'):
        merge_boxes(raw,policy='weighted_fusion',class_agnostic=True)


def test_frame_boundary_and_source_uniqueness():
    assert len(merge_boxes([b(0,0,0,10,10,frame=1),
                            b(1,0,0,10,10,frame=2)],policy='hard_nms'))==2
    with pytest.raises(ValueError,match='Repeated'):
        merge_boxes([b(0,0,0,10,10),b(0,1,1,11,11)],policy='hard_nms')


def test_all_25_candidate_matrix_configs_are_unique_and_label_blind():
    matrix=candidate_matrix()
    assert len(matrix)==25
    assert len({p['name'] for p in matrix})==25
    assert not any(x['class_agnostic'] for x in matrix)


def test_class_aware_pixel_scoring_protects_motorcycles():
    truth=[dict(frame='11',id='1',cx='10',cy='10',type='moped'),
           dict(frame='11',id='2',cx='40',cy='10',type='moped'),
           dict(frame='11',id='3',cx='100',cy='10',type='car'),
           dict(frame='11',id='4',cx='',cy='',type='truck')]
    detections=[b(1,5,5,15,15,.81),b(2,35,5,45,15,.78),
                b(3,95,5,105,15,.9,'CAR')]
    s=score_centers(detections,truth,[10]);assert s['truth_points']==3
    assert s['matched_points']==3 and s['per_class']['MOTORCYCLE']['matched']==2
    assert s['unprojected_supported_truth_rows']==1
    with pytest.raises(ValueError,match='Frozen'):
        score_centers(detections,truth,[10],frame_offset=0)
    r=evaluate_candidates(detections,truth,[10],configs=[candidate_matrix()[0]])
    assert r['eligible_for_promotion'] is False
    assert r['candidates'][0]['guardrails']['eligible_for_production'] is False


def test_absolute_tile_origins_and_six_classes(monkeypatch):
    class O:
        def __init__(self,klass,x=2):
            self.category=SimpleNamespace(name=klass)
            self.bbox=SimpleNamespace(to_xyxy=lambda:[x,2,x+10,12])
            self.score=SimpleNamespace(value=.7)
    def slicer(**kw):return [[0,0,40,40],[30,0,70,40]]
    def predictor(tile,model,verbose=0):
        return SimpleNamespace(object_prediction_list=[O('Motorcycle'),O('Pedestrian'),O('Bicycle')])
    rgb=np.zeros((40,70,3),dtype=np.uint8)
    raw,stats=capture_boxes([(10,rgb)],object(),slicer=slicer,predictor=predictor)
    assert len(raw)==6 and stats[0]['tile_count']==2
    assert sorted({r.x1 for r in raw})==[2,32]
    assert sorted({r.vehicle_class for r in raw})==['BICYCLE','MOTORCYCLE','PEDESTRIAN']
    assert {r.tile_id for r in raw}=={0,1}


def test_reject_already_shifted_coordinates():
    def slicer(**kwargs):return [[50,0,100,50]]
    class O:
        category=SimpleNamespace(name='Car')
        bbox=SimpleNamespace(to_xyxy=lambda:[75,0,80,10])
        score=SimpleNamespace(value=.9)
    predictor=lambda *args,**kwargs: SimpleNamespace(object_prediction_list=[O()])
    with pytest.raises(ValueError,match='outside crop'):
        capture_boxes([(10,np.zeros((50,100,3)))],object(),slicer=slicer,predictor=predictor)


def real_bundle_path():return Path('/mnt/data/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip')

@pytest.mark.skipif(not real_bundle_path().is_file(),reason='Optional local W04 bundle')
def test_real_w04_bundle_sha_and_frozen_scores():
    frames,labels,ev=bundle_data(real_bundle_path())
    with Path('/mnt/data/streetlab_integrated/input/audits/openvino/sliced_detections.csv').open() as f:
        points=list(csv.DictReader(f))
    got=score_centers(points,labels,frames)
    assert (got['truth_points'],got['predicted_points'],got['matched_points'])==(739,857,670)
    assert got['per_class']['MOTORCYCLE']['matched']==320
    assert ev['checkpoint_sha256'].startswith('7d462ae523b1')


def test_staged_output_immutable_and_no_model_requirement(monkeypatch,tmp_path):
    import streetlab_phase3.video.premerge_box_capture as mod
    raw=[b(0,0,0,10,10,frame=10750)]
    mock=lambda *args,**kwargs:(raw,[{'frame':10750,'tile_count':1,'captured_box_count':1,'elapsed_s':.1}])
    monkeypatch.setattr(mod,'capture_boxes',mock)
    monkeypatch.setattr(mod,'sample_rgb_frames',lambda *args,**kwargs:[(10750,np.zeros((1,1,3)))])
    def loader(**kwargs):return object()
    model=tmp_path/'model';model.mkdir();(model/'a.xml').write_text('xml');(model/'b.bin').write_bytes(b'b')
    # fake source archive evidence must still pass checksum validation
    source=real_bundle_path()
    if not source.is_file():pytest.skip('Requires local original evidence archive')
    folder=tmp_path/'output'
    result=mod.run_integrated_lab(bundle=source,model_path=model,output_dir=folder,loader=loader)
    assert result['candidate_count']==25 and folder.joinpath('matrix_report.json').is_file()
    assert folder.joinpath('pre_global_merge_boxes.jsonl').is_file()
    with pytest.raises(FileExistsError):
        mod.run_integrated_lab(bundle=source,model_path=model,output_dir=folder,loader=loader)
