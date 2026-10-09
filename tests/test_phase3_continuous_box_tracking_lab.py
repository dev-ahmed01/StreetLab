"""Synthetic tracker contracts only: never mistaken for real ByteTrack validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streetlab_phase3.video.continuous_box_tracking_lab import (
    RawBox, ContinuousTrial, policy_matrix, as_tracker_detections,
    stream_tracking_matrix, STATUS,
)


class FakeDetections:
    def __init__(self, *, xyxy, confidence, class_id, tracker_id=None):
        self.xyxy=xyxy
        self.confidence=confidence
        self.class_id=class_id
        self.tracker_id=tracker_id


class FakeTracker:
    all=[]
    def __init__(self, **params):
        self.updates=[]
        self.params=params
        FakeTracker.all.append(self)
    def update(self, detection):
        self.updates.append(len(detection.xyxy))
        # Distinct track IDs consistently assigned to sequential objects.
        return FakeDetections(
            xyxy=detection.xyxy,confidence=detection.confidence,
            class_id=detection.class_id,
            tracker_id=np.asarray([i+1 for i in range(len(detection.xyxy))]))


def as_rows(det,frame):
    ids=np.asarray(det.tracker_id)
    out=[]
    for xy,score,cls,tid in zip(det.xyxy,det.confidence,det.class_id,ids):
        x1,y1,x2,y2=map(float,xy)
        x,y=(x1+x2)/2,(y1+y2)/2
        w,h=x2-x1,y2-y1
        out.append([frame,int(tid),x,y,w,h,x,y,w,h,int(cls),float(score),w,h])
    return out,0


def score_stub(*,tracks,fluid_tracks,max_distance_px,start_frame,end_frame):
    with tracks.open('r',encoding='utf-8') as f:
        rows=list(csv.reader(f))
    if rows and any(len(row)!=14 for row in rows):
        raise ValueError('Incorrect GeoTrax row shape')
    return (
        {'frame_offset':1,'max_distance_px':max_distance_px,
         'truth_points':8,'predicted_points':len(rows),
         'evaluation_frame_start':start_frame+1,
         'evaluation_frame_end':end_frame+1,
         'point_recall':min(len(rows)/8,1),
         'point_precision':1.0 if rows else None},
        {'frame_offset':1,'max_distance_px':max_distance_px,
         'fraction_truth_tracks_fragmented':0.0,
         'total_contiguous_id_switches':0,
         'predicted_tracks':len({r[1] for r in rows})},
    )


def raw_seq(*,gap=False,missing_last=False):
    for i in range(3 if missing_last else 4):
        frame=10+i+(int(gap) if i==2 else 0)
        if frame in (10,12):
            boxes=[]
        else:
            boxes=[RawBox(frame,100,100,120,120,.95,'MOTORCYCLE',0,0),
                   RawBox(frame,101,100,121,120,.70,'MOTORCYCLE',1,1),
                   RawBox(frame,200,100,290,190,.88,'BUS',2,2),
                   RawBox(frame,350,100,380,140,.81,'PEDESTRIAN',3,3)]
        yield frame,boxes,.02


def test_tracking_bank_updates_every_consecutive_frame_and_exports_no_fake_ids(tmp_path):
    FakeTracker.all=[]
    configurations=policy_matrix(('raw_unmerged','hard_nms_iou_0.50',
                                  'weighted_fusion_iou_0.50'))
    truth=tmp_path/'original_truth.csv';truth.write_text('frame,id,cx,cy,type\n')
    out=tmp_path/'tracks'
    report=stream_tracking_matrix(
        raw_stream=raw_seq(),start_frame=11,end_frame=13,first_frame=10,
        output_dir=out,fluid_tracks=truth,configs=configurations,
        tracker_factory=FakeTracker,detection_class=FakeDetections,
        rows_converter=as_rows,score_fn=score_stub)
    assert report['status']==STATUS
    assert report['eligible_for_promotion'] is False
    assert report['processed_frames']==4
    assert report['warmup_frames_processed']==1
    assert report['candidate_count']==3
    assert report['independent_tracker_instance_per_policy'] is True
    assert len(FakeTracker.all)==3 and len({id(t) for t in FakeTracker.all})==3
    assert all(len(t.updates)==4 for t in FakeTracker.all)
    assert all(t.updates[0]==0 and t.updates[2]==0 for t in FakeTracker.all)
    assert FakeTracker.all[0].updates==[0,3,0,3]
    assert FakeTracker.all[1].updates==[0,2,0,2]
    assert FakeTracker.all[2].updates==[0,2,0,2]
    assert all(item['identity']['predicted_tracks']>0 for item in report['policies'])
    assert set(f.name for f in out.glob('*.txt'))=={
        'raw_unmerged.txt','hard_nms_iou_0.50.txt','weighted_fusion_iou_0.50.txt'}
    assert (out/'batch_report.json').is_file()
    assert all(item['tracks_sha256'] for item in report['policies'])
    assert all(item['eligible_for_production'] is False for item in report['policies'])
    with pytest.raises(FileExistsError):
        stream_tracking_matrix(
            raw_stream=raw_seq(),start_frame=11,end_frame=13,first_frame=10,
            output_dir=out,fluid_tracks=truth,configs=configurations,
            tracker_factory=FakeTracker,detection_class=FakeDetections,
            rows_converter=as_rows,score_fn=score_stub)


def test_sparse_frame_tracking_is_rejected_and_no_partial_output(tmp_path):
    truth=tmp_path/'truth.csv';truth.write_text('frame,id,cx,cy,type\n')
    for variant in ('gap','missing_last'):
        root=tmp_path/variant
        with pytest.raises(ValueError,match='consecutive|Missing last'):
            stream_tracking_matrix(
                raw_stream=raw_seq(gap=variant=='gap',missing_last=variant=='missing_last'),
                first_frame=10,start_frame=11,end_frame=13,output_dir=root,
                fluid_tracks=truth,configs=policy_matrix(('raw_unmerged',)),
                tracker_factory=FakeTracker,detection_class=FakeDetections,
                rows_converter=as_rows,score_fn=score_stub)
        assert not root.exists()


def test_continuous_tracking_rejects_fabricated_ids_and_does_not_publish(tmp_path):
    truth=tmp_path/'truth.csv';truth.write_text('frame,id,cx,cy,type\n')
    def invalid_rows(tracked,frame):
        data,n=as_rows(tracked,frame)
        return data+data,n
    with pytest.raises(ValueError,match='Exported confirmed'):
        stream_tracking_matrix(
            raw_stream=raw_seq(),first_frame=10,start_frame=11,end_frame=13,
            output_dir=tmp_path/'invalid',fluid_tracks=truth,
            configs=policy_matrix(('raw_unmerged',)),
            tracker_factory=FakeTracker,detection_class=FakeDetections,
            rows_converter=invalid_rows,score_fn=score_stub)
    assert not (tmp_path/'invalid').exists()


def test_cross_class_geometry_never_remaps_non_vehicle(tmp_path):
    demo=[SimpleNamespace(x1=1,y1=2,x2=3,y2=4,confidence=.9,
                          vehicle_class='PEDESTRIAN'),
          SimpleNamespace(x1=1,y1=2,x2=3,y2=4,confidence=.8,
                          vehicle_class='MOTORCYCLE'),
          SimpleNamespace(x1=4,y1=5,x2=6,y2=7,confidence=.7,
                          vehicle_class='BICYCLE')]
    detections=as_tracker_detections(demo,FakeDetections)
    assert detections.xyxy.shape==(1,4)
    assert detections.class_id.tolist()==[3]
    empty=as_tracker_detections([],FakeDetections)
    assert empty.xyxy.shape==(0,4)
    assert empty.class_id.shape==(0,)


def test_policy_predeclared_and_cli_rejects_unknown_approach(tmp_path):
    assert len(policy_matrix())==25
    assert len({c['name'] for c in policy_matrix()})==25
    with pytest.raises(ValueError,match='Policy name'):
        policy_matrix(('made_up',))
    with pytest.raises(ValueError,match='unique'):
        policy_matrix(('raw_unmerged','raw_unmerged'))
    wrong=tmp_path/'anything'
    with pytest.raises(FileNotFoundError):
        ContinuousTrial(video=wrong,bundle=wrong,runtime_model=wrong,
                        fluid_tracks=wrong,output_dir=tmp_path/'out',
                        start_frame=0,end_frame=3).validate()


def test_real_bundle_provenance_is_read_only_when_locally_available(tmp_path):
    from streetlab_phase3.video.premerge_box_capture import bundle_data
    source=Path('/mnt/data/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip')
    if not source.exists():
        pytest.skip('User W04 bundle is not included in CI')
    frames,truth,provenance=bundle_data(source)
    assert len(frames)==21 and frames[0]==10750 and frames[-1]==11350
    assert provenance['frozen_reference']['matched_points']==670
    assert provenance['frozen_reference']['predicted_points']==857
    assert provenance['frozen_reference']['truth_points']==739


def test_baseline_cohort_drift_fails_closed_before_results_are_published(tmp_path):
    truth=tmp_path/'truth.csv';truth.write_text('frame,id,cx,cy,type\n')
    baseline=tmp_path/'T000.txt';baseline.write_text('')
    def wrong_baseline(*,tracks,fluid_tracks,max_distance_px,start_frame,end_frame):
        pixel,identity=score_stub(tracks=tracks,fluid_tracks=fluid_tracks,
                                  max_distance_px=max_distance_px,
                                  start_frame=start_frame,end_frame=end_frame)
        if tracks == baseline:
            pixel['truth_points'] = 100
        return pixel,identity
    with pytest.raises(ValueError,match='Baseline video cohort mismatch'):
        stream_tracking_matrix(raw_stream=raw_seq(),first_frame=10,
            start_frame=11,end_frame=13,output_dir=tmp_path/'drift',
            fluid_tracks=truth,configs=policy_matrix(('raw_unmerged',)),
            tracker_factory=FakeTracker,detection_class=FakeDetections,
            rows_converter=as_rows,score_fn=wrong_baseline,baseline_tracks=baseline)
    assert not (tmp_path/'drift').exists()


def test_nearby_real_motorcycles_can_be_suppressed_by_nms_so_never_auto_promote():
    from streetlab_phase3.video.integrated_box_lab import merge_boxes
    # Two physically independent motorcycles CAN occupy overlapping boxes.
    # Geometry alone has no authority to interpret physical identity.
    bikes=[RawBox(50,100,100,120,120,.95,'MOTORCYCLE',0,0),
           RawBox(50,102,100,122,120,.94,'MOTORCYCLE',1,1)]
    candidates=policy_matrix(('hard_nms_iou_0.50','raw_unmerged'))
    hard=merge_boxes(bikes,**{k:v for k,v in candidates[0].items() if k!='name'})
    raw=merge_boxes(bikes,**{k:v for k,v in candidates[1].items() if k!='name'})
    assert len(hard)==1 and len(raw)==2
    # The sidecar must be independently adjudicated before calling this safe.


def test_source_provenance_is_atomic_with_report(tmp_path):
    truth=tmp_path/'truth.csv';truth.write_text('frame,id,cx,cy,type\n')
    out=tmp_path/'run'
    data={'status':'synthetic_only','eligible_for_promotion':False}
    r=stream_tracking_matrix(raw_stream=raw_seq(),first_frame=10,
        start_frame=11,end_frame=13,output_dir=out,fluid_tracks=truth,
        configs=policy_matrix(('raw_unmerged',)),tracker_factory=FakeTracker,
        detection_class=FakeDetections,rows_converter=as_rows,score_fn=score_stub,
        provenance=data)
    assert (out/'source_provenance.json').is_file()
    assert json.loads((out/'source_provenance.json').read_text()) == data
    assert r['eligible_for_promotion'] is False
    with pytest.raises(ValueError,match='forbid promotion'):
        stream_tracking_matrix(raw_stream=raw_seq(),first_frame=10,
            start_frame=11,end_frame=13,output_dir=tmp_path/'bad',
            fluid_tracks=truth,configs=policy_matrix(('raw_unmerged',)),
            tracker_factory=FakeTracker,detection_class=FakeDetections,
            rows_converter=as_rows,score_fn=score_stub,
            provenance={'eligible_for_promotion':True})
    assert not (tmp_path/'bad').exists()


def test_contiguous_video_decoder_seeks_only_once_and_validates_absolute_frame():
    from streetlab_phase3.video.continuous_box_tracking_lab import sequential_video_rgb_frames
    class FakeVideo:
        def __init__(self):
            self.position=0
            self.seeks=[]
            self.released=False
        def isOpened(self): return True
        def set(self, key, pos):
            self.seeks.append(pos)
            self.position=int(pos)
            return True
        def read(self):
            self.position+=1
            return True,np.zeros((2160,3840,3),dtype=np.uint8)
        def get(self,key):return float(self.position)
        def release(self):self.released=True
    class FakeCv2:
        CAP_PROP_POS_FRAMES=1
        COLOR_BGR2RGB=2
        cap=FakeVideo()
        @classmethod
        def VideoCapture(cls,path):return cls.cap
        @staticmethod
        def cvtColor(img,code):return img
    frames=list(sequential_video_rgb_frames(Path('unused.mp4'),3,5,cv2_module=FakeCv2))
    assert [a for a,_ in frames]==[3,4,5]
    assert FakeCv2.cap.seeks==[3]
    assert FakeCv2.cap.released is True


def test_empty_supervision_result_without_ids_still_advances_tracking(tmp_path):
    class NoIdOnEmptyTracker(FakeTracker):
        def update(self,detection):
            result=super().update(detection)
            if not len(result.xyxy):
                result.tracker_id=None
            return result
    truth=tmp_path/'original.csv';truth.write_text('frame,id,cx,cy,type\n')
    report=stream_tracking_matrix(
        raw_stream=raw_seq(),first_frame=10,start_frame=11,end_frame=13,
        output_dir=tmp_path/'empty',fluid_tracks=truth,
        configs=policy_matrix(('raw_unmerged',)),
        tracker_factory=NoIdOnEmptyTracker,detection_class=FakeDetections,
        rows_converter=as_rows,score_fn=score_stub)
    assert report['processed_frames']==4
    assert len(list((tmp_path/'empty').glob('*.txt')))==1
    assert report['policies'][0]['evaluation_confirmed_rows']==6