"""Build 4 fail-closed evidence tests. ALL sample metrics are synthetic."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from streetlab_phase3.video.holdout_validation import (
    LOCK_STATUS, VALIDATION_STATUS, lock_policy, sha, validate_holdout,
    verify_batch,
)
from streetlab_phase3.video.continuous_box_tracking_lab import STATUS, policy_matrix


def fixture_batch(tmp_path, label, *, video_data=b'W04 video', policy='raw_unmerged',
                  frames=180, warmup=50, model_sha='m'*64, with_baseline=True):
    directory=tmp_path/label
    directory.mkdir()
    first=warmup
    last=warmup+frames-1
    video=tmp_path/(label+'.mp4');video.write_bytes(video_data)
    fluid=tmp_path/(label+'.csv');fluid.write_text(f'frame,id,cx,cy,type\n1,{label},1,1,pedestrian\n')
    raw=directory/'original_pre_global_merge_boxes.jsonl';raw.write_text('{}\n')
    cfg=next(p for p in policy_matrix() if p['name']==policy)
    tracks=directory/(policy+'.txt')
    tracks.write_text(f'{first},1,100,120,20,20,100,120,20,20,3,0.8,20,20\n')
    pixel={'frame_offset':1,'max_distance_px':50.,'evaluation_frame_start':first+1,
           'evaluation_frame_end':last+1,'truth_points':800,'truth_tracks':70,
           'point_recall':.9,'point_precision':.94}
    identity={'frame_offset':1,'max_distance_px':50.,
              'fraction_truth_tracks_fragmented':.04,'total_contiguous_id_switches':0}
    classes={'by_class':{'MOTORCYCLE':{'correct_class_recall':.85},
                         'CAR':{'correct_class_recall':.90}}}
    p={'name':policy,'settings':cfg,'tracks':tracks.name,'tracks_sha256':sha(tracks),
       'evaluation_detections':1,'evaluation_confirmed_rows':1,
       'evaluation_unconfirmed_rows':0,'evaluation_merged_boxes_all_classes':1,
       'postprocess_median_seconds':.04,'tracking_median_seconds':.02,
       'pixel':pixel,'identity':identity,'class_diagnostics':classes,
       'eligible_for_production':False}
    (directory/(policy+'.score.json')).write_text(json.dumps(p))
    baseline_file=tmp_path/(label+'.T000.txt')
    baseline_file.write_bytes(tracks.read_bytes())
    baseline=(dict(pixel=copy.deepcopy(pixel),identity=copy.deepcopy(identity),
                   class_diagnostics=copy.deepcopy(classes),source_sha256=sha(baseline_file))
              if with_baseline else None)
    batch={'status':STATUS,'eligible_for_promotion':False,
           'tracking_updates_are_consecutive':True,
           'independent_tracker_instance_per_policy':True,
           'evaluated_first_frame':first,'evaluated_last_frame':last,
           'first_decoded_frame':0,'warmup_frames_processed':warmup,
           'processed_frames':last+1,'evaluated_frames':frames,
           'candidate_count':1,'raw_tile_boxes':10,
           'raw_pre_global_merge_boxes_file':raw.name,
           'raw_pre_global_merge_boxes_sha256':sha(raw),
           'detector_median_seconds':.15,'policies':[p],
           'same_window_T000':baseline,
           'physical_object_review_available':False,
           'unseen_footage_validated':False,
           'benchmark_is_continuous_tracking_not_sparse_W04_21':True}
    (directory/'batch_report.json').write_text(json.dumps(batch))
    provenance={'status':'EXPERIMENTAL_OPENVINO_FULL_SOURCE_VIDEO_PROVENANCE',
                'eligible_for_promotion':False,
                'video_path':str(video),'video_sha256':sha(video),
                'fluid_path':str(fluid),'fluid_sha256':sha(fluid),
                'bundle_sha256':'b'*64,'openvino_model_sha256':model_sha,
                'video_frame_window':[first,last],
                'policies':[policy]}
    (directory/'source_provenance.json').write_text(json.dumps(provenance))
    return directory,video,fluid


def make_lock(tmp_path):
    development,video,fluid=fixture_batch(tmp_path,'develop',with_baseline=False)
    path=tmp_path/'locked.json'
    result=lock_policy(development,'raw_unmerged',path)
    assert result['status']==LOCK_STATUS
    return path,development


def test_verify_integrity_and_production_never_promoted(tmp_path):
    folder,video,fluid=fixture_batch(tmp_path,'good')
    verified=verify_batch(folder)
    assert verified['original_source_sha256_verified'] is True
    assert verified['policy_count']==1
    old=sha(folder/'batch_report.json')
    assert verify_batch(folder)['batch_sha256']==old
    assert sha(fluid)==verified['provenance']['fluid_sha256']


def test_reject_swapped_track_bytes_and_stale_scorecard(tmp_path):
    folder,_,_=fixture_batch(tmp_path,'good')
    with (folder/'raw_unmerged.txt').open('a') as stream:
        stream.write('110,1,100,120,20,20,100,120,20,20,3,0.8,20,20\n')
    with pytest.raises(ValueError,match='checksum'):
        verify_batch(folder)
    folder2,_,_=fixture_batch(tmp_path,'other')
    score=folder2/'raw_unmerged.score.json';obj=json.loads(score.read_text())
    obj['pixel']['point_precision']=1.0
    score.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='scorecard disagrees'):
        verify_batch(folder2)


def test_bad_track_identity_and_frame_fail_even_with_forged_hash(tmp_path):
    folder,_,_=fixture_batch(tmp_path,'bad')
    path=folder/'raw_unmerged.txt'
    line=path.read_text()
    path.write_text(line+line)
    report=json.loads((folder/'batch_report.json').read_text())
    report['policies'][0]['tracks_sha256']=sha(path)
    report['policies'][0]['evaluation_confirmed_rows']=2
    (folder/'batch_report.json').write_text(json.dumps(report))
    (folder/'raw_unmerged.score.json').write_text(json.dumps(report['policies'][0]))
    with pytest.raises(ValueError,match='Duplicate genuine tracker ID'):
        verify_batch(folder)


def test_cannot_claim_same_video_as_unseen_holdout(tmp_path):
    lock,dev=make_lock(tmp_path)
    hold,_,_=fixture_batch(tmp_path,'held',video_data=b'W04 video')
    result=validate_holdout(lock,hold,tmp_path/'result.json')
    assert result['status']==VALIDATION_STATUS
    assert result['eligible_for_production'] is False
    assert result['eligible_for_tracking_promotion'] is False
    assert result['numerical_validation_passed'] is False
    assert any('Same video' in problem for problem in result['unmet_requirements'])


def test_unseen_numeric_pass_still_blocked_by_review_and_release_proof(tmp_path):
    lock,_=make_lock(tmp_path)
    hold,video,fluid=fixture_batch(tmp_path,'held',video_data=b'completely different video')
    result=validate_holdout(lock,hold,tmp_path/'report.json')
    assert result['numerical_thresholds_passed'] is True
    assert result['numerical_validation_passed'] is False # replay still required
    assert result['eligible_for_production'] is False
    assert len(result['unmet_requirements'])==5
    assert 'independent' in result['unmet_requirements'][0].lower()
    assert result['comparisons']['motorcycle_correct_class_recall_delta_pp']==0
    assert result['comparisons']['contiguous_id_switches_added']==0
    with pytest.raises(FileExistsError):
        validate_holdout(lock,hold,tmp_path/'report.json')


def test_regressed_motorcycle_recall_and_fragmentation_are_blocked(tmp_path):
    lock,_=make_lock(tmp_path)
    hold,_,_=fixture_batch(tmp_path,'held',video_data=b'another video')
    batch=hold/'batch_report.json';obj=json.loads(batch.read_text())
    policy=obj['policies'][0]
    policy['class_diagnostics']['by_class']['MOTORCYCLE']['correct_class_recall']=.75
    policy['identity']['fraction_truth_tracks_fragmented']=.20
    policy['identity']['total_contiguous_id_switches']=10
    batch.write_text(json.dumps(obj))
    (hold/'raw_unmerged.score.json').write_text(json.dumps(policy))
    r=validate_holdout(lock,hold,tmp_path/'bad.json')
    assert not r['numerical_validation_passed']
    assert any('motorcycle' in why for why in r['unmet_requirements'])
    assert any('fragmentation' in why for why in r['unmet_requirements'])
    assert any('switch' in why for why in r['unmet_requirements'])


def test_development_lock_must_match_policy_and_original_sha(tmp_path):
    lock,dev=make_lock(tmp_path)
    with pytest.raises(ValueError,match='Policy absent'):
        lock_policy(dev,'hard_nms_iou_0.70',tmp_path/'unknown.json')
    orig=dev/'source_provenance.json'
    data=json.loads(orig.read_text())
    data['video_sha256']='not sha'
    orig.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='missing or altered'):
        lock_policy(dev,'raw_unmerged',tmp_path/'bad.json')


def test_absent_same_cohort_t000_and_insufficient_window_do_not_pass(tmp_path):
    lock,_=make_lock(tmp_path)
    hold,_,_=fixture_batch(tmp_path,'short',video_data=b'new video',frames=30,warmup=5,with_baseline=False)
    r=validate_holdout(lock,hold,tmp_path/'short.json')
    assert not r['numerical_validation_passed']
    assert any('Fewer continuous' in s for s in r['unmet_requirements'])
    assert any('warmup' in s for s in r['unmet_requirements'])
    assert any('T000' in s for s in r['unmet_requirements'])


def test_source_video_tampering_rejected(tmp_path):
    folder,video,_=fixture_batch(tmp_path,'original')
    video.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Original source missing or altered'):
        verify_batch(folder)


def test_baseline_bytes_must_match_saved_same_window_baseline(tmp_path):
    lock,_=make_lock(tmp_path)
    hold,_,_=fixture_batch(tmp_path,'held',video_data=b'new camera')
    baseline=tmp_path/'held.T000.txt'
    baseline.write_text(baseline.read_text()+'garbage\n')
    with pytest.raises(ValueError,match='T000 track checksum mismatch'):
        validate_holdout(lock,hold,tmp_path/'no.json',baseline_tracks=baseline)
    assert not (tmp_path/'no.json').exists()


def test_score_replay_detects_changed_pixel_and_class_stats(tmp_path):
    lock,_=make_lock(tmp_path)
    hold,_,_=fixture_batch(tmp_path,'held',video_data=b'new sensor')
    baseline=tmp_path/'held.T000.txt'
    report=json.loads((hold/'batch_report.json').read_text())
    def mocked_independent_replay(**args):
        return ({'frame_offset':1,'max_distance_px':50.,
                 'truth_points':800,'point_recall':.92},
                {'frame_offset':1,'max_distance_px':50.})
    with pytest.raises(ValueError,match='Independent pixel score replay mismatch'):
        validate_holdout(lock,hold,tmp_path/'no.json',
                         baseline_tracks=baseline,scorer=mocked_independent_replay)
    assert not (tmp_path/'no.json').exists()


def test_injected_replay_can_check_consistency_but_never_counts_as_real_validation(tmp_path):
    lock,_=make_lock(tmp_path)
    hold,_,_=fixture_batch(tmp_path,'held',video_data=b'new actual footage')
    baseline=tmp_path/'held.T000.txt'
    expected=json.loads((hold/'batch_report.json').read_text())['policies'][0]
    def synthetic_replay(**args):
        return (expected['pixel'],expected['identity'])
    r=validate_holdout(lock,hold,tmp_path/'r.json',
                       baseline_tracks=baseline,scorer=synthetic_replay)
    assert r['numerical_thresholds_passed'] is True
    assert r['numerical_validation_passed'] is False
    assert r['independent_scores_recomputed_from_raw_tracks'] is False
    assert r['eligible_for_production'] is False