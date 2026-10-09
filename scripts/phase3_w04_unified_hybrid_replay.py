"""One-shot W04 cache-to-ByteTrack experiment for ALL recommended hybrid solutions.

Research only. No source-video decoding, no OpenVINO inference, no FLUID edits.
Replays one SHA-verified 291-frame raw-box cache through one original control
and three hybrid suppression policies with actual independent ByteTrack.
Rejects the entire experiment unless the replayed original control reproduces
original frozen pixel, identity, and class-diagnostic scores exactly.

Original 25 policy tracks and production Geo-trax/T000 remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any

import numpy as np

from streetlab_phase3.video.continuous_box_tracking_lab import (
    as_tracker_detections, cached_window_scorer, _sha)
from streetlab_phase3.video.holdout_validation import verify_batch
from streetlab_phase3.video.integrated_box_lab import RawBox, merge_boxes
from streetlab_phase3.video.sahi_tracker_trial import rows_from_tracks
from streetlab_phase3.video.w04_hybrid_rare_suppression import merge_rare_preserving
from streetlab_phase3.video.w04_class_evidence_stream import ClassEvidenceStream

STATUS = 'W04_FROZEN_291FRAME_UNIFIED_HYBRID_BYTETRACK_REPLAY_NOT_PRODUCTION'
FIRST, START, END = 10660, 10750, 10950
ORIGINAL_CONTROL = 'hard_nms_ios_0.30'
EXPERIMENTS = (
    ('control_ios030', 'original', ORIGINAL_CONTROL),
    ('rare_iou05', 'hybrid', 'rare_iou05'),
    ('rare_center_gate', 'hybrid', 'rare_center_gate'),
    ('rare_cross_tile_center_gate', 'hybrid', 'rare_cross_tile_center_gate'),
)
FROZEN_BASELINE = ('hard_nms_ios_0.30', 'hard_nms_iou_0.50', 'raw_unmerged')


def load_cached_frames(directory: Path, batch: dict) -> dict[int, list[RawBox]]:
    path = directory / batch['raw_pre_global_merge_boxes_file']
    if path.name != 'original_pre_global_merge_boxes.jsonl':
        raise ValueError('Unexpected frozen raw box filename')
    if _sha(path) != batch['raw_pre_global_merge_boxes_sha256']:
        raise ValueError('Cached detector source SHA mismatch')
    frames = {x: [] for x in range(FIRST, END+1)}
    sources = set()
    total = 0
    with path.open(encoding='utf-8') as file:
        for line in file:
            b = RawBox(**json.loads(line))
            if b.frame not in frames:
                raise ValueError('Raw detector frame outside original W04')
            k = (b.frame, b.source_index)
            if k in sources:
                raise ValueError('Duplicate source prediction in a frame')
            sources.add(k)
            frames[b.frame].append(b)
            total += 1
    if total != 21104 or total != batch['raw_tile_boxes']:
        raise ValueError('W04 raw detector count changed')
    return frames


def control_merger(frame_raw: list[RawBox]):
    return merge_boxes(frame_raw, policy='hard_nms',
                       threshold=.30, metric='ios', score_floor=.15)


def merge_for_experiment(raw: list[RawBox], mode: str):
    if mode == 'control_ios030':
        return control_merger(raw)
    if mode not in {x[0] for x in EXPERIMENTS[1:]}:
        raise ValueError('Unknown preregistered hybrid mode')
    return merge_rare_preserving(raw, mode=mode)


def frozen_score_comparison(actual: dict, expected: dict) -> dict:
    """Strict control parity; do not score hybrids if the tracker runtime drifts."""
    checks = {
        'predicted_points': (actual['pixel'].get('predicted_points'),
                             expected['pixel'].get('predicted_points')),
        'matched_points': (actual['pixel'].get('matched_points'),
                           expected['pixel'].get('matched_points')),
        'truth_points': (actual['pixel'].get('truth_points'),
                         expected['pixel'].get('truth_points')),
        'predicted_tracks': (actual['identity'].get('predicted_tracks'),
                             expected['identity'].get('predicted_tracks')),
        'contiguous_id_switches': (
            actual['identity'].get('total_contiguous_id_switches'),
            expected['identity'].get('total_contiguous_id_switches')),
        'fragmented_truth_tracks_fraction': (
            actual['identity'].get('fraction_truth_tracks_fragmented'),
            expected['identity'].get('fraction_truth_tracks_fragmented')),
        'correct_class_matches': (
            actual['class_diagnostics'].get('correct_class_matches'),
            expected['class_diagnostics'].get('correct_class_matches')),
    }
    for cls in ('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE'):
        checks[cls+'_correct_matches'] = (
            actual['class_diagnostics']['by_class'][cls].get('correct_class_matches'),
            expected['class_diagnostics']['by_class'][cls].get('correct_class_matches'))
    failures = {
        key:{'replay':a,'original':b}
        for key,(a,b) in checks.items()
        if a!=b
    }
    if failures:
        raise ValueError('Original W04 control did NOT reproduce frozen metrics: '
                         +json.dumps(failures,sort_keys=True))
    return {'status':'FROZEN_CONTROL_EXACT_METRICS_REPRODUCED',
            'checks':len(checks),'successful':True}


def development_safety_gate(experiment: dict, control: dict) -> dict:
    """Conservative development filter; NEVER a holdout or production gate.

    Fail closed on any known correct-class rare-vehicle regression, added
    identity switches, large fragmentation, or a >=2pp precision loss.
    """
    cls=experiment['class_diagnostics']['by_class']
    baseline=control['class_diagnostics']['by_class']
    issues=[]
    def numeric(x):
        if not isinstance(x,(int,float)) or not math.isfinite(x):
            raise ValueError('Nonfinite W04 policy metric')
        return float(x)
    if numeric(experiment['pixel']['point_precision']) < numeric(
            control['pixel']['point_precision'])-.02:
        issues.append('point precision fell more than 2 percentage points')
    if numeric(experiment['pixel']['point_recall']) < numeric(
            control['pixel']['point_recall'])-.01:
        issues.append('point recall fell more than 1 percentage point')
    for label in ('MOTORCYCLE','HEAVY_VEHICLE'):
        if numeric(cls[label]['correct_class_recall']) < numeric(
                baseline[label]['correct_class_recall']):
            issues.append(label+' correct-class recall regressed')
    if numeric(cls['CAR']['correct_class_recall']) < numeric(
            baseline['CAR']['correct_class_recall'])-.01:
        issues.append('CAR correct-class recall regressed by >1pp')
    if experiment['identity']['total_contiguous_id_switches'] > (
            control['identity']['total_contiguous_id_switches']):
        issues.append('additional contiguous FLUID-matched ID switches')
    if numeric(experiment['identity']['fraction_truth_tracks_fragmented']) > numeric(
            control['identity']['fraction_truth_tracks_fragmented'])+.02:
        issues.append('truth-track fragmentation rose by >2pp')
    return {'development_gate_passed':not issues,
            'violations':issues,
            'production_eligible':False,
            'no_T000_comparison':True,
            'physical_truth_not_verified':True}


def _safe_delta(new: float | None, old: float | None):
    if new is None or old is None:
        return None
    return new-old


def compare_to_original(current: dict, original: dict) -> dict:
    curr_cls=current['class_diagnostics']['by_class']
    old_cls=original['class_diagnostics']['by_class']
    return {
        'point_precision_delta_pp':100*_safe_delta(
            current['pixel']['point_precision'],original['pixel']['point_precision']),
        'point_recall_delta_pp':100*_safe_delta(
            current['pixel']['point_recall'],original['pixel']['point_recall']),
        'identity_switches_delta':(
            current['identity']['total_contiguous_id_switches']
            -original['identity']['total_contiguous_id_switches']),
        'fragmented_truth_tracks_delta':(
            current['identity']['truth_tracks_with_2_predicted_ids']
            +current['identity']['truth_tracks_with_3plus_predicted_ids']
            -original['identity']['truth_tracks_with_2_predicted_ids']
            -original['identity']['truth_tracks_with_3plus_predicted_ids']),
        'per_class_correct_recall_delta_pp':{
            cls:(None if old_cls[cls]['correct_class_recall'] is None
                     or curr_cls[cls]['correct_class_recall'] is None else
                 100*(curr_cls[cls]['correct_class_recall']
                      -old_cls[cls]['correct_class_recall']))
            for cls in ('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE')
        },
        'per_class_correct_match_count_delta':{
            cls:curr_cls[cls]['correct_class_matches']-old_cls[cls]['correct_class_matches']
            for cls in ('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE')
        },
    }


def run_all(batch_dir: Path, output_dir: Path, *,
            tracker_factory=None, detection_class=None) -> dict[str, Any]:
    """Atomic publication of one unified original-control + 3-hybrid experiment."""
    if output_dir.exists():
        raise FileExistsError('Cannot overwrite immutable W04 experiment')
    verified = verify_batch(batch_dir,verify_originals=True)
    batch = verified['report']
    origin = verified['provenance']
    if (verified['original_source_sha256_verified'] is not True
        or batch['candidate_count']!=25
        or batch['first_decoded_frame']!=FIRST
        or batch['evaluated_first_frame']!=START
        or batch['evaluated_last_frame']!=END
        or batch['processed_frames']!=291
        or batch['same_window_T000'] is not None
        or origin['video_frame_window']!=[START,END]):
        raise ValueError('Batch provenance differs from true W04 201-frame cohort')
    originals = {p['name']:p for p in batch['policies']}
    if any(n not in originals for n in FROZEN_BASELINE):
        raise ValueError('One of the immutable source-control policies is missing')
    fps = origin['source_fps_user_supplied']
    if not isinstance(fps,(float,int)) or not math.isfinite(fps) or fps<=0:
        raise ValueError('Invalid original source FPS')
    truth = Path(origin['fluid_path'])
    if not truth.is_file() or _sha(truth)!=origin['fluid_sha256']:
        raise ValueError('Original untouched FLUID source file unavailable or modified')
    frames = load_cached_frames(batch_dir,batch)
    if tracker_factory is None:
        from trackers import ByteTrackTracker
        tracker_factory=ByteTrackTracker
    if detection_class is None:
        from supervision import Detections
        detection_class=Detections
    scorer=cached_window_scorer(truth,START,END)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'.stage-',
                                dir=output_dir.parent))
    results=[]
    control_check=None
    try:
        for label,kind,mode in EXPERIMENTS:
            tracker=tracker_factory(
                track_activation_threshold=.20,
                high_conf_det_threshold=.15,
                lost_track_buffer=45,
                minimum_consecutive_frames=2,frame_rate=fps)
            stabilizer=ClassEvidenceStream(consecutive_required=2,min_box_iou=.5)
            target=stage/(label+'.txt')
            sidecar=stage/(label+'.class_evidence.csv')
            confidence_counter=Counter()
            box_counter=Counter()
            post_times=[]
            track_times=[]
            confirmed_rows=0
            total_class_changes=0
            supported_proposals=0
            emitted_proposals=0
            class_rows=0
            last_class={}
            with (target.open('x',encoding='utf-8',newline='') as tf,
                  sidecar.open('x',encoding='utf-8',newline='') as sf):
                writer=csv.writer(tf)
                evidence_writer=csv.writer(sf)
                evidence_writer.writerow([
                    'frame','tracker_id','native_class','stable_evidence_class',
                    'raw_supports_evidence_class','raw_supports_native_class',
                    'conflicting_raw_class_hypotheses','label_differs_from_native',
                    'raw_classes_overlapping_track'
                ])
                for frame in range(FIRST,END+1):
                    raw=frames[frame]
                    t0=time.perf_counter()
                    merged=merge_for_experiment(raw,label)
                    post_times.append(time.perf_counter()-t0)
                    det=as_tracker_detections(merged,detection_class)
                    t1=time.perf_counter()
                    tracked=tracker.update(det)
                    track_times.append(time.perf_counter()-t1)
                    if frame<START:
                        continue
                    for b in merged:
                        if b.vehicle_class in ('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE'):
                            box_counter[b.vehicle_class]+=1
                            if b.confidence>=.5:
                                confidence_counter[b.vehicle_class]+=1
                    if (len(np.asarray(tracked.xyxy))==0
                        and getattr(tracked,'tracker_id',None) is None):
                        rows=[]
                    else:
                        rows,_=rows_from_tracks(tracked,frame)
                    if any(len(r)!=14 or int(r[0])!=frame for r in rows):
                        raise ValueError('Invalid frozen Geo-trax row')
                    writer.writerows(rows)
                    confirmed_rows+=len(rows)
                    for row in rows:
                        tid=int(row[1])
                        native_id=int(row[10])
                        names=('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE')
                        native=names[native_id]
                        cx,cy,w,h=map(float,row[2:6])
                        bbox=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
                        entry=stabilizer.observe(
                            tracker_id=tid,frame=frame,
                            instantaneous_class=native,track_box=bbox,
                            raw_detections=(vars(b) for b in raw))
                        if tid in last_class and last_class[tid]!=native:
                            total_class_changes+=1
                        last_class[tid]=native
                        class_rows+=1
                        if entry.provisional_class_change:
                            emitted_proposals+=1
                            if entry.stable_class_supported_by_raw:
                                supported_proposals+=1
                        evidence_writer.writerow([
                            frame,tid,native,entry.stable_class,
                            int(entry.stable_class_supported_by_raw),
                            int(native in entry.detector_classes_at_track_box),
                            int(entry.conflicting_detector_hypotheses),
                            int(entry.provisional_class_change),
                            '|'.join(entry.detector_classes_at_track_box)
                        ])
            pixel,identity=scorer(tracks=target,fluid_tracks=truth,
                                  max_distance_px=50.,
                                  start_frame=START,end_frame=END)
            classes=scorer.class_report(target)
            if (pixel.get('frame_offset')!=1
                or pixel.get('max_distance_px')!=50
                or identity.get('frame_offset')!=1
                or identity.get('max_distance_px')!=50
                or pixel.get('truth_points')!=originals[ORIGINAL_CONTROL]['pixel']['truth_points']):
                raise ValueError('Frozen +1 / 50 pixel cohort changed')
            result={
                'name':label,'kind':kind,'mode':mode,
                'tracks':target.name,'tracks_sha256':_sha(target),
                'class_evidence':sidecar.name,'class_evidence_sha256':_sha(sidecar),
                'evaluation_confirmed_rows':confirmed_rows,
                'evaluation_boxes_by_class':dict(box_counter),
                'evaluation_high_conf_boxes_by_class':dict(confidence_counter),
                'postprocess_median_seconds':statistics.median(post_times),
                'tracker_median_seconds':statistics.median(track_times),
                'native_class_change_events':total_class_changes,
                'sidecar_evidence_rows':class_rows,
                'proposed_differences_from_native':emitted_proposals,
                'proposed_differences_raw_supported':supported_proposals,
                'pixel':pixel,'identity':identity,'class_diagnostics':classes,
                'development_safety_gate': development_safety_gate(
                    {'pixel':pixel,'identity':identity,'class_diagnostics':classes},
                    originals[ORIGINAL_CONTROL]),
                'delta_vs_frozen_ios030':compare_to_original(
                    {'pixel':pixel,'identity':identity,'class_diagnostics':classes},
                    originals[ORIGINAL_CONTROL]),
                'eligible_for_production':False,
            }
            if kind=='original':
                control_check=frozen_score_comparison(result,originals[ORIGINAL_CONTROL])
            else:
                if control_check is None:
                    raise ValueError('Original tracker control must pass BEFORE hybrids')
            (stage/(label+'.score.json')).write_text(
                json.dumps(result,indent=2),encoding='utf-8')
            results.append(result)
        report={
            'status':STATUS,'eligible_for_production':False,
            'source_original_batch_sha256':verified['batch_sha256'],
            'source_video_sha256_verified':origin['video_sha256'],
            'source_fluid_sha256_verified':origin['fluid_sha256'],
            'source_original_raw_sha256':batch['raw_pre_global_merge_boxes_sha256'],
            'source_frames':[FIRST,END],'evaluation_frames':[START,END],
            'new_detector_inference_performed':False,
            'source_video_frames_decoded':False,
            'original_control_reproduction':control_check,
            'original_25_policy_control_metrics':{
                n:originals[n] for n in FROZEN_BASELINE
            },
            'experimental_results':results,
            'same_window_historical_T000_available':False,
            'limitations':(
                'W04 development footage; raw boxes have local NMS already. '
                'A retained detector box is not proof of a second physical '
                'vehicle; BUS FLUID truth count is zero. Auxiliary class '
                'labels are not human-grounded accuracy improvements. '
                'Unseen footage, replacement T000 baseline, independent '
                'physical review and CPU p95 are still required.'
            ),
        }
        (stage/'unified_hybrid_report.json').write_text(
            json.dumps(report,indent=2),encoding='utf-8')
        if output_dir.exists():
            raise FileExistsError('New immutable output unexpectedly exists')
        os.replace(stage,output_dir)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    report=run_all(args.batch_dir,args.output_dir)
    print(json.dumps({
        'status':report['status'],
        'evaluated_frames':END-START+1,
        'control_reproduction':report['original_control_reproduction'],
        'modes':[x['name'] for x in report['experimental_results']],
        'output_directory':str(args.output_dir),
        'production_promotion':False,
    },indent=2))


if __name__=='__main__':
    main()
