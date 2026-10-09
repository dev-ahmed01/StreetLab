"""W04 controlled class-partitioned ByteTrack replay from immutable cached raw boxes.

- NO new OpenVINO inference or video decoding.
- Uses original frozen 25-policy postprocess geometry (selected 4 development policies).
- Four independent ByteTrack trackers per policy: a class label can never migrate
  across trackers. This *constructs* zero within-ID class flips and is NOT by
  itself evidence of better physical identity or vehicle recall.
- Uses original frozen +1-frame/50px class-agnostic pixel/identity scorers,
  with separate class-aware diagnostics, against unchanged original FLUID.
- All original 25 track outputs and reports remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from streetlab_phase3.video.continuous_box_tracking_lab import (
    _sha, as_tracker_detections, cached_window_scorer, policy_matrix)
from streetlab_phase3.video.holdout_validation import verify_batch
from streetlab_phase3.video.integrated_box_lab import RawBox, merge_boxes
from streetlab_phase3.video.sahi_tracker_trial import rows_from_tracks

STATUS = 'W04_CACHED_CLASS_PARTITIONED_ABLATION_NOT_PRODUCTION'
POLICIES = ('hard_nms_ios_0.30', 'hard_nms_ios_0.50',
            'hard_nms_iou_0.30', 'raw_unmerged')
CLASSES = tuple(range(4))
FIRST, START, END = 10660, 10750, 10950


def verified_raw_frames(batch_dir: Path, report: dict[str, Any]) -> dict[int, list[RawBox]]:
    """SHA is verified by Build 4; enforce complete cohort + box invariants."""
    raw_path = batch_dir / 'original_pre_global_merge_boxes.jsonl'
    if _sha(raw_path) != report['raw_pre_global_merge_boxes_sha256']:
        raise ValueError('Original pre-global-merge raw-box SHA mismatch')
    by_frame = {frame: [] for frame in range(FIRST, END+1)}
    total = 0
    source_keys: set[tuple[int, int]] = set()
    with raw_path.open(encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            frame = d.get('frame')
            if type(frame) is not int or frame not in by_frame:
                raise ValueError('Raw detector evidence outside frozen 291-frame interval')
            raw = RawBox(**d)
            k = (frame, raw.source_index)
            if k in source_keys:
                raise ValueError('Duplicate source_index within one raw detector frame')
            source_keys.add(k)
            by_frame[frame].append(raw)
            total += 1
    if total != report['raw_tile_boxes'] or total != 21104:
        raise ValueError('Cached raw detector box count differs from original')
    return by_frame


def encode_class_identity(class_id: int, local_tracker_id: int) -> int:
    """Deterministic globally unique IDs for independent per-class trackers."""
    if type(class_id) is not int or class_id not in CLASSES:
        raise ValueError('Unsupported frozen model class')
    tid = int(local_tracker_id)
    if tid < 0 or float(local_tracker_id) != float(tid):
        raise ValueError('Unconfirmed or noninteger ByteTrack ID')
    return tid*len(CLASSES)+class_id


def step_partitioned(trackers: dict[int, Any], merged: list[Any],
                     frame: int, detection_cls: Callable[..., Any],
                     converter: Callable[..., Any] = rows_from_tracks):
    """Update all four independent trackers, including on empty frames."""
    rows = []
    unconfirmed = 0
    selected = [x for x in merged if x.vehicle_class in
                ('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE')]
    from streetlab_phase3.video.continuous_box_tracking_lab import CANONICAL_IDS
    class_names = {value: key for key,value in CANONICAL_IDS.items()}
    for cls in CLASSES:
        detections = as_tracker_detections(
            [p for p in selected if p.vehicle_class == class_names[cls]], detection_cls)
        tracked = trackers[cls].update(detections)
        if len(tracked.xyxy) == 0 and getattr(tracked, 'tracker_id', None) is None:
            continue
        class_rows, not_confirmed = converter(tracked, frame)
        if not_confirmed < 0:
            raise ValueError('Invalid unconfirmed detections')
        unconfirmed += not_confirmed
        for row in class_rows:
            if len(row)!=14 or int(row[0])!=frame or int(row[10])!=cls:
                raise ValueError('Class tracker exported wrong class/frame/schema')
            new = list(row)
            new[1] = encode_class_identity(cls, new[1])
            rows.append(new)
    ids = [int(r[1]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Global class-scoped tracker ID collision in one frame')
    return rows, unconfirmed


def count_class_flip_events(track_file: Path) -> tuple[int, int]:
    last = {}
    flips = 0
    ids = set()
    with track_file.open(newline='',encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row)!=14:
                raise ValueError('Unexpected original 14-column track schema')
            track_id,cls = int(row[1]),int(row[10])
            if track_id in last and cls != last[track_id]:
                flips += 1
            last[track_id] = cls
            ids.add(track_id)
    return flips,len(ids)


def execute(batch_dir: Path, out_dir: Path, *, tracker_factory=None, detection_cls=None,
            source_evidence=None) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError('Immutable class-ablation output already exists')
    verified = verify_batch(batch_dir)
    original = verified['report']
    provenance = verified['provenance']
    if (not verified['original_source_sha256_verified'] or
        original.get('same_window_T000') is not None or
        original.get('first_decoded_frame') != FIRST or
        original.get('evaluated_first_frame') != START or
        original.get('evaluated_last_frame') != END or
        original.get('candidate_count') != 25):
        raise ValueError('Unsupported or unverified W04 original cohort')
    if provenance.get('video_frame_window') != [START,END]:
        raise ValueError('Source-video window differs from class ablation')
    truth = Path(provenance['fluid_path'])
    if not truth.is_file() or _sha(truth) != provenance['fluid_sha256']:
        raise ValueError('Original FLUID source missing or SHA changed')
    records = {p['name']:p for p in original['policies']}
    if not all(name in records for name in POLICIES):
        raise ValueError('Required unchanged comparison policy missing')
    raw = verified_raw_frames(batch_dir,original)
    if tracker_factory is None:
        from trackers import ByteTrackTracker
        tracker_factory = ByteTrackTracker
    if detection_cls is None:
        from supervision import Detections
        detection_cls = Detections
    fps = float(provenance['source_fps_user_supplied'])
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('Recorded original source FPS invalid')
    scorer = cached_window_scorer(truth,START,END)
    configs = policy_matrix(POLICIES)
    out_dir.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{out_dir.name}.stage-',dir=out_dir.parent))
    try:
        results=[]
        for cfg in configs:
            name=cfg['name']
            # Each mode has four wholly separate tracker instances, one per class.
            trackers = {cls:tracker_factory(
                track_activation_threshold=.20, high_conf_det_threshold=.15,
                lost_track_buffer=45, minimum_consecutive_frames=2,
                frame_rate=fps) for cls in CLASSES}
            track_file = stage / f'{name}_class_partitioned.txt'
            post_seconds=[]
            update_seconds=[]
            rows_written=0
            with track_file.open('x',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                for frame in range(FIRST,END+1):
                    begin=time.perf_counter()
                    merged=merge_boxes(raw[frame], **{k:v for k,v in cfg.items() if k!='name'})
                    post_seconds.append(time.perf_counter()-begin)
                    begin=time.perf_counter()
                    rows,ignored=step_partitioned(trackers,merged,frame,detection_cls)
                    update_seconds.append(time.perf_counter()-begin)
                    if frame>=START:
                        writer.writerows(rows)
                        rows_written+=len(rows)
            pixel,identity=scorer(tracks=track_file,fluid_tracks=truth,
                                  max_distance_px=50.,start_frame=START,end_frame=END)
            cls_report=scorer.class_report(track_file)
            if (pixel.get('frame_offset')!=1 or pixel.get('max_distance_px')!=50.
                or identity.get('frame_offset')!=1
                or identity.get('max_distance_px')!=50.):
                raise ValueError('Scoring contract drift')
            reference=records[name]
            for fld in ('truth_points','evaluation_frame_start','evaluation_frame_end'):
                if pixel.get(fld) != reference['pixel'].get(fld):
                    raise ValueError('Cached and original baseline scoring cohort differs')
            original_flips,original_ids=count_class_flip_events(batch_dir/reference['tracks'])
            replay_flips,replay_ids=count_class_flip_events(track_file)
            if replay_flips:
                raise ValueError('Class-scoped identity output changed classification')
            result={
                'name':name,'setting':cfg,'original_policy':reference,
                'class_partitioned_tracks':track_file.name,
                'class_partitioned_tracks_sha256':_sha(track_file),
                'new_confirmed_rows':rows_written,
                'new_tracker_ids':replay_ids,
                'original_tracker_ids':original_ids,
                'original_emitted_class_changes':original_flips,
                'class_partitioned_class_changes_by_construction':replay_flips,
                'postprocess_median_seconds':statistics.median(post_seconds),
                'per_frame_four_tracker_updates_median_seconds':statistics.median(update_seconds),
                'class_partitioned_pixel':pixel,
                'class_partitioned_identity':identity,
                'class_partitioned_class_diagnostics':cls_report,
                'eligible_for_production':False,
            }
            (stage/f'{name}_class_partitioned.score.json').write_text(
                json.dumps(result,indent=2),encoding='utf-8')
            results.append(result)
        summary={
            'status':STATUS,'eligible_for_promotion':False,'eligible_for_production':False,
            'original_build3_batch_sha256':verified['batch_sha256'],
            'original_raw_boxes_sha256':original['raw_pre_global_merge_boxes_sha256'],
            'original_video_sha256':provenance['video_sha256'],
            'original_fluid_sha256':provenance['fluid_sha256'],
            'source_interval':[FIRST,END],'evaluation_interval':[START,END],
            'real_video_decoding_or_openvino_inference_rerun':False,
            'original_source_sha256_verified':True,
            'mode':'four_independent_Bytetrack_trackers_per_policy',
            'class_identity_immutable_by_design':True,
            'same_window_T000_available':False,
            'results':results,
            'limitations':(
                'Class partition prevents label flips by design but may fragment one '
                'physical object when the detector changes class. Compare MOTORCYCLE '
                'and HEAVY_VEHICLE correct-class recall, FLUID identity switches and '
                'fragmentation before making any accuracy claim. Development W04 '
                'is tuned; no physical proof, unseen holdout, T000 or production promotion.'
            ),
        }
        (stage/'class_partition_ablation_summary.json').write_text(
            json.dumps(summary,indent=2),encoding='utf-8')
        if out_dir.exists():
            raise FileExistsError('Output appeared while writing experiment')
        os.replace(stage,out_dir)
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-dir',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    args=ap.parse_args(argv)
    report=execute(args.batch_dir,args.output_dir)
    print(json.dumps({'status':report['status'],'policy_count':len(report['results']),
                      'original_sha':report['original_build3_batch_sha256'],
                      'output':str(args.output_dir),
                      'eligible_for_production':False},indent=2))


if __name__=='__main__':
    main()
