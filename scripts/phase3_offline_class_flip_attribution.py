"""Offline W04 class-flip evidence attribution from immutable Build 3 outputs.

Reads original *pre-global-merge* detector boxes and real ByteTrack exports.
No inference, truth relabeling, tracker modification, or production promotion.
The overlap test finds coincident detector hypotheses; it cannot establish which
physical vehicle an ID really represents or causally diagnose association.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

CLASS_IDS = {0: 'CAR', 1: 'BUS', 2: 'HEAVY_VEHICLE', 3: 'MOTORCYCLE'}
STATUS = 'W04_CACHED_RAW_BOX_CLASS_FLIP_DIAGNOSTIC_NOT_PRODUCTION'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def overlap_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = width * height
    denominator = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / denominator if denominator > 0 else 0.


class Evidence:
    def __init__(self, *, archive: Path | None, batch_dir: Path | None):
        if (archive is None) == (batch_dir is None):
            raise ValueError('Provide exactly one of --archive or --batch-dir')
        self.zip = zipfile.ZipFile(archive) if archive else None
        self.dir = batch_dir
        if self.zip:
            names = self.zip.namelist()
            if len(names) != len(set(names)) or any(n.startswith('/') or '..' in Path(n).parts for n in names):
                raise ValueError('Unsafe archive member paths')

    def read(self, name: str) -> bytes:
        if name.startswith('/') or '..' in Path(name).parts or not name:
            raise ValueError('Unsafe evidence path')
        return self.zip.read(name) if self.zip else (self.dir / name).read_bytes()

    def close(self):
        if self.zip:
            self.zip.close()


def read_track_rows(raw: bytes, policy: str) -> dict[int, list[tuple[int, str, tuple[float, ...], float]]]:
    tracks = defaultdict(list)
    observed = set()
    for row in csv.reader(io.StringIO(raw.decode('utf-8-sig'))):
        if len(row) != 14:
            raise ValueError(f'{policy}: tracking row must have 14 columns')
        frame, tid, cls_id = int(row[0]), int(row[1]), int(row[10])
        if cls_id not in CLASS_IDS or tid < 0 or (frame, tid) in observed:
            raise ValueError(f'{policy}: invalid class, ID, or duplicate frame-ID')
        observed.add((frame, tid))
        cx, cy, w, h, conf = (float(row[i]) for i in (2, 3, 4, 5, 11))
        if not all(math.isfinite(v) for v in (cx, cy, w, h, conf)) or w <= 0 or h <= 0 or not 0 <= conf <= 1:
            raise ValueError(f'{policy}: invalid track geometry/confidence')
        bbox = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
        tracks[tid].append((frame, CLASS_IDS[cls_id], bbox, conf))
    for track in tracks.values():
        track.sort(key=lambda v: v[0])
    return tracks


def raw_box_support(raw_by_frame: dict, frame: int, bbox: tuple[float, ...], min_iou: float) -> dict[str, dict]:
    evidence = {}
    for d in raw_by_frame.get(frame, []):
        box = (d['x1'], d['y1'], d['x2'], d['y2'])
        iou = overlap_iou(bbox, box)
        if iou < min_iou:
            continue
        label = d['vehicle_class']
        entry = evidence.setdefault(label, {'count': 0, 'max_iou': 0., 'max_confidence': 0.})
        entry['count'] += 1
        entry['max_iou'] = max(entry['max_iou'], iou)
        entry['max_confidence'] = max(entry['max_confidence'], float(d['confidence']))
    return evidence


def flip_events(policy: str, tracks: dict, raw_by_frame: dict, min_iou: float) -> tuple[list[dict], dict]:
    events = []
    for tid, track in sorted(tracks.items()):
        for prev, current in zip(track, track[1:]):
            prior_frame, prior_class, prior_box, _ = prev
            frame, current_class, box, conf = current
            if prior_class == current_class:
                continue
            evidence = raw_box_support(raw_by_frame, frame, box, min_iou)
            prior_evidence = raw_box_support(raw_by_frame, prior_frame, prior_box, min_iou)
            has_current, has_prior = current_class in evidence, prior_class in evidence
            kind = ('both_classes_in_current_raw' if has_current and has_prior else
                    'new_class_only_in_current_raw' if has_current else
                    'prior_class_only_in_current_raw' if has_prior else
                    'no_qualifying_raw_overlap')
            events.append({
                'policy': policy, 'tracker_id': tid, 'prior_frame': prior_frame,
                'frame': frame, 'gap_frames': frame - prior_frame,
                'prior_class': prior_class, 'current_class': current_class,
                'current_track_confidence': round(conf, 6),
                'classification': kind,
                'prior_class_on_prior_frame': prior_class in prior_evidence,
                'current_class_on_prior_frame': current_class in prior_evidence,
                'current_class_in_raw': has_current, 'prior_class_in_raw': has_prior,
                'prior_max_raw_confidence': round(evidence.get(prior_class, {}).get('max_confidence', 0.), 6),
                'current_max_raw_confidence': round(evidence.get(current_class, {}).get('max_confidence', 0.), 6),
                'current_max_raw_iou': round(evidence.get(current_class, {}).get('max_iou', 0.), 6),
                'raw_boxes_overlapping_current': sum(v['count'] for v in evidence.values()),
                'x1': round(box[0], 3), 'y1': round(box[1], 3),
                'x2': round(box[2], 3), 'y2': round(box[3], 3),
            })
    flags = Counter(e['classification'] for e in events)
    pairs = Counter(e['prior_class'] + ' → ' + e['current_class'] for e in events)
    ids_changed = len(set(e['tracker_id'] for e in events))
    return events, {
        'tracker_ids_total': len(tracks),
        'tracker_ids_with_class_change': ids_changed,
        'observed_class_change_events': len(events),
        'contiguous_class_change_events': sum(e['gap_frames'] == 1 for e in events),
        'classification_counts': dict(sorted(flags.items())),
        'pair_counts': dict(sorted(pairs.items())),
        'events_with_prior_class_present_in_previous_raw': sum(e['prior_class_on_prior_frame'] for e in events),
    }


def analyze(evidence: Evidence, *, min_iou: float = .5, integrity: dict | None = None):
    if not (0. < min_iou <= 1.):
        raise ValueError('min_iou must be in (0,1]')
    report_bytes = evidence.read('batch_report.json')
    report = json.loads(report_bytes)
    if (report.get('status') != 'EXPERIMENTAL_CONTINUOUS_BOX_BYTETRACK_BENCHMARK_NOT_PRODUCTION'
        or report.get('eligible_for_promotion') is not False
        or report.get('tracking_updates_are_consecutive') is not True
        or report.get('first_decoded_frame') != 10660
        or report.get('evaluated_first_frame') != 10750
        or report.get('evaluated_last_frame') != 10950
        or report.get('candidate_count') != 25):
        raise ValueError('Unexpected W04 evidence cohort; do not silently change benchmark')
    if integrity is not None:
        if (integrity.get('status') != 'BUILD4_VERIFIED_BATCH_NOT_PRODUCTION'
            or integrity.get('original_source_sha256_verified') is not True
            or integrity.get('batch_report_sha256') != sha(report_bytes)
            or integrity.get('policy_count') != 25
            or integrity.get('evaluated_frames') != 201
            or integrity.get('has_same_window_baseline') is not False):
            raise ValueError('Build 4 original-source verification does not match batch')
    raw_bytes = evidence.read('original_pre_global_merge_boxes.jsonl')
    if sha(raw_bytes) != report['raw_pre_global_merge_boxes_sha256']:
        raise ValueError('Original pre-merge raw detector evidence SHA mismatch')
    raw_by_frame = defaultdict(list)
    count = 0
    for line in io.BytesIO(raw_bytes):
        d = json.loads(line)
        frame = d['frame']
        if not 10660 <= frame <= 10950:
            raise ValueError('Out-of-window raw detector frame')
        raw_by_frame[frame].append(d)
        count += 1
    if count != report['raw_tile_boxes']:
        raise ValueError('Raw detector box count differs from recorded manifest')
    rows, stats = [], {}
    if len(report['policies']) != 25 or len({p['name'] for p in report['policies']}) != 25:
        raise ValueError('Expected 25 distinct policies')
    for p in report['policies']:
        name = p['name']
        if p['tracks'] != name+'.txt':
            raise ValueError('Unexpected path for track evidence')
        track_bytes = evidence.read(p['tracks'])
        if sha(track_bytes) != p['tracks_sha256']:
            raise ValueError(f'Track evidence SHA mismatch: {name}')
        tracks = read_track_rows(track_bytes, name)
        events, st = flip_events(name, tracks, raw_by_frame, min_iou)
        if len(tracks) != p['pixel']['predicted_tracks']:
            raise ValueError(f'{name}: predicted track ID count mismatch')
        st['track_rows'] = sum(len(v) for v in tracks.values())
        if st['track_rows'] != p['evaluation_confirmed_rows']:
            raise ValueError(f'{name}: track row count mismatch')
        stats[name] = st
        rows.extend(events)
    rows.sort(key=lambda e: (e['policy'], e['frame'], e['tracker_id']))
    result = {
        'status': STATUS, 'eligible_for_production': False,
        'batch_report_sha256': sha(report_bytes),
        'raw_box_file_sha256': sha(raw_bytes),
        'original_source_sha256_verified_by_build4': integrity is not None,
        'raw_detector_boxes': count,
        'policy_count': 25,
        'minimum_raw_track_box_iou': min_iou,
        'policies': stats,
        'limitations': 'Same-video model-class support only; overlapping alternative raw boxes cannot distinguish tracker association error from model ambiguity; no physical-object proof, no T000 baseline, no unseen holdout. Does not modify tracks.',
    }
    return result, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument('--archive', type=Path)
    inp.add_argument('--batch-dir', type=Path)
    parser.add_argument('--integrity-report', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minimum-iou', type=float, default=.5)
    a = parser.parse_args(argv)
    if a.output_dir.exists():
        parser.error('Output path already exists; immutable result cannot be overwritten')
    integrity = json.loads(a.integrity_report.read_text(encoding='utf-8')) if a.integrity_report else None
    evidence = Evidence(archive=a.archive, batch_dir=a.batch_dir)
    try:
        result, events = analyze(evidence, min_iou=a.minimum_iou, integrity=integrity)
    finally:
        evidence.close()
    a.output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f'.{a.output_dir.name}.stage-', dir=a.output_dir.parent))
    try:
        (tmp/'class_flip_attribution_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        with (tmp/'class_flip_attribution_events.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(events[0]) if events else ['policy','tracker_id','prior_frame','frame','classification'])
            writer.writeheader()
            writer.writerows(events)
        if a.output_dir.exists():
            raise FileExistsError('Immutable experiment path appeared before publication')
        os.replace(tmp, a.output_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    lead = result['policies']['hard_nms_ios_0.30']
    print(json.dumps({'status':result['status'], 'verified_build4': result['original_source_sha256_verified_by_build4'],
      'policy_count':result['policy_count'], 'event_count':len(events),
      'leading_policy':lead,'output_dir':str(a.output_dir)},indent=2))


if __name__ == '__main__':
    main()
