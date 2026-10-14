"""Read-only discovery of possible original Geo-trax/T000 track files for W04.

This is an inventory, NOT a provenance validation or a T000 score comparison.
It does not create, convert, relabel or alter any baseline track file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

START, END = 10750, 10950

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def examine(path: Path) -> dict:
    stats = {'path': str(path.resolve()), 'bytes': path.stat().st_size,
             'candidate_only_not_verified_T000': True,
             'valid_geotrax_14_column_layout': True,
             'row_count': 0, 'min_frame': None, 'max_frame': None,
             'window_rows': 0, 'window_frames_present': 0,
             'frame_coverage_complete': False, 'distinct_track_ids_in_window': 0,
             'duplicate_frame_id_pairs_in_window': 0}
    window_frames = set()
    window_ids = set()
    window_seen = set()
    try:
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            for row in csv.reader(f):
                if len(row) != 14:
                    stats['valid_geotrax_14_column_layout'] = False
                    stats['reason'] = f'Row length {len(row)} is not the frozen 14-column layout'
                    break
                try:
                    frame, tid = int(row[0]), int(row[1])
                except (ValueError, TypeError):
                    stats['valid_geotrax_14_column_layout'] = False
                    stats['reason'] = 'First columns not integer source-frame and tracker ID'
                    break
                stats['row_count'] += 1
                stats['min_frame'] = frame if stats['min_frame'] is None else min(stats['min_frame'],frame)
                stats['max_frame'] = frame if stats['max_frame'] is None else max(stats['max_frame'],frame)
                if START <= frame <= END:
                    stats['window_rows'] += 1
                    window_frames.add(frame)
                    window_ids.add(tid)
                    if (frame,tid) in window_seen:
                        stats['duplicate_frame_id_pairs_in_window'] += 1
                    window_seen.add((frame,tid))
    except (OSError,UnicodeError) as exc:
        stats.update(valid_geotrax_14_column_layout=False,reason=f'Cannot read track candidate: {exc}')
    stats['window_frames_present'] = len(window_frames)
    stats['frame_coverage_complete'] = len(window_frames) == END-START+1
    stats['distinct_track_ids_in_window'] = len(window_ids)
    if stats['valid_geotrax_14_column_layout'] and stats['frame_coverage_complete'] and stats['duplicate_frame_id_pairs_in_window']==0:
        stats['sha256'] = sha256_file(path)
    return stats


def scan(roots: list[Path],max_files=100) -> dict:
    candidates=[]
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob('*.txt'))
        for file in files:
            if 'sahi_detector_trials' in file.parts or not file.is_file():
                continue
            if len(candidates) >= max_files:
                raise ValueError('Candidate file count exceeds bound; narrow roots')
            candidates.append(examine(file))
    ranked=sorted(candidates,key=lambda x:(not (x['valid_geotrax_14_column_layout'] and x['frame_coverage_complete']),-x['window_frames_present'],x['path']))
    return {'status':'T000_UNVERIFIED_FILE_INVENTORY_ONLY','eligible_for_production':False,
            'w04_source_window':[START,END],'w04_fluid_window':[START+1,END+1],
            'scanned_roots':[str(x) for x in roots],'candidate_file_count':len(candidates),
            'complete_14col_window_candidate_count':sum(r['valid_geotrax_14_column_layout'] and r['frame_coverage_complete'] for r in candidates),
            'candidates':ranked,
            'limitations':'Complete frame coverage does not establish T000 provenance, original model/config, actual source video identity, spatial coordinate parity, or same-window scoring. No source files modified.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; refusing to overwrite baseline evidence inventory')
    result=scan(a.roots)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'candidates':result['candidate_file_count'],
      'complete_14col_window_candidates':result['complete_14col_window_candidate_count'],
      'report':str(a.output),'not_t000_verified':True},indent=2))


if __name__=='__main__':main()
