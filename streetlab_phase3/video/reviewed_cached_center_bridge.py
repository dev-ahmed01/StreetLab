"""Link accepted independent W04 case boxes to unchanged detector center caches.

No box IoU, physical precision, tracker claims, FLUID relabeling or promotions.
This compares only targeted case-level localization against existing predictions.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .manual_review_protocol import CONSENSUS_STATUS, _hash, validate_pack

MODES = ('openvino', 'pytorch')
CANONICAL = ('CAR','BUS','HEAVY_VEHICLE','MOTORCYCLE')
STATUS = 'W04_ADJUDICATED_CASE_CENTER_COVERAGE_NOT_PHYSICAL_PRECISION'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_reviewed_centers(*, bundle: Path, review_dir: Path,
                            consensus_file: Path) -> dict[str, Any]:
    manifest = validate_pack(review_dir)
    consensus_bytes = consensus_file.read_bytes()
    consensus = json.loads(consensus_bytes)
    if (consensus.get('status') != CONSENSUS_STATUS
            or consensus.get('eligible_for_promotion') is not False
            or consensus.get('review_manifest_sha256') != _hash((review_dir/'manifest.json').read_bytes())
            or consensus.get('case_count') != manifest['case_count']):
        raise ValueError('Independent case consensus provenance mismatch')
    answers = {c['case_id']: c for c in consensus['cases']}
    cases = manifest['cases']
    if len(answers) != len(cases) or set(answers) != {c['case_id'] for c in cases}:
        raise ValueError('Consensus case IDs differ from blind package')
    by_mode = {}
    scores = {}
    with zipfile.ZipFile(bundle) as archive:
        source_manifest_bytes = archive.read('manifest.json')
        src = json.loads(source_manifest_bytes)
        if (src.get('status') != 'UNPROMOTED_W04_SELF_CONTAINED_RESEARCH_INPUT'
                or _digest(source_manifest_bytes) != manifest['source_bundle_manifest_sha256']):
            raise ValueError('Original FLUID/frame evidence bundle differs from reviewer pack')
        for backend in MODES:
            member = f'audits/{backend}/sliced_detections.csv'
            raw = archive.read(member)
            if _digest(raw) != src['files'][member]:
                raise ValueError(f'Frozen {backend} sliced detection checksum mismatch')
            reportmember = f'audits/{backend}/report.json'
            reportbytes = archive.read(reportmember)
            if _digest(reportbytes) != src['files'][reportmember]:
                raise ValueError(f'Frozen {backend} report checksum mismatch')
            report = json.loads(reportbytes)
            if (report.get('sample_frames') != src['frames']
                or report.get('model_sha256') != src['original_model_sha256']
                or (backend == 'openvino' and report.get('runtime_model_sha256')
                    != src['openvino_export_sha256'])):
                raise ValueError('Detector sample frames or frozen model provenance differ from review source')
            entries = defaultdict(list)
            class_counts = Counter()
            count = 0
            with io.StringIO(raw.decode('utf-8-sig')) as f:
                reader = csv.DictReader(f)
                if not {'frame','x_px','y_px','vehicle_class','confidence'}.issubset(reader.fieldnames or ()):
                    raise ValueError('Unsupported detection cache format')
                for row in reader:
                    frame = int(row['frame'])
                    klass = row['vehicle_class']
                    x,y,confidence = (float(row[k]) for k in ('x_px','y_px','confidence'))
                    if (frame not in src['frames'] or klass not in CANONICAL
                        or not all(math.isfinite(v) for v in (x,y,confidence))
                        or not 0 <= x <= 3840 or not 0 <= y <= 2160
                        or not 0 <= confidence <= 1):
                        raise ValueError('Invalid frozen detector cache entry')
                    entries[frame].append({'class': klass, 'x_px': x,'y_px': y,
                                           'confidence': confidence})
                    class_counts[klass] += 1
                    count += 1
            sliced = report['sliced']
            if (count != sliced['predicted_points'] or
                any(class_counts[c] != sliced['per_class'][c]['predicted'] for c in CANONICAL)):
                raise ValueError('Frozen detector class counts differ from archived report')
            by_mode[backend] = entries
            scores[backend] = {
                'cached_detections': count, 'official_matched': sliced['matched_points'],
                'official_truth': sliced['truth_points'],
                'reference_not_recomputed': True,
            }
    results=[]
    summaries = {b:Counter() for b in MODES}
    for case in cases:
        answer = answers[case['case_id']]
        decision = answer['decision']
        if decision not in ('AGREED_PRESENT','AGREED_ABSENT','UNRESOLVED'):
            raise ValueError('Unknown review outcome')
        item = {'case_id': case['case_id'], 'video_frame':case['source_frame'],
                'decision':decision,'reviewed_class':answer['physical_class']}
        if decision == 'UNRESOLVED':
            results.append(item)
            continue
        if decision == 'AGREED_ABSENT':
            # An agreed absence is an anchor-level decision. Not used as a
            # full-frame FP/precision denominator.
            results.append(item)
            continue
        box = answer['consensus_bbox_crop']
        if (not isinstance(box,list) or len(box)!=4 or
            not all(type(v) in (int,float) and math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= case['crop_size']
            or not 0 <= box[1] < box[3] <= case['crop_size']):
            raise ValueError('Malformed accepted manual box geometry')
        ox,oy=case['crop_origin']
        abs_box=(box[0]+ox,box[1]+oy,box[2]+ox,box[3]+oy)
        item['consensus_bbox_absolute'] = list(abs_box)
        item['class_supported_by_detector'] = answer['physical_class'] in CANONICAL
        item['models']={}
        for backend in MODES:
            local = [d for d in by_mode[backend][case['source_frame']]
                     if abs_box[0] <= d['x_px'] <= abs_box[2]
                     and abs_box[1] <= d['y_px'] <= abs_box[3]]
            same = [d for d in local if d['class']==answer['physical_class']]
            local.sort(key=lambda d: -d['confidence'])
            stat = {'center_inside_review_box_count':len(local),
                    'correct_class_center_inside_review_box_count':len(same),
                    'localized_any_class':bool(local),
                    'localized_compatible_class':bool(same),
                    'classes_of_centers_inside_review_box':sorted({d['class'] for d in local}),
                    'max_confidence_center_inside_review_box':local[0]['confidence'] if local else None}
            item['models'][backend]=stat
            summaries[backend]['agreed_present_cases']+=1
            if local:summaries[backend]['localized_any_class_cases']+=1
            if same:summaries[backend]['localized_compatible_class_cases']+=1
            if len(local)>1:summaries[backend]['multiple_centers_in_manual_box_cases']+=1
            if not item['class_supported_by_detector']:
                summaries[backend]['physical_class_not_in_detector_ontology']+=1
        results.append(item)
    if sum(x['decision']=='AGREED_PRESENT' for x in results) != 13:
        # This audit is frozen to the reviewed W04 case-level cohort;
        # changing reviewer outcomes requires a new, versioned experiment.
        raise ValueError('Unexpected accepted W04 dual-review cohort')
    return {
        'status': STATUS, 'eligible_for_promotion':False,
        'frozen_source_bundle_manifest_sha256':manifest['source_bundle_manifest_sha256'],
        'review_manifest_sha256':consensus['review_manifest_sha256'],
        'consensus_sha256':_digest(consensus_bytes),
        'original_cached_detector_scores':scores,
        'case_level_center_coverage_by_backend':{b:dict(summaries[b]) for b in MODES},
        'cases':results,
        'physical_precision':None,'whole_frame_recall':None,'track_identity':None,
        'limitations':('15 targeted W04 case annotations, of which 13 have independent '
            'two-person class and box consensus, do not provide a random physical '
            'precision sample or exhaustive truth. Cache rows contain only centers; '
            'center-inside-manual-box is NOT model bounding-box IoU or track identity. '
            'AUTO_RICKSHAW is outside original four detector classes. Unresolved cases '
            'remain excluded. This does not change FLUID, model checkpoints, or frozen scores.'),
    }