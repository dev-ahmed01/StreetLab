"""Blind two-reviewer W04 object-presence study, explicitly NOT new FLUID truth.

No model calls, automatic labeling, or edits to source FLUID/Geo-trax.
Reviewed anchors are selected deliberately (not a representative random sample).
PRESENT requires an independently drawn crop-relative bounding box.
Disagreement and UNKNOWN remain unresolved; they are never consensus labels.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from streetlab_phase3.video.premerge_box_capture import bundle_data

STATUS = 'W04_BLIND_CASE_REVIEW_NOT_OFFICIAL_GROUND_TRUTH'
CONSENSUS_STATUS = 'W04_DUAL_HUMAN_CONSENSUS_CASE_LEVEL_ONLY'
CLASSES = ('CAR', 'BUS', 'HEAVY_VEHICLE', 'MOTORCYCLE', 'PEDESTRIAN',
           'BICYCLE', 'AUTO_RICKSHAW', 'OTHER', 'UNKNOWN')
PRESENCE = ('PRESENT', 'ABSENT', 'UNCERTAIN')
FIELDS = ('case_id', 'reviewer_id', 'presence', 'physical_class',
          'x1', 'y1', 'x2', 'y2', 'note')


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_csv(path: Path, cases: list[dict[str, Any]], reviewer_id: str) -> None:
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for case in cases:
            w.writerow({'case_id': case['case_id'], 'reviewer_id': reviewer_id,
                        'presence': 'UNREVIEWED', 'physical_class': '',
                        'x1': '', 'y1': '', 'x2': '', 'y2': '', 'note': ''})


def generate_review_pack(bundle: Path, output_dir: Path, *, crop_size: int = 480,
                         case_limit: int = 15) -> dict[str, Any]:
    """Prepare blind local JPGs plus two empty, independent review sheets.

    Do not reveal model prediction type or FLUID label to human reviewers.
    Bundle origin and frame JPEG hashes are recorded for later validation.
    """
    if output_dir.exists():
        raise FileExistsError(f'Review pack output exists: {output_dir}')
    if not 256 <= crop_size <= 800 or not 1 <= case_limit <= 50:
        raise ValueError('Unsupported crop size or case limit')
    from PIL import Image, ImageDraw
    frames, _truth, provenance = bundle_data(bundle)
    with zipfile.ZipFile(bundle) as z:
        path = 'previous/W04_raw_fluid_label_nearest15_01.json'
        if path not in z.namelist():
            raise ValueError('Raw FLUID review case index missing from verified bundle')
        sidecar_bytes = z.read(path)
        sidecar = json.loads(sidecar_bytes)
        if (sidecar.get('status') != 'W04_RAW_FLUID_LABEL_ONTOLOGY_REVIEW_NOT_RELABELED'
            or sidecar.get('eligible_for_promotion') is not False):
            raise ValueError('Expected unpromoted W04 case index')
        source = sidecar.get('cases', [])
        if len(source) < case_limit or sidecar.get('case_count') != len(source):
            raise ValueError('Requested cases absent from source evidence')
        # The input index was pre-selected during visual review. Preserve the
        # predefined ordering and selection rather than cherry-pick by result.
        selected = source[:case_limit]
        task_rows=[]
        staged_parent=output_dir.parent
        staged_parent.mkdir(parents=True, exist_ok=True)
        temp=Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.tmp-',dir=staged_parent))
        try:
            (temp/'images').mkdir()
            frame_cache={}
            for i, src in enumerate(selected, 1):
                frame=src['source_frame']
                if type(frame) is not int or frame not in frames:
                    raise ValueError('Case frame outside hashed W04 evidence')
                x,y=float(src['selected_x_px']),float(src['selected_y_px'])
                if not (math.isfinite(x) and math.isfinite(y)
                    and 0 <= x < 3840 and 0 <= y < 2160):
                    raise ValueError('Invalid review anchor')
                if frame not in frame_cache:
                    original=z.read(f'frames/frame_{frame:06d}.jpg')
                    pic=Image.open(io.BytesIO(original)).convert('RGB')
                    if pic.size != (3840, 2160):
                        raise ValueError('W04 source JPEG dimensions changed')
                    frame_cache[frame] = (pic, _hash(original))
                img, frame_sha = frame_cache[frame]
                x0=max(0,min(round(x)-crop_size//2,3840-crop_size))
                y0=max(0,min(round(y)-crop_size//2,2160-crop_size))
                raw=img.crop((x0,y0,x0+crop_size,y0+crop_size))
                marked=raw.copy()
                draw=ImageDraw.Draw(marked)
                cx,cy=x-x0,y-y0
                draw.ellipse((cx-12,cy-12,cx+12,cy+12),outline='#e83ca8',width=3)
                # side-by-side; original raw on left, marker context on right
                combined=Image.new('RGB',(crop_size*2,crop_size),'white')
                combined.paste(raw,(0,0));combined.paste(marked,(crop_size,0))
                filename=f'case_{i:02d}.jpg'
                target=temp/'images'/filename
                combined.save(target,quality=94)
                cid=_hash(f"{provenance['manifest_sha256']}|{_hash(sidecar_bytes)}|{i}|{frame}|{x:.6f}|{y:.6f}".encode())[:20]
                task_rows.append({'case_id': cid, 'source_frame':frame,
                                  'image_file':f'images/{filename}',
                                  'image_sha256':_hash(target.read_bytes()),
                                  'source_frame_jpeg_sha256':frame_sha,
                                  'crop_origin':[x0,y0], 'crop_size':crop_size,
                                  'anchor_in_crop':[round(cx,4),round(cy,4)]})
            manifest={
                'status':STATUS,'eligible_for_promotion':False,
                'source_bundle_manifest_sha256':provenance['manifest_sha256'],
                'source_cases_sha256':_hash(sidecar_bytes),
                'case_count':len(task_rows),'cases':task_rows,
                'reviewer_instruction':(
                    'Two DIFFERENT humans annotate independently without access '
                    'to FLUID labels or model predictions. Left image is raw; '
                    'right image marks target location. Mark PRESENT only if '
                    'the ring is on the assessed object; provide the object '
                    'bbox in LEFT IMAGE CROP pixel coordinates 0..crop_size. '
                    'Classify according to the visible object, not FLUID; '
                    'use UNCERTAIN for occlusion/ambiguous class. '
                    'ABSENT means no physical target at that anchor. '
                    'Do not claim cases are random or whole-frame exhaustive.'),
                'scope':'preselected_15_cases_only_no_full_frame_recall_or_precision',
            }
            (temp/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            _write_csv(temp/'reviewer_A_TEMPLATE.csv',task_rows,'REPLACE_REVIEWER_A')
            _write_csv(temp/'reviewer_B_TEMPLATE.csv',task_rows,'REPLACE_REVIEWER_B')
            from streetlab_phase3.video.review_ui import offline_review_html
            (temp/'reviewer_ui.html').write_text(offline_review_html(manifest),encoding='utf-8')
            # No truth/FLUID mapping appears in any reviewer-facing file.
            if output_dir.exists():
                raise FileExistsError('Review output appeared during write')
            os.replace(temp,output_dir)
            return manifest
        finally:
            if temp.exists():shutil.rmtree(temp)


def _review(path: Path, manifest: dict[str, Any]) -> tuple[str,dict[str,dict[str,Any]]]:
    accepted={x['case_id']:x for x in manifest['cases']}
    records={}
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Review CSV requires all protocol fields')
        names=set()
        for row in reader:
            cid=row['case_id'];name=row['reviewer_id'].strip()
            if cid not in accepted or cid in records:
                raise ValueError('Unknown or duplicate review case ID')
            if (not name or name.startswith('REPLACE_') or len(name)>100):
                raise ValueError('Each review requires a real reviewer identifier')
            names.add(name)
            present=row['presence'].strip().upper()
            if present not in PRESENCE:
                raise ValueError(f'Incomplete/invalid review for {cid}: {present}')
            klass=row['physical_class'].strip().upper()
            box_fields=[row[k].strip() for k in ('x1','y1','x2','y2')]
            coords=None
            if present=='PRESENT':
                if klass not in CLASSES:
                    raise ValueError('PRESENT requires explicit physical class')
                if any(not x for x in box_fields):
                    raise ValueError('PRESENT requires manually drawn crop bounding box')
                coords=tuple(float(x) for x in box_fields)
                size=accepted[cid]['crop_size']
                if (not all(math.isfinite(c) for c in coords)
                    or not 0 <= coords[0] < coords[2] <= size
                    or not 0 <= coords[1] < coords[3] <= size):
                    raise ValueError('Review bbox outside left crop pixel coordinates')
                # The target anchor must lie within the reviewed object box
                ax,ay=accepted[cid]['anchor_in_crop']
                if not (coords[0]-5 <= ax <= coords[2]+5
                    and coords[1]-5 <= ay <= coords[3]+5):
                    raise ValueError('Reviewed object box is not at selected target')
            else:
                if (klass not in ('','UNKNOWN','NONE') or any(box_fields)):
                    raise ValueError('ABSENT/UNCERTAIN must not carry object bbox/class')
            records[cid]={'presence':present,'class':klass,'bbox':coords,
                          'note':row.get('note','')[:500]}
        if len(names)!=1 or set(records)!=set(accepted):
            raise ValueError('One complete reviewer per file and every case required')
    return names.pop(),records


def _iou(a:tuple[float,...],b:tuple[float,...])->float:
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    ar=(a[2]-a[0])*(a[3]-a[1]);br=(b[2]-b[0])*(b[3]-b[1])
    den=ar+br-inter
    return inter/den if den else 0.


def validate_pack(pack_dir:Path)->dict[str,Any]:
    manifest=json.loads((pack_dir/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status')!=STATUS or manifest.get('eligible_for_promotion') is not False
        or manifest.get('scope')!='preselected_15_cases_only_no_full_frame_recall_or_precision'):
        raise ValueError('Unexpected adjudication review provenance or promotion status')
    cases=manifest.get('cases',[])
    if not cases or len(cases)!=manifest.get('case_count'):
        raise ValueError('Review manifest count mismatch')
    if len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate review case IDs')
    for c in cases:
        member=Path(c['image_file'])
        if (member.is_absolute() or '..' in member.parts
            or member.parts[0]!='images'):
            raise ValueError('Unsafe image path in review manifest')
        if _hash((pack_dir/member).read_bytes())!=c['image_sha256']:
            raise ValueError('Tampered reviewed image file')
    return manifest


def adjudicate_pair(pack_dir:Path, review_a:Path, review_b:Path,
                    output:Path,*,min_iou:float=.5)->dict[str,Any]:
    """Only two independent agreeing labels can yield an adjudicated case.

    No predicted labels or FLUID annotations are consulted. The result is a
    case-level sidecar, never a derived whole-frame precision/recall estimate.
    """
    if not 0<min_iou<=1:raise ValueError('Invalid independent-review IoU gate')
    manifest=validate_pack(pack_dir)
    a_name,a=_review(review_a,manifest)
    b_name,b=_review(review_b,manifest)
    if a_name==b_name or review_a.resolve()==review_b.resolve():
        raise ValueError('Reviews must come from two different people and files')
    result=[]
    summary=Counter()
    for case in manifest['cases']:
        cid=case['case_id'];left,right=a[cid],b[cid]
        if (left['presence']=='ABSENT' and right['presence']=='ABSENT'):
            outcome='AGREED_ABSENT';box=None;klass=None
        elif (left['presence']==right['presence']=='PRESENT'
              and left['class']==right['class']
              and left['class']!='UNKNOWN'
              and _iou(left['bbox'],right['bbox'])>=min_iou):
            outcome='AGREED_PRESENT';klass=left['class']
            box=[(x+y)/2 for x,y in zip(left['bbox'],right['bbox'])]
        else:
            outcome='UNRESOLVED';box=None;klass=None
        summary[outcome]+=1
        result.append({'case_id':cid,'source_frame':case['source_frame'],
                       'decision':outcome,'physical_class':klass,
                       'consensus_bbox_crop':box,
                       'source_image_file':case['image_file'],
                       'reviewer_bbox_iou':(_iou(left['bbox'],right['bbox'])
                           if left['bbox'] and right['bbox'] else None)})
    report={'status':CONSENSUS_STATUS,'eligible_for_promotion':False,
            'review_manifest_sha256':_hash((pack_dir/'manifest.json').read_bytes()),
            'review_A_sha256':_hash(review_a.read_bytes()),
            'review_B_sha256':_hash(review_b.read_bytes()),
            'reviewer_ids':[a_name,b_name],'iou_agreement_gate':min_iou,
            'cases':result,'case_count':len(result),
            'decisions':dict(summary),
            'reviewed_vehicle_instances_not_exhaustive':True,
            'physical_precision':None,'full_frame_recall':None,
            'is_original_FLUID_ground_truth':False,
            'caution':('Selected W04 cases were targeted using earlier detector '
                       'errors; no unbiased population precision or whole-frame '
                       'recall can be inferred. UNKNOWN/UNCERTAIN/different-class '
                       'reviewer decisions remain unresolved. Existing FLUID truth '
                       'and candidate detection geometry are not rewritten.')}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    return report


def evaluate_adjudicated_candidates(
    *, pack_dir:Path, consensus_file:Path, box_lab_dir:Path, output:Path,
) ->dict[str,Any]:
    """Link adjudicated cases to immutable candidate BOXES without altering FLUID.

    Reports case-level presence/compatible-class coverage only, NEVER full-frame
    recall, detector precision, or production eligibility. No missing label
    is inferred from unreviewed pixels.
    """
    manifest=validate_pack(pack_dir)
    consensus=json.loads(consensus_file.read_text(encoding='utf-8'))
    if (consensus.get('status')!=CONSENSUS_STATUS
        or consensus.get('eligible_for_promotion') is not False
        or consensus.get('review_manifest_sha256')!=_hash((pack_dir/'manifest.json').read_bytes())
        or consensus.get('case_count')!=len(manifest['cases'])):
        raise ValueError('Missing/invalid independent manual consensus provenance')
    from streetlab_phase3.video.integrated_box_lab import ALL_CLASSES
    matrix_file=box_lab_dir/'matrix_report.json'
    matrix=json.loads(matrix_file.read_text(encoding='utf-8'))
    if (matrix.get('status')!='EXPERIMENTAL_PRETRACK_BOX_MATRIX_W04_TUNING_ONLY'
        or matrix.get('eligible_for_promotion') is not False
        or matrix.get('source_provenance',{}).get('manifest_sha256')
             !=manifest['source_bundle_manifest_sha256']):
        raise ValueError('Box candidate source manifest does not match review evidence')
    official=matrix.get('candidates',[])
    if len(official)!=matrix.get('candidate_count') or not official:
        raise ValueError('Incomplete model candidate matrix')
    if not all(case['source_frame'] in matrix['frames'] for case in manifest['cases']):
        raise ValueError('Review case missing from captured candidate frames')
    consensus_by_id={c['case_id']:c for c in consensus['cases']}
    if len(consensus_by_id)!=len(manifest['cases']) or set(consensus_by_id)!={c['case_id'] for c in manifest['cases']}:
        raise ValueError('Consensus cases mismatched')
    observations=[]
    for setting in official:
        name=setting['name']
        if not name or any(t not in 'abcdefghijklmnopqrstuvwxyz0123456789_.' for t in name):
            raise ValueError('Unsafe candidate name')
        csvfile=box_lab_dir/'candidate_boxes'/(name+'.csv')
        if _hash(csvfile.read_bytes())!=matrix['candidate_box_csv_sha256'][name]:
            raise ValueError(f'Candidate geometry checksum changed: {name}')
        by_frame={}
        with csvfile.open('r',newline='',encoding='utf-8') as fh:
            reader=csv.DictReader(fh)
            if not {'frame','x1','y1','x2','y2','vehicle_class'}.issubset(reader.fieldnames or []):
                raise ValueError('Incomplete candidate geometry CSV')
            for r in reader:
                frame=int(r['frame']);klass=r['vehicle_class']
                bbox=[float(r[k]) for k in ('x1','y1','x2','y2')]
                if (frame not in matrix['frames'] or klass not in ALL_CLASSES
                    or not all(math.isfinite(x) for x in bbox)
                    or not 0<=bbox[0]<bbox[2]<=3840
                    or not 0<=bbox[1]<bbox[3]<=2160):
                    raise ValueError('Invalid candidate box or frame')
                by_frame.setdefault(frame,[]).append((klass,bbox))
        counts=Counter()
        entries=[]
        for case in manifest['cases']:
            answer=consensus_by_id[case['case_id']]
            decision=answer['decision']
            if decision not in ('AGREED_PRESENT','AGREED_ABSENT','UNRESOLVED'):
                raise ValueError('Invalid consensus decision')
            if decision=='UNRESOLVED':
                counts['unresolved_cases']+=1
                continue
            candidates=by_frame.get(case['source_frame'],[])
            ox,oy=case['crop_origin'];px,py=case['anchor_in_crop']
            if decision=='AGREED_PRESENT':
                box=answer.get('consensus_bbox_crop')
                if (not isinstance(box,list) or len(box)!=4
                    or answer.get('physical_class') not in CLASSES
                    or answer['physical_class']=='UNKNOWN'):
                    raise ValueError('Malformed agreed-present manual case')
                abs_box=[box[0]+ox,box[1]+oy,box[2]+ox,box[3]+oy]
                overlapping=[(klass,geom) for klass,geom in candidates
                             if abs_box[0] <= (geom[0]+geom[2])/2 <= abs_box[2]
                             and abs_box[1] <= (geom[1]+geom[3])/2 <= abs_box[3]]
                compatible=[klass for klass,_ in overlapping
                            if klass==answer['physical_class']]
                counts['agreed_present_cases']+=1
                if overlapping:counts['any_class_center_inside_manual_box']+=1
                if compatible:counts['correct_class_center_inside_manual_box']+=1
                entries.append({'case_id':case['case_id'],'decision':decision,
                                'physical_class':answer['physical_class'],
                                'candidate_count_center_in_manual_box':len(overlapping),
                                'candidate_classes_center_in_manual_box':sorted(set(x[0] for x in overlapping)),
                                'at_least_one_correct_class_center':bool(compatible)})
            else:
                # Positive candidate at an agreed-absent target is only a
                # *reviewed-anchor* event; it does not imply an absent frame.
                bx,by=ox+px,oy+py
                at_anchor=sum(math.hypot((q[0]+q[2])/2-bx,(q[1]+q[3])/2-by)<=15
                              for _klass,q in candidates)
                counts['agreed_absent_cases']+=1
                if at_anchor:counts['prediction_center_near_agreed_absent_anchor']+=1
                entries.append({'case_id':case['case_id'],'decision':decision,
                                'candidate_count_near_absent_anchor':at_anchor})
        observations.append({'candidate':name,'reviewed_case_counts':dict(counts),
                             'reviewed_case_evidence':entries})
    report={'status':'W04_REVIEWED_CASE_BOX_COVERAGE_EXPLORATORY',
            'eligible_for_promotion':False,
            'box_lab_manifest_sha256':_hash(matrix_file.read_bytes()),
            'consensus_sha256':_hash(consensus_file.read_bytes()),
            'review_manifest_sha256':consensus['review_manifest_sha256'],
            'candidate_count':len(observations),'candidates':observations,
            'physical_precision':None,'full_frame_recall':None,
            'limitations':('Manual review is selected around pre-existing detector '
                'cases, not an unbiased exhaustive frame census. Box-center '
                'coverage of independently adjudicated objects is not overall '
                'precision/recall or track identity. Candidate score uses NONE '
                'of FLUID labels; frozen original metrics remain untouched.')}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    return report