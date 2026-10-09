"""Evidence-only Phase 3 Build 4: preregistration + holdout validation.

Never writes production files, reclassifies FLUID, approves a model, or uses
W04 development footage as unseen validation. A 'pass' only means numerical
checks passed; release requires independent physical-object adjudication and
human approval outside this module. SHA and 14-column checks prevent swapping
candidate files after scoring. All scorecards are Build 3 observations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .continuous_box_tracking_lab import STATUS as TRACK_STATUS, policy_matrix

LOCK_STATUS = 'PHASE3_PREDECLARED_POLICY_LOCK_NOT_HOLDOUT_RESULT'
VALIDATION_STATUS = 'PHASE3_EVIDENCE_VALIDATION_NOT_PRODUCTION_PROMOTION'
PASS_LIMITS = {
    'min_holdout_frames': 120,
    'min_warmup_frames': 30,
    'max_point_recall_regression_pp': 1.0,
    'max_point_precision_regression_pp': 2.0,
    'max_correct_class_motorcycle_recall_regression_pp': 0.0,
    'max_correct_class_car_recall_regression_pp': 1.0,
    'max_fragmentation_regression_pp': 2.0,
    'max_contiguous_id_switches_added': 0,
    'max_cpu_seconds_per_frame': 1.0,
}


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    obj=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj,dict):
        raise ValueError('Evidence JSON root must be an object')
    return obj


def _number(v: Any, name: str) -> float:
    if type(v) not in (float,int) or not math.isfinite(v):
        raise ValueError(f'Missing/nonfinite numeric evidence: {name}')
    return float(v)


def _relative(directory: Path, name: str) -> Path:
    candidate=Path(name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name!=name or name.startswith('.'):
        raise ValueError('Untrusted evidence filename')
    return directory/candidate


def _validate_track_rows(path: Path, first: int, last: int) -> int:
    """Forbid duplicate (frame,ID), out-of-cohort rows or fake -1 IDs."""
    seen=set()
    count=0
    with path.open('r',encoding='utf-8',newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row)!=14:
                raise ValueError('Frozen Geo-trax track schema requires 14 columns')
            vals=[_number(float(x),'track field') for x in row]
            frame,tid=vals[0],vals[1]
            if (not frame.is_integer() or not tid.is_integer()
                or frame<first or frame>last or tid<0):
                raise ValueError('Invalid absolute frame or tracker identity')
            key=(int(frame),int(tid))
            if key in seen:
                raise ValueError('Duplicate genuine tracker ID in one frame')
            seen.add(key)
            if (vals[4]<=0 or vals[5]<=0 or vals[12]<=0 or vals[13]<=0
                or vals[10] not in (0.,1.,2.,3.) or not 0<=vals[11]<=1):
                raise ValueError('Invalid tracked geometry/class/confidence')
            count+=1
    return count


def verify_batch(directory: Path, *, verify_originals: bool = True) -> dict[str,Any]:
    """Verify saved batch and every policy report against on-disk track bytes.

    SHA verification for original video and raw FLUID is compulsory by default;
    synthetic CI callers explicitly opt out and cannot pass as a real holdout.
    """
    batch_file=directory/'batch_report.json'
    report=_read(batch_file)
    if report.get('status') != TRACK_STATUS or report.get('eligible_for_promotion') is not False:
        raise ValueError('Not an unpromoted Build 3 batch')
    if report.get('tracking_updates_are_consecutive') is not True or report.get('independent_tracker_instance_per_policy') is not True:
        raise ValueError('Continuous or independent tracking evidence missing')
    first,last=report.get('evaluated_first_frame'),report.get('evaluated_last_frame')
    start=report.get('first_decoded_frame')
    if (type(first) is not int or type(last) is not int or type(start) is not int
        or start<0 or not start<=first<=last
        or report.get('processed_frames')!=last-start+1
        or report.get('evaluated_frames')!=last-first+1
        or report.get('warmup_frames_processed')!=first-start):
        raise ValueError('Inconsistent evaluated frame cohort or warmup')
    policies=report.get('policies')
    if (not isinstance(policies,list) or not policies
        or len(policies)!=report.get('candidate_count')
        or len({p.get('name') for p in policies})!=len(policies)):
        raise ValueError('Invalid distinct policy matrix')
    approved={c['name']:c for c in policy_matrix()}
    raw=_relative(directory,report['raw_pre_global_merge_boxes_file'])
    if not raw.is_file() or sha(raw)!=report['raw_pre_global_merge_boxes_sha256']:
        raise ValueError('Original per-tile capture checksum mismatch')
    names={p['name'] for p in policies}
    all_names=names
    for p in policies:
        name=p['name']
        if name not in approved or p.get('settings')!=approved[name]:
            raise ValueError('Changed policy matrix or unknown candidate')
        path=_relative(directory,p['tracks'])
        if path.name!=name+'.txt' or not path.is_file() or sha(path)!=p['tracks_sha256']:
            raise ValueError('Candidate tracking file checksum mismatch')
        count=_validate_track_rows(path,first,last)
        if count!=p['evaluation_confirmed_rows']:
            raise ValueError('Confirmed track count differs from exported rows')
        score=_read(directory/(name+'.score.json'))
        if score!=p:
            raise ValueError('Individual candidate scorecard disagrees with batch')
        for family in ('pixel','identity'):
            if p[family].get('frame_offset') != 1 or p[family].get('max_distance_px') != 50.0:
                raise ValueError('Frozen FLUID offset/gate mutated')
        pix=p['pixel']
        if (pix.get('evaluation_frame_start')!=first+1
            or pix.get('evaluation_frame_end')!=last+1):
            raise ValueError('Candidate scoring cohort differs from source frames')
        for timing in ('postprocess_median_seconds','tracking_median_seconds'):
            if _number(p[timing],timing)<0:
                raise ValueError('Negative postprocess/tracker runtime')
    baseline=report.get('same_window_T000')
    if baseline is not None:
        for family in ('pixel','identity'):
            if baseline[family].get('frame_offset')!=1 or baseline[family].get('max_distance_px')!=50.0:
                raise ValueError('Frozen baseline matching rule changed')
        for p in policies:
            for key in ('truth_points','truth_tracks','evaluation_frame_start','evaluation_frame_end'):
                if key in baseline['pixel'] and baseline['pixel'].get(key)!=p['pixel'].get(key):
                    raise ValueError('Baseline and policy different cohort')
    provenance_path=directory/'source_provenance.json'
    origin=_read(provenance_path) if provenance_path.is_file() else None
    original_checks_verified=False
    if origin is not None:
        if origin.get('eligible_for_promotion') is not False:
            raise ValueError('Promotion provenance cannot be true')
        if (origin.get('video_frame_window')!=[first,last]
            or set(origin.get('policies',[]))!=all_names):
            raise ValueError('Source manifest cohort/policies drifted')
        if verify_originals:
            for pkey,hkey in (('video_path','video_sha256'),('fluid_path','fluid_sha256')):
                source=Path(origin[pkey])
                if not source.is_file() or sha(source)!=origin[hkey]:
                    raise ValueError(f'Original source missing or altered: {pkey}')
            original_checks_verified=True
    elif verify_originals:
        raise ValueError('Real source provenance missing')
    if _number(report['detector_median_seconds'],'detector_median_seconds')<0:
        raise ValueError('Detector runtime cannot be negative')
    return {'report':report,'provenance':origin,'batch_sha256':sha(batch_file),
            'original_source_sha256_verified':original_checks_verified,
            'policy_count':len(names)}


def lock_policy(development: Path, policy_name: str, output: Path, *,
                max_cpu_seconds_per_frame: float = 1.0) -> dict[str,Any]:
    """Freeze EXACT policy *before* assessing another footage's results."""
    evidence=verify_batch(development)
    report=evidence['report']; provenance=evidence['provenance']
    if not provenance or not evidence['original_source_sha256_verified']:
        raise ValueError('Cannot lock unverified development footage')
    selected=next((p for p in report['policies'] if p['name']==policy_name),None)
    if selected is None:
        raise ValueError('Policy absent from development experiment')
    if not 0<max_cpu_seconds_per_frame<=60:
        raise ValueError('Invalid predeclared performance limit')
    limits=dict(PASS_LIMITS)
    limits['max_cpu_seconds_per_frame']=float(max_cpu_seconds_per_frame)
    result={
        'status':LOCK_STATUS,'eligible_for_production':False,
        'source_development_batch_sha256':evidence['batch_sha256'],
        'source_development_video_sha256':provenance['video_sha256'],
        'source_development_bundle_sha256':provenance['bundle_sha256'],
        'source_fluid_sha256':provenance['fluid_sha256'],
        'source_model_sha256':provenance['openvino_model_sha256'],
        'selected_policy':policy_name,'selected_settings':selected['settings'],
        'predeclared_limits':limits,
        'selection_scope':'DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION',
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    return result


def _difference(actual: Any, baseline: Any, label: str, *,
                maximum_drop_pp: float = 0., increase_good: bool=True) -> tuple[float | None,str | None]:
    if actual is None or baseline is None:
        return None,f'{label}: unavailable for same-window comparison'
    a,b=_number(actual,label),_number(baseline,label)
    diff=(a-b)*100
    good= diff >= -maximum_drop_pp if increase_good else diff <= maximum_drop_pp
    return diff,(None if good else f'{label}: regression {diff:+.2f} pp exceeds {maximum_drop_pp} pp limit')



def reasons_other_than_external(reasons: list[str]) -> list[str]:
    external={
        'Independent blinded two-reviewer physical-object holdout audit not verified',
        'Held-out dataset independence/untouched status requires external audit',
        'End-to-end CPU tail latency and long-run tracking stability not independently verified',
    }
    return [reason for reason in reasons if reason not in external]

def validate_holdout(locked: Path, holdout: Path, output: Path, *,
                    baseline_tracks: Path | None = None,
                    scorer: Any = None) -> dict[str,Any]:
    """Read only: compare a preregistered policy with SAME-WINDOW T000.

    Never declares a production promotion; human physical-object review and
    genuine untouched test evidence are REQUIRED as independent processes.
    """
    policy_lock=_read(locked)
    if (policy_lock.get('status')!=LOCK_STATUS or
        policy_lock.get('eligible_for_production') is not False or
        policy_lock.get('selection_scope')!='DEVELOPMENT_ONLY_NOT_HOLDOUT_AND_NOT_PRODUCTION'):
        raise ValueError('Policy preregistration is invalid')
    limits=policy_lock.get('predeclared_limits')
    if not isinstance(limits,dict) or set(limits)!=set(PASS_LIMITS):
        raise ValueError('Missing predeclared validation gates')
    for key,value in limits.items():
        if _number(value,key) < 0:
            raise ValueError('Negative predeclared gate')
        if key!='max_cpu_seconds_per_frame' and value!=PASS_LIMITS[key]:
            raise ValueError('Thresholds modified after development plan')
    evidence=verify_batch(holdout)
    report=evidence['report']; provenance=evidence['provenance']
    reasons=[]
    if (not provenance or not evidence['original_source_sha256_verified']):
        reasons.append('Original holdout video/FLUID SHA provenance missing')
    else:
        if provenance['video_sha256']==policy_lock['source_development_video_sha256']:
            reasons.append('Same video as tuned development W04: not an unseen holdout')
        if provenance['openvino_model_sha256']!=policy_lock['source_model_sha256']:
            reasons.append('OpenVINO model fingerprint changed after selection')
        if provenance['fluid_sha256']==policy_lock['source_fluid_sha256']:
            reasons.append('FLUID label source reused from development footage')
        if provenance.get('video_frame_window')!=[
            report['evaluated_first_frame'],report['evaluated_last_frame']]:
            reasons.append('Scoring frame cohort differs from video provenance')
    if report.get('evaluated_frames',0)<limits['min_holdout_frames']:
        reasons.append('Fewer continuous holdout frames than predeclared')
    if report.get('warmup_frames_processed',0)<limits['min_warmup_frames']:
        reasons.append('Insufficient tracker warmup')
    if report.get('same_window_T000') is None:
        reasons.append('Same-window independent T000 baseline is missing')
    if baseline_tracks is None:
        reasons.append('Original T000 track-file bytes not supplied for independent replay')
    elif report.get('same_window_T000') is not None:
        if not baseline_tracks.is_file() or sha(baseline_tracks)!=report['same_window_T000'].get('source_sha256'):
            raise ValueError('Original same-window T000 track checksum mismatch')
        _validate_track_rows(baseline_tracks,report['evaluated_first_frame'],report['evaluated_last_frame'])

    selected=next((p for p in report['policies']
                   if p['name']==policy_lock['selected_policy']),None)
    if selected is None:
        reasons.append('Preregistered policy not present in holdout batch')
    elif selected['settings']!=policy_lock['selected_settings']:
        reasons.append('Candidate parameters differ from frozen preregistration')
    comparison={}
    if selected is not None:
        detector=_number(report['detector_median_seconds'],'detector_median_seconds')
        post=_number(selected['postprocess_median_seconds'],'postprocess_median_seconds')
        track=_number(selected['tracking_median_seconds'],'tracking_median_seconds')
        estimated_median=detector+post+track
        comparison['sum_of_stage_medians_seconds_per_frame']=estimated_median
        comparison['stage_medians_are_not_end_to_end_p95']=True
        if estimated_median>limits['max_cpu_seconds_per_frame']:
            reasons.append('CPU stage-median budget exceeded')
    baseline=report.get('same_window_T000')
    if selected is not None and baseline is not None:
        for label,segment,key,max_drop in (
            ('point_recall','pixel','point_recall','max_point_recall_regression_pp'),
            ('point_precision','pixel','point_precision','max_point_precision_regression_pp'),
            ('motorcycle_correct_class_recall','class_diagnostics','MOTORCYCLE','max_correct_class_motorcycle_recall_regression_pp'),
            ('car_correct_class_recall','class_diagnostics','CAR','max_correct_class_car_recall_regression_pp'),
            ('fragmentation','identity','fraction_truth_tracks_fragmented','max_fragmentation_regression_pp'),
        ):
            if segment=='class_diagnostics':
                new=selected.get('class_diagnostics'); old=baseline.get('class_diagnostics')
                v=new.get('by_class',{}).get(key,{}).get('correct_class_recall') if new else None
                base=old.get('by_class',{}).get(key,{}).get('correct_class_recall') if old else None
            else:
                v=selected[segment].get(key);base=baseline[segment].get(key)
            delta,problem=_difference(v,base,label,maximum_drop_pp=limits[max_drop],
                                      increase_good=(label!='fragmentation'))
            comparison[label+'_delta_pp']=delta
            if problem:
                reasons.append(problem)
        ids=selected['identity'].get('total_contiguous_id_switches')
        base_ids=baseline['identity'].get('total_contiguous_id_switches')
        if ids is None or base_ids is None:
            reasons.append('Identity switch comparison unavailable')
        else:
            extra=_number(ids,'switches')-_number(base_ids,'baseline switches')
            comparison['contiguous_id_switches_added']=extra
            if extra>limits['max_contiguous_id_switches_added']:
                reasons.append('Additional contiguous tracker ID switches')
    # Replay the scorer independently from unchanged FLUID + actual track bytes;
    # pass a scorer only in explicitly synthetic tests. A failed replay is never
    # converted to a benchmark pass by comparing two cached JSON scorecards.
    scorer_verified=False
    if baseline_tracks is not None and baseline is not None and selected is not None:
        if scorer is None:
            from .continuous_box_tracking_lab import cached_window_scorer
            scorer=cached_window_scorer(
                Path(provenance['fluid_path']),
                report['evaluated_first_frame'],report['evaluated_last_frame'])
            scorer_verified=True
        for path, target in ((baseline_tracks,baseline),
                             (holdout/selected['tracks'],selected)):
            pix,ident=scorer(tracks=path,
                fluid_tracks=Path(provenance['fluid_path']),max_distance_px=50.,
                start_frame=report['evaluated_first_frame'],
                end_frame=report['evaluated_last_frame'])
            # Every numerical contract key represented in the report must match.
            for key in ('truth_points','predicted_points','matched_points',
                        'frame_offset','max_distance_px','point_recall',
                        'point_precision','evaluation_frame_start',
                        'evaluation_frame_end'):
                if key in target['pixel'] and target['pixel'][key]!=pix.get(key):
                    raise ValueError(f'Independent pixel score replay mismatch: {key}')
            for key in ('truth_tracks','predicted_tracks','matched_truth_tracks',
                        'frame_offset','max_distance_px',
                        'fraction_truth_tracks_fragmented',
                        'total_contiguous_id_switches'):
                if key in target['identity'] and target['identity'][key]!=ident.get(key):
                    raise ValueError(f'Independent identity score replay mismatch: {key}')
            if target.get('class_diagnostics') is not None:
                if not hasattr(scorer,'class_report'):
                    if scorer_verified:
                        raise ValueError('Independent class diagnostics replay unavailable')
                else:
                    actual_class=scorer.class_report(path)
                    for klass in ('CAR','MOTORCYCLE'):
                        expected=target['class_diagnostics'].get('by_class',{}).get(klass,{}).get('correct_class_recall')
                        recomputed=actual_class.get('by_class',{}).get(klass,{}).get('correct_class_recall')
                        if expected!=recomputed:
                            raise ValueError(f'Independent class recall replay mismatch: {klass}')
    else:
        reasons.append('Independent pixel/identity/class recall replay unavailable')
    # A visual consensus for W04 TARGETED CASES cannot be reused as holdout
    # physical truth, and no artifact can establish test-set independence by
    # self-assertion. Keep these as unmet independent requirements.
    reasons.extend([
        'Independent blinded two-reviewer physical-object holdout audit not verified',
        'Held-out dataset independence/untouched status requires external audit',
        'End-to-end CPU tail latency and long-run tracking stability not independently verified',
    ])
    result={
        'status':VALIDATION_STATUS,'eligible_for_production':False,
        'eligible_for_tracking_promotion':False,
        'numerical_thresholds_passed':not any(
            r for r in reasons if r not in (
                'Independent blinded two-reviewer physical-object holdout audit not verified',
                'Held-out dataset independence/untouched status requires external audit',
                'End-to-end CPU tail latency and long-run tracking stability not independently verified',
                'Original T000 track-file bytes not supplied for independent replay',
                'Independent pixel/identity/class recall replay unavailable')),
        'numerical_validation_passed':(scorer_verified and not reasons_other_than_external(reasons)),
        'independent_scores_recomputed_from_raw_tracks':scorer_verified,
        'source_preregistration_sha256':sha(locked),
        'source_holdout_batch_sha256':evidence['batch_sha256'],
        'policy':policy_lock['selected_policy'],'holdout_frame_window':[
            report['evaluated_first_frame'],report['evaluated_last_frame']],
        'predeclared_limits':limits,'comparisons':comparison,
        'unmet_requirements':reasons,
        'frozen_fluid_offset':1,'frozen_spatial_radius_px':50.0,
        'limits':'Selected W04 policy evaluated against separate video, never promoted automatically. FLUID taxonomy known to be noisy; visual physical-object preservation and unseen-status audit remain external mandatory requirements.',
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    return result