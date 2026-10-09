"""Atomic W04 real-box capture + independent-case comparison in one invocation.

User-local immutable development evidence. W04 is not unseen holdout;
no tracking, FLUID relabeling, or model promotion occurs here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .manual_review_protocol import evaluate_adjudicated_candidates, validate_pack
from .premerge_box_capture import bundle_data, run_integrated_lab
from .reviewed_cached_center_bridge import compare_reviewed_centers


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def run_w04_reviewed_box_batch(*, bundle:Path, review_dir:Path,
        consensus_file:Path, output_dir:Path, video:Path|None=None,
        model_dir:Path|None=None, cached_only:bool=False,
        box_runner:Callable[...,dict[str,Any]]|None=None,
        box_review:Callable[...,dict[str,Any]]|None=None) -> dict[str,Any]:
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite evidence folder: {output_dir}')
    # Complete bundle integrity and exact cohort are validated before creating output.
    frames,_,source=bundle_data(bundle)
    reviewed=validate_pack(review_dir)
    if reviewed['source_bundle_manifest_sha256'] != source['manifest_sha256']:
        raise ValueError('Independent reviews do not belong to the selected evidence bundle')
    baseline=compare_reviewed_centers(bundle=bundle,review_dir=review_dir,
                                      consensus_file=consensus_file)
    if video is not None and not video.is_file():
        raise FileNotFoundError('Actual source video path does not exist')
    if not cached_only and model_dir is None:
        raise ValueError('Real OpenVINO run requires --model-dir pointing to the original complete SHA-verified export; the bundle only contains XML/BIN')
    if model_dir is not None and not model_dir.is_dir():
        raise FileNotFoundError('OpenVINO model directory does not exist')
    if not cached_only:
        from .openvino_export import hash_model_tree
        if hash_model_tree(model_dir) != source['openvino_sha256']:
            raise ValueError('Actual original OpenVINO directory differs from SHA-verified W04 export; do not use incomplete ZIP model files')
    if box_runner is None:
        box_runner=run_integrated_lab
    if box_review is None:
        box_review=evaluate_adjudicated_candidates

    output_dir.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.stage-',dir=output_dir.parent))
    try:
        cache_file=temp/'cached_review.json'
        cache_file.write_text(json.dumps(baseline,indent=2),encoding='utf-8')
        report={
            'status':'W04_SINGLE_BATCH_CACHED_ONLY' if cached_only
                    else 'W04_PRETRACK_BOX_AND_DUAL_REVIEW_SINGLE_BATCH',
            'eligible_for_promotion':False,
            'source_bundle_manifest_sha256':source['manifest_sha256'],
            'review_manifest_sha256':baseline['review_manifest_sha256'],
            'human_consensus_sha256':baseline['consensus_sha256'],
            'frames':frames,'reviewed_cases':13,'unresolved_cases':2,
            'cached_review_summary':baseline['case_level_center_coverage_by_backend'],
            'no_full_frame_physical_precision':True,
            'no_long_run_tracking_claim':True,
        }
        if not cached_only:
            # Full isolated model export verified above and by run_integrated_lab.
            box_dir=temp/'box_lab'
            matrix=box_runner(bundle=bundle,model_path=model_dir,
                              output_dir=box_dir,video=video)
            if (matrix.get('status')!='EXPERIMENTAL_PRETRACK_BOX_MATRIX_W04_TUNING_ONLY'
                or matrix.get('eligible_for_promotion') is not False
                or matrix.get('frames') != frames
                or matrix.get('candidate_count') != 25):
                raise ValueError('Unexpected real box candidate matrix cohort')
            compared=box_review(pack_dir=review_dir,consensus_file=consensus_file,
                                box_lab_dir=box_dir,
                                output=temp/'reviewed_candidate_comparison.json')
            if (compared['candidate_count']!=25
                or compared.get('eligible_for_promotion') is not False):
                raise ValueError('Review evaluation did not cover all candidates')
            report.update({
                'real_premerge_box_count':matrix['raw_box_count'],
                'candidate_count':compared['candidate_count'],
                'source_frame_evidence':matrix['source_frame_evidence'],
                'box_matrix_report_sha256':_sha(box_dir/'matrix_report.json'),
                'reviewed_candidate_comparison_sha256':
                    _sha(temp/'reviewed_candidate_comparison.json'),
            })
        report['cached_review_file_sha256']=_sha(cache_file)
        (temp/'one_batch_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if output_dir.exists():
            raise FileExistsError('Refusing to overwrite completed integrated experiment')
        os.replace(temp,output_dir)
        return report
    finally:
        if temp.exists():
            shutil.rmtree(temp)