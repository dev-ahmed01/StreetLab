"""Synthetic contract tests for a one-shot, real-data controlled W04 experiment."""
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
from streetlab_phase3.video.integrated_box_lab import RawBox, merge_boxes

RUNNER=Path(__file__).resolve().parents[1]/'scripts'/'phase3_w04_unified_hybrid_replay.py'
if not RUNNER.is_file():
    pytest.skip('Unified W04 runner absent in sparse checkout',allow_module_level=True)
spec=importlib.util.spec_from_file_location('w04_unified',RUNNER)
assert spec and spec.loader
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_result():
    return {
        'pixel':{'predicted_points':100,'matched_points':90,'truth_points':100,
                 'point_precision':.9,'point_recall':.9},
        'identity':{'predicted_tracks':10,
                    'total_contiguous_id_switches':3,
                    'fraction_truth_tracks_fragmented':.10},
        'class_diagnostics':{
            'correct_class_matches':85,
            'by_class':{
                cls:{'correct_class_matches':count,'correct_class_recall':ratio}
                for cls,count,ratio in (
                    ('CAR',55,.95),('BUS',0,None),
                    ('HEAVY_VEHICLE',10,.80),('MOTORCYCLE',20,.85))
            }}
    }


def test_replayed_original_score_must_match_exactly():
    reference=make_result()
    assert mod.frozen_score_comparison(reference,deepcopy(reference))['successful']
    changed=deepcopy(reference)
    changed['identity']['total_contiguous_id_switches']=4
    with pytest.raises(ValueError,match='did NOT reproduce'):
        mod.frozen_score_comparison(changed,reference)
    changed=deepcopy(reference)
    changed['class_diagnostics']['by_class']['MOTORCYCLE']['correct_class_matches']=19
    with pytest.raises(ValueError,match='did NOT reproduce'):
        mod.frozen_score_comparison(changed,reference)


def test_development_filter_rejects_rare_recall_or_switch_loss():
    original=make_result()
    assert mod.development_safety_gate(deepcopy(original),original)['development_gate_passed']
    regression=deepcopy(original)
    regression['class_diagnostics']['by_class']['MOTORCYCLE']['correct_class_recall']=.83
    result=mod.development_safety_gate(regression,original)
    assert not result['development_gate_passed']
    assert any('MOTORCYCLE' in r for r in result['violations'])
    regression=deepcopy(original)
    regression['identity']['total_contiguous_id_switches']=4
    assert not mod.development_safety_gate(regression,original)['development_gate_passed']
    regression=deepcopy(original)
    regression['pixel']['point_precision']=.88-1e-3
    assert not mod.development_safety_gate(regression,original)['development_gate_passed']


def test_hybrid_saves_adjacent_motorcycle_hypotheses_but_not_car():
    raw=[
        RawBox(10750,0,0,10,10,.9,'CAR',0,1),
        RawBox(10750,5,0,15,10,.8,'CAR',1,2),
        RawBox(10750,0,0,10,10,.9,'MOTORCYCLE',0,3),
        RawBox(10750,5,0,15,10,.8,'MOTORCYCLE',1,4),
    ]
    standard=mod.merge_for_experiment(raw,'control_ios030')
    hybrid=mod.merge_for_experiment(raw,'rare_iou05')
    assert len(standard)==2
    assert len(hybrid)==3
    assert sum(b.vehicle_class=='CAR' for b in hybrid)==1
    assert sum(b.vehicle_class=='MOTORCYCLE' for b in hybrid)==2
    assert all(b.frame==10750 for b in hybrid)
    with pytest.raises(ValueError):
        mod.merge_for_experiment(raw,'made_up_policy')


def test_original_control_uses_exact_unchanged_global_merge():
    originals=[
        RawBox(10,0,0,20,20,.81,'MOTORCYCLE',1,1),
        RawBox(10,1,1,20,20,.77,'MOTORCYCLE',2,2),
        RawBox(10,5,5,50,50,.8,'BUS',0,3)
    ]
    expected=merge_boxes(originals,policy='hard_nms',threshold=.30,
                         metric='ios',score_floor=.15)
    assert mod.control_merger(originals)==expected


def test_immutable_destination_rejected_without_any_expensive_work(tmp_path):
    target=tmp_path/'already_exists'
    target.mkdir()
    with pytest.raises(FileExistsError):
        mod.run_all(tmp_path,target)


def test_direct_script_entrypoint_works_outside_repo_without_pythonpath(tmp_path):
    """Regression for Windows PowerShell launching scripts/ file directly."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(RUNNER), '--help'],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert '--batch-dir' in result.stdout
    assert '--output-dir' in result.stdout
