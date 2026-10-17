"""Fail-closed guarantees for StreetLab's no-new-field-visit research track."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from streetlab_integration.research_submission import (
    CHENNAI,FLUID_MANIFEST,W04_DEVELOPMENT,
    EvidenceError,chennai_evidence,checked_json,fluid_evidence,
    markdown,study,validate_simulation
)

ROOT=Path(__file__).resolve().parents[1]


def test_existing_heldout_failure_is_exposed_not_erased():
    data=checked_json(ROOT,CHENNAI)
    value=chennai_evidence(data)
    assert value["holdout_generalization_pass"] is False
    assert value["capacity_gate_pass"] is False
    assert value["retuning_allowed"] is False
    assert "failed" in value["conclusion"].lower()
    forged=dict(data,holdout_generalization_pass="true")
    with pytest.raises(EvidenceError):
        chennai_evidence(forged)


def test_fluid_qa_is_not_promoted_to_an_independent_holdout():
    out=fluid_evidence(
        checked_json(ROOT,FLUID_MANIFEST),
        (ROOT/W04_DEVELOPMENT).read_text(encoding="utf-8"),
    )
    assert out["reported_point_precision_pct"]==86.70
    assert out["reported_point_recall_pct"]==91.37
    assert out["scores_recomputed_in_this_run"] is False
    assert "NOT independent" in out["data_split"]
    with pytest.raises(EvidenceError):
        fluid_evidence(checked_json(ROOT,FLUID_MANIFEST),"no such score exists")


def test_evidence_only_creates_valid_limited_submission(tmp_path):
    out=study(ROOT,tmp_path,False)
    assert out["native_sumo"]["status"]=="NOT_EXECUTED"
    assert out["research_workflow_status"]=="RECORDED_EVIDENCE_ONLY_NATIVE_SUMO_NOT_RUN"
    assert len(out["source_manifest"])==6
    assert all(len(x["sha256"])==64 for x in out["source_manifest"])
    assert out["external_site_transfer_validated"] is False
    assert out["real_world_intervention_effect_proven"] is False
    assert out["production_release_approved"] is False
    assert out["datasets_are_from_different_sites"] is True
    md=markdown(out)
    assert "held-out generalization: **FAIL**" in md
    assert "NOT performed" in md


def test_sumo_report_cannot_falsely_claim_authentic_data():
    sample={
        "incomplete_study":{"status":"NEEDS_DATA"},
        "supported_study":{"status":"SUPPORTED"},
        "guardrails":{
            "confidence_fabricated":False,
            "authority_remains_final_decision_maker":True,
            "live_runtime_mutated_by_evaluated_branch":False,
        },
        "ensemble":{"metadata":{
            "provenance":"ASSUMED","same_snapshot_for_all_members":True,"members":3
        },"branch_summaries":{
            "BASELINE":{},"DETOUR_GUIDED":{},"DETOUR_MIXED":{}
        }},
    }
    valid=validate_simulation(sample)
    assert valid["status"]=="SUMO_EXECUTED_SYNTHETIC_ONLY"
    assert valid["site_specific_field_validation"] is False
    assert valid["production_release_approved"] is False
    sample["ensemble"]["metadata"]["provenance"]="OBSERVED"
    with pytest.raises(EvidenceError):
        validate_simulation(sample)
    sample["ensemble"]["metadata"]["provenance"]="ASSUMED"
    sample["incomplete_study"]["status"]="SUPPORTED"
    with pytest.raises(EvidenceError):
        validate_simulation(sample)


def test_native_failure_exposes_root_cause_and_remedy(tmp_path):
    from streetlab_integration.research_submission import _native_failure
    (tmp_path/"native_sumo.log").write_text(
        "Traceback (most recent call last):\nModuleNotFoundError: No module named 'traci'\n",
        encoding="utf-8"
    )
    failure=_native_failure(tmp_path,1)
    assert "ModuleNotFoundError" in str(failure)
    assert "pip install traci sumolib" in str(failure)
    assert "native_sumo.log" in str(failure)


def test_sumo_home_discovery_for_windows_layout(tmp_path, monkeypatch):
    import os
    import shutil
    from streetlab_integration.research_submission import _sumo_environment
    home=tmp_path/"Eclipse"/"Sumo"
    binary=home/"bin"/"sumo.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test only")
    (home/"tools").mkdir()
    monkeypatch.delenv("SUMO_HOME", raising=False)
    monkeypatch.setattr(shutil,"which",
        lambda command: str(binary) if command=="sumo"
        else str(home/"bin"/"netconvert.exe") if command=="netconvert"
        else None)
    env=_sumo_environment()
    assert env["SUMO_HOME"]==str(home)
    assert os.environ.get("SUMO_HOME") is None


def test_native_demo_is_importable_without_pythonpath(tmp_path, monkeypatch):
    """Regression: Windows launched the file from scripts/, hiding streetlab_phase2."""
    import os
    import subprocess
    import sys
    from streetlab_integration import research_submission as module

    # Model a terminal with no PYTHONPATH override; Python -m relies on cwd.
    monkeypatch.setattr(
        module, "_sumo_environment", lambda: {"PATH": os.environ.get("PATH", "")}
    )
    result = {
        "incomplete_study": {"status": "NEEDS_DATA"},
        "supported_study": {"status": "SUPPORTED"},
        "guardrails": {
            "confidence_fabricated": False,
            "authority_remains_final_decision_maker": True,
            "live_runtime_mutated_by_evaluated_branch": False,
        },
        "ensemble": {
            "metadata": {
                "provenance": "ASSUMED",
                "same_snapshot_for_all_members": True,
                "members": 3,
            },
            "branch_summaries": {
                "BASELINE": {}, "DETOUR_GUIDED": {}, "DETOUR_MIXED": {}
            },
        },
    }
    called = []

    def fake_run(argv, **kwargs):
        called.append(argv)
        assert argv[:3] == [sys.executable, "-m", "scripts.phase2_m6_study_demo"]
        assert kwargs["cwd"] == ROOT
        assert kwargs["env"].get("PYTHONPATH") is None
        assert "--project-root" in argv and "--workdir" in argv
        (tmp_path / "native_sumo_study.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, "simulated native process output")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    validated = module.native_demo(ROOT, tmp_path, timeout=12)
    assert validated["status"] == "SUMO_EXECUTED_SYNTHETIC_ONLY"
    assert len(called) == 1
    assert "simulated native process output" in (
        tmp_path / "native_sumo.log"
    ).read_text(encoding="utf-8")


def test_research_markdown_has_no_escaped_backticks(tmp_path):
    doc = markdown(study(ROOT, tmp_path, False))
    assert chr(92) + chr(96) not in doc
    assert f"{chr(96)}submission_evidence.json{chr(96)}" in doc
