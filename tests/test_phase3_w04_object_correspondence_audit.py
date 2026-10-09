"""Tests for immutable W04 correspondence audit, no reviewer-label training."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import subprocess
import sys
import pytest

RUNNER=Path(__file__).resolve().parents[1]/"scripts"/"phase3_w04_object_correspondence_audit.py"
if not RUNNER.is_file():
    pytest.skip("W04 correspondence runner absent from sparse checkout",
                allow_module_level=True)
scripts=str(RUNNER.parent)
if scripts not in sys.path:
    sys.path.insert(0,scripts)
spec=importlib.util.spec_from_file_location("w04_correspondence_audit",RUNNER)
assert spec and spec.loader
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_cli_direct_execution_from_outside_project(tmp_path):
    env=dict(os.environ)
    env.pop("PYTHONPATH",None)
    r=subprocess.run([sys.executable,str(RUNNER),"--help"],cwd=tmp_path,
                     env=env,capture_output=True,text=True,
                     timeout=30,check=False)
    assert r.returncode==0, r.stderr
    assert "--original-batch-zip" in r.stdout
    assert "--recorded-consensus-zip" in r.stdout


def test_immutable_audit_target_refuses_before_input_read(tmp_path):
    output=tmp_path/"prior_evidence"
    output.mkdir()
    with pytest.raises(SystemExit):
        mod.main(["--original-batch-zip","missing1.zip",
                  "--unified-replay-zip","missing2.zip",
                  "--shadow-review-zip","missing3.zip",
                  "--recorded-consensus-zip","missing4.zip",
                  "--output-dir",str(output)])


def test_zip_member_safety_contract():
    import zipfile
    with pytest.raises(ValueError,match="Unsafe"):
        mod.zip_bytes(None,"../untrusted.txt")
    with pytest.raises(ValueError,match="Unsafe"):
        mod.zip_bytes(None,"nested/mixed.csv")
    with pytest.raises(ValueError,match="Unsafe"):
        mod.zip_bytes(None,r"folder\bad.txt")


def test_optional_real_uploaded_original_files_for_reproducible_report():
    base=Path("/mnt/data")
    files=[
        base/"W04_CONTINUOUS_201_RESULTS.zip",
        base/"W04_UNIFIED_HYBRID_REPLAY_01.zip",
        base/"W04_SHADOW_REVIEW_V2.zip",
        base/"W04_TWO_REVIEWER_CONSENSUS_20261009.zip"
    ]
    if not all(x.is_file() for x in files):
        pytest.skip("Private real W04 source archives are not in GitHub CI")
    report,obs,pairs,tracklets,reviewed=mod.generate(*files)
    assert report["original_primary_byte_identical"]
    assert len(obs)==191 and len(tracklets)==40 and len(reviewed)==7
    assert all(x["eligible_for_count"] is False for x in tracklets)
    assert all(x["geometry_is_independent_of_human_consensus"] for x in reviewed)
    assert report["new_physical_vehicles_inferred"]==0
    assert report["eligible_for_production"] is False
