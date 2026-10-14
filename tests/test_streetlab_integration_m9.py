"""M9 synthetic software tests, NOT actual survey acceptance.

Artificial M8 originals only exercise schema and mutation rejection.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_streetlab_integration_m3 import local,site
from test_streetlab_integration_m4 import field_fixture
from test_streetlab_integration_m8 import artificial_field_records
from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.reconstruction import reconstruct
from streetlab_integration.site_baseline import save_baseline
from streetlab_integration.primary_runner import sha
from streetlab_integration.fieldwork import (
    field_plan,create_field_kit,blank_kit,FieldworkError,
    check_fieldwork,original_file_hashes,
)
from streetlab_integration.field_acceptance import FieldEvidenceError
from streetlab_phase2.api import create_app


def prepared(local):
    store,project,job=local
    geo=reconstruct(store,project["id"],job["id"],site())
    return store,project,job,geo


def test_no_geometry_never_creates_kit(local,tmp_path):
    store,project,_=local
    p=field_plan(store,project["id"])
    assert p["status"]=="NEEDS_REVIEWED_GEOMETRY"
    assert p["movements"]==[]
    with pytest.raises(FieldworkError,match="M3 calibration"):
        create_field_kit(store,project["id"],tmp_path/"fieldkit")
    assert not (tmp_path/"fieldkit").exists()
    assert check_fieldwork(store,project["id"],tmp_path)["status"]=="NEEDS_REVIEWED_GEOMETRY"


def test_fieldkit_blank_rows_are_explicitly_not_measurements(local,tmp_path):
    store,project,job,geo=prepared(local)
    p=field_plan(store,project["id"])
    assert p["status"]=="PLAN_READY_FOR_INDEPENDENT_FIELD_MEASUREMENT"
    assert p["source_video_sha256"]==store.source(job["source_id"])["sha256"]
    assert p["spatial_revision"]==geo["revision"]
    assert len(p["movements"])>0
    out=tmp_path/"new-kit"
    result=create_field_kit(store,project["id"],out)
    assert result["status"]=="BLANK_KIT_CREATED_NOT_FIELD_EVIDENCE"
    assert result["release_approved"] is False
    assert result["original_field_observations_created"] is False
    template=json.loads((out/"attestation.template.json").read_text())
    assert template["data_origin"]=="UNVERIFIED_DRAFT_NOT_FIELD_EVIDENCE"
    assert template["field_data_collector"] is None
    assert template["original_file_sha256"]["census.csv"] is None
    survey=json.loads((out/"survey.template.json").read_text())
    assert survey["center_world_m"] is None
    assert all(x["outer_world_m"] is None for x in survey["arms"])
    with (out/"census.template.csv").open(newline="") as f:
        data=list(csv.DictReader(f))
    assert len(data)==len(p["movements"])
    assert all(x["count"]=="" and x["type_id"]=="" for x in data)
    with (out/"holdout.template.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(p["movements"])
    assert all(x["travel_time_s"]=="" and x["session_ref"]=="" for x in rows)
    checks=json.loads((out/"TEMPLATE_SHA256SUMS.json").read_text())
    for name,digest in checks.items():
        assert digest==sha(out/name)
    assert not (out/"census.csv").exists()
    assert not (out/"attestation.json").exists()
    assert "not field evidence" in (out/"FIELD_COLLECTION_GUIDE.md").read_text().lower()
    with pytest.raises(FieldworkError,match="overwrite"):
        create_field_kit(store,project["id"],out)
    assert json.loads((out/"TEMPLATE_SHA256SUMS.json").read_text())==checks


def test_fieldkit_requires_existing_non_symlink_parent(local,tmp_path):
    store,project,job,geo=prepared(local)
    with pytest.raises(FieldworkError,match="Output parent"):
        create_field_kit(store,project["id"],tmp_path/"missing"/"kit")
    sub=tmp_path/"good";sub.mkdir()
    link=tmp_path/"symlink"
    try:
        link.symlink_to(sub,target_is_directory=True)
    except (OSError,NotImplementedError):
        pytest.skip("Symlinks unavailable for current local account")
    with pytest.raises(FieldworkError,match="parent"):
        create_field_kit(store,project["id"],link/"kit")


def test_preflight_waits_for_m4_before_original_files(local,tmp_path):
    store,project,job,geo=prepared(local)
    response=check_fieldwork(store,project["id"],tmp_path/"anything")
    assert response["status"]=="NEEDS_M4_BASELINE"
    assert response["field_records_reconciled"] is False


def test_field_preflight_reconciles_only_original_m4_attestation(local,tmp_path):
    store,project,job,geo=prepared(local)
    base=save_baseline(store,project["id"],geo["revision"],field_fixture())
    original=artificial_field_records(tmp_path/"originals",project["id"],base)
    result=check_fieldwork(store,project["id"],original)
    assert result["status"]=="FILES_RECONCILED_OPERATOR_ATTESTED_ONLY"
    assert result["release_approved"] is False
    assert result["observer_identity_authenticated"] is False
    assert result["real_world_intervention_effect_verified"] is False
    hashes=original_file_hashes(original)
    assert hashes["status"]=="CHECKSUMS_ONLY_NOT_VALIDATION"
    assert hashes["original_file_sha256"]["holdout.csv"]==sha(original/"holdout.csv")
    # Alter raw values, even if files remain syntactically valid.
    f=original/"census.csv"
    f.write_text(f.read_text().replace(",5\n",",8\n"))
    altered=check_fieldwork(store,project["id"],original)
    assert altered["status"]=="FIELD_EVIDENCE_CONTRADICTION"
    assert altered["field_records_reconciled"] is False
    assert altered["release_approved"] is False


def test_missing_attestation_or_symlink_is_not_accepted(local,tmp_path):
    store,project,job,geo=prepared(local)
    base=save_baseline(store,project["id"],geo["revision"],field_fixture())
    original=artificial_field_records(tmp_path/"originals",project["id"],base)
    (original/"attestation.json").unlink()
    report=check_fieldwork(store,project["id"],original)
    assert report["status"]=="NEEDS_ORIGINAL_RECORDS"
    assert report["missing_files"]==["attestation.json"]
    link=tmp_path/"alias"
    try:
        link.symlink_to(original,target_is_directory=True)
    except (OSError,NotImplementedError):
        pytest.skip("Symlinks unsupported")
    with pytest.raises(FieldworkError,match="untrusted"):
        original_file_hashes(link)


def test_fieldwork_http_never_accepts_observations(local):
    store,project,job,geo=prepared(local)
    client=TestClient(create_app(service=object(),observation_workdir=store.root))
    page=client.get("/fieldwork")
    assert page.status_code==200
    assert "software never generates physical counts" in page.text
    result=client.get(f"/api/projects/{project['id']}/fieldwork-plan")
    assert result.status_code==200
    assert result.json()["spatial_revision"]==geo["revision"]
    assert result.json()["original_field_observations_created"] is False
    assert client.post(f"/api/projects/{project['id']}/fieldwork-plan",json={}).status_code==405
    homepage=client.get("/")
    assert homepage.status_code==200
    assert 'id="slwFieldwork"' in homepage.text
    assert "Fieldwork plan" in homepage.text
    release=client.get("/release")
    assert "Collect field evidence" in release.text
