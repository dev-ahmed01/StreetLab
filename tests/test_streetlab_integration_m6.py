"""M6 integrated workspace acceptance: project isolation, readiness, provenance.

Synthetic test fixtures are NOT measured traffic or proof of physical impact.
"""
from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

pytest.importorskip("streetlab_integration.workspace_read")

from test_streetlab_integration_m3 import local, site
from test_streetlab_integration_m4 import field_fixture
from test_streetlab_integration_m5 import publish_fake_passing_m4, scenario_input
from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.workspace_read import project_workspace
from streetlab_integration.reconstruction import reconstruct
from streetlab_integration.site_baseline import save_baseline
from streetlab_integration.scenario_experiments import create_scenario
from streetlab_phase2.api import create_app


def test_empty_workspace_requires_source(tmp_path):
    db=VideoStore(tmp_path/"empty")
    project=db.create_project("Unsigned live junction")
    result=project_workspace(db,project["id"])
    assert result["stages"]["observation"]["status"]=="NEEDS_SOURCE"
    assert result["stages"]["geometry"]["status"]=="WAITING_FOR_OBSERVATION"
    assert result["stages"]["baseline"]["status"]=="WAITING_FOR_GEOMETRY"
    assert result["stages"]["scenarios"]["status"]=="WAITING_FOR_BASELINE"
    assert result["next_action"]["stage"]=="observation"
    assert result["real_world_impact_verified"] is False
    assert result["physical_vehicle_count_inferred_from_tracker_ids"] is False
    assert result["synthetic_phase2_separate"] is True


def test_verified_video_project_starts_at_geometry_not_physical_counts(local):
    db,project,job=local
    result=project_workspace(db,project["id"])
    stage=result["stages"]["observation"]
    assert stage["status"]=="OBSERVATION_VERIFIED"
    assert stage["last_verified_job"]["tracking_identity_count"]==2
    assert stage["last_verified_job"]["observed_points"]==10
    assert stage["last_verified_job"]["physical_vehicle_count_confirmed"] is False
    assert stage["last_verified_job"]["coordinate_system"]=="SOURCE_IMAGE_PIXELS"
    assert result["next_action"]["stage"]=="geometry"
    assert result["stages"]["geometry"]["movement_count"] is None
    assert result["stages"]["baseline"]["manual_distinct_vehicle_count"] is None


def test_reviewed_site_then_unrun_baseline_stays_gated(local):
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    result=project_workspace(db,project["id"])
    assert result["stages"]["geometry"]["status"]=="GEOMETRY_REVIEWED"
    assert result["stages"]["geometry"]["surveyed_local_meters"] is True
    assert result["stages"]["geometry"]["anchors"]==4
    assert result["stages"]["geometry"]["checks"]==1
    assert result["next_action"]["stage"]=="baseline"
    base=save_baseline(db,project["id"],geo["revision"],field_fixture())
    snapshot=project_workspace(VideoStore(db.root),project["id"])
    assert snapshot["stages"]["baseline"]["revision"]==base["revision"]
    assert snapshot["stages"]["baseline"]["status"]=="BASELINE_NOT_EXECUTED"
    assert snapshot["stages"]["baseline"]["manual_distinct_vehicle_count"]==5
    assert snapshot["stages"]["baseline"]["road_graph"]["coordinate_system"]=="LOCAL_GROUND_PLANE_METERS_NOT_GPS"
    assert snapshot["stages"]["baseline"]["scenario_eligible"] is False


def test_passed_baseline_is_provisional_and_scenarios_are_hypothetical(local,monkeypatch):
    db,project,geo,base=publish_fake_passing_m4(local,monkeypatch)
    snap=project_workspace(db,project["id"])
    assert snap["stages"]["baseline"]["status"]=="BASELINE_QA_PASSED"
    assert snap["stages"]["baseline"]["scenario_eligible"] is True
    assert snap["stages"]["scenarios"]["status"]=="NEEDS_SCENARIO"
    created=create_scenario(db,project["id"],base["revision"],scenario_input())
    updated=project_workspace(db,project["id"])
    assert updated["stages"]["scenarios"]["status"]=="PROPOSAL_NOT_EXECUTED"
    assert updated["stages"]["scenarios"]["revision"]==created["revision"]
    assert updated["stages"]["scenarios"]["comparison"] is None
    assert updated["next_action"]["stage"]=="scenarios"
    assert updated["real_world_impact_verified"] is False


def test_stale_m4_baseline_does_not_promote_old_scenario(local,monkeypatch):
    db,project,geo,base=publish_fake_passing_m4(local,monkeypatch)
    create_scenario(db,project["id"],base["revision"],scenario_input())
    updated=site()
    updated["zones"][0]["label"]="Observed revised lane entrance"
    new_geo=reconstruct(db,project["id"],geo["model"]["source_job_id"],updated)
    assert new_geo["revision"]!=geo["revision"]
    snapshot=project_workspace(db,project["id"])
    assert snapshot["stages"]["geometry"]["status"]=="GEOMETRY_REVIEWED"
    assert snapshot["stages"]["baseline"]["status"]=="STALE_GEOMETRY"
    assert snapshot["stages"]["baseline"]["scenario_eligible"] is False
    assert snapshot["stages"]["scenarios"]["status"]=="STALE_BASELINE"
    assert snapshot["stages"]["scenarios"]["comparison"] is None
    assert snapshot["next_action"]["stage"]=="baseline"


def test_workspace_isolation_and_tamper_fails_closed(local):
    db,project,job=local
    other=db.create_project("Independent adjacent junction")
    assert project_workspace(db,other["id"])["stages"]["observation"]["last_verified_job"] is None
    from streetlab_integration.worker import job_report
    report=job_report(db,job["id"])
    folder=db.root/"observations"/report["source_tracking_sha256"]
    (folder/"report.json").write_text('{"tampered":true}',encoding="utf-8")
    with pytest.raises((ValueError,KeyError),match="integrity|Invalid|tampered|status"):
        project_workspace(db,project["id"])


def test_new_home_has_one_primary_workspace_and_collapsed_legacy(local):
    db,project,job=local
    client=TestClient(create_app(service=object(),observation_workdir=db.root))
    page=client.get("/")
    assert page.status_code==200
    assert 'id="slw"' in page.text
    assert 'id="slwAdvanced"' in page.text
    assert 'id="m2Panel"' in page.text
    assert 'href="/scenarios"' in page.text
    assert '/assets/streetlab-workspace.js' in page.text
    assert '/assets/streetlab-workspace.css' in page.text
    css=client.get('/assets/streetlab-workspace.css')
    script=client.get('/assets/streetlab-workspace.js')
    assert css.status_code==200 and 'text/css' in css.headers['content-type']
    assert script.status_code==200 and 'javascript' in script.headers['content-type']
    assert 'document.createElement' in script.text
    assert 'innerHTML=' not in script.text
    assert '/api/projects/' in script.text
    url='/api/projects/'+project["id"]+'/workspace'
    r=client.get(url)
    assert r.status_code==200
    assert r.json()["project"]["id"]==project["id"]
    assert r.json()["stages"]["observation"]["status"]=="OBSERVATION_VERIFIED"
    assert client.get('/api/projects/'+'0'*32+'/workspace').status_code==404
    assert client.get('/api/projects/invalid/workspace').status_code in (404,409)


def test_invalid_workspace_evidence_is_not_rendered_as_success(local):
    db,project,job=local
    from streetlab_integration.worker import job_report
    tracked=job_report(db,job["id"])
    (db.root/"observations"/tracked["source_tracking_sha256"]/"report.json").write_text(
        '{"not":"trusted"}',encoding="utf-8")
    client=TestClient(create_app(service=object(),observation_workdir=db.root))
    response=client.get('/api/projects/'+project["id"]+'/workspace')
    assert response.status_code==409
    assert response.json()["detail"]=="Project evidence could not be verified; inspect source receipts and retry"
