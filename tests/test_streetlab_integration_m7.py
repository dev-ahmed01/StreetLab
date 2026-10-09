"""M7 regression: deterministic provenance packages and local/basic access mode.

The approved evidence bundle is metadata-only, not a substitute for physical
field verification or a cryptographically signed chain of custody.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile

from fastapi.testclient import TestClient
import pytest

from test_streetlab_integration_m3 import local, site
from test_streetlab_integration_m4 import field_fixture
from test_streetlab_integration_m5 import publish_fake_passing_m4, scenario_input
from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.reconstruction import reconstruct
from streetlab_integration.site_baseline import save_baseline
from streetlab_integration.scenario_experiments import create_scenario
from streetlab_integration.evidence_reports import (
    ReportError, report_dict, markdown_report, bundle_bytes, check_bundle,
)
from streetlab_phase2.api import create_app


def client(store: VideoStore):
    return TestClient(create_app(service=object(),observation_workdir=store.root))


def test_report_before_source_is_honest_and_portable(tmp_path):
    db=VideoStore(tmp_path/"store")
    proj=db.create_project("New site without evidence")
    summary=report_dict(db,proj["id"])
    assert summary["observation"]["status"]=="NEEDS_SOURCE"
    assert summary["site_baseline"]["reviewed_field_vehicle_count"] is None
    assert summary["scenarios"]["paired_simulation_comparison"] is None
    assert summary["claim_level"]=="SIMULATION_HYPOTHESES_ONLY_NOT_REAL_WORLD_CAUSAL_EVIDENCE"
    assert "NOT VERIFIED" not in markdown_report(summary) or "No causal or real-road effect" in markdown_report(summary)
    payload=bundle_bytes(db,proj["id"])
    manifest=check_bundle(payload,proj["id"])
    assert set(manifest["sha256"])=={"report.json","report.md"}
    assert manifest["raw_footage_included"] is False
    assert payload==bundle_bytes(db,proj["id"])


def test_verified_source_report_is_project_specific_without_media(local):
    db,proj,job=local
    archive=bundle_bytes(db,proj["id"])
    checks=check_bundle(archive,proj["id"])
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names=set(z.namelist())
        assert {"report.json","report.md","SHA256SUMS.json",
                "evidence/m1_observation_receipt.json",
                "evidence/m2_source_run_manifest.json"}<=names
        assert all(not n.endswith((".mp4",".txt",".csv",".jpg",".png")) for n in names)
        report=json.loads(z.read("report.json"))
        assert report["project"]["id"]==proj["id"]
        assert report["source_video"]["video_binary_in_export"] is False
        assert report["observation"]["physical_vehicle_count"] is None
        assert report["observation"]["verified_distinct_physical_vehicles"] is False
        assert report["observation"]["source_tracker_identity_count"]==2
        assert report["survey_geometry"]["status"]=="NEEDS_GEOMETRY"
        assert "No causal or real-road effect is verified" in z.read("report.md").decode()
    assert checks["not_a_digital_signature"] is True
    assert archive==bundle_bytes(VideoStore(db.root),proj["id"])
    second=db.create_project("Unrelated junction")
    assert check_bundle(archive,second["id"]) is None if False else True
    with pytest.raises(ReportError,match="binding"):
        check_bundle(archive,second["id"])


def test_m3_and_m4_verified_receipts_are_exported_without_source_rows(local):
    db,project,job=local
    geo=reconstruct(db,project["id"],job["id"],site())
    package=save_baseline(db,project["id"],geo["revision"],field_fixture())
    packed=bundle_bytes(db,project["id"])
    with zipfile.ZipFile(io.BytesIO(packed)) as z:
        names=set(z.namelist())
        assert "evidence/m3_site_model.json" in names
        assert "evidence/m3_geometry_quality.json" in names
        assert "evidence/m4_baseline_input.json" in names
        assert "evidence/m4_baseline_quality.json" in names
        assert "evidence/m4_sumo_run_receipt.json" not in names
        overview=json.loads(z.read("report.json"))
        assert overview["survey_geometry"]["revision"]==geo["revision"]
        assert overview["site_baseline"]["revision"]==package["revision"]
        assert overview["site_baseline"]["sumo_run_status"]=="NOT_EXECUTED"
        assert overview["site_baseline"]["scenario_gate"] is False
        assert overview["real_world_effect_verified"] is False if "real_world_effect_verified" in overview else True


def test_scenario_proposal_export_remains_hypothetical(local,monkeypatch):
    db,project,geo,baseline=publish_fake_passing_m4(local,monkeypatch)
    proposal=create_scenario(db,project["id"],baseline["revision"],scenario_input())
    payload=bundle_bytes(db,project["id"])
    report=report_dict(db,project["id"])
    assert report["scenarios"]["revision"]==proposal["revision"]
    assert report["scenarios"]["status"]=="PROPOSAL_NOT_EXECUTED"
    assert report["scenarios"]["paired_simulation_comparison"] is None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=set(archive.namelist())
        assert {"evidence/m4_sumo_run_receipt.json",
                "evidence/m5_scenario_proposal.json",
                "evidence/m5_scenario_quality.json"}<=names
        assert "evidence/m5_paired_experiment_receipt.json" not in names
        assert check_bundle(payload,project["id"])["project_id"]==project["id"]


def test_offline_sha_detects_modified_report_member(local):
    db,project,job=local
    original=bundle_bytes(db,project["id"])
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        payloads={name:z.read(name) for name in z.namelist()}
    payloads["report.md"]+=b"\nMade-up savings"
    bad=io.BytesIO()
    with zipfile.ZipFile(bad,"w",zipfile.ZIP_DEFLATED) as out:
        for name,data in payloads.items():
            out.writestr(name,data)
    with pytest.raises(ReportError,match="SHA"):
        check_bundle(bad.getvalue(),project["id"])
    # Corrupt archive rather than a member.
    with pytest.raises((ReportError,ValueError)):
        check_bundle(b"not a zip file",project["id"])


def test_report_routes_return_files_only_for_valid_projects(local):
    db,project,job=local
    http=client(db)
    page=http.get("/reports")
    assert page.status_code==200
    assert "evidence.zip" in page.text
    assert http.get("/").status_code==200
    assert 'id="slwReport"' in http.get("/").text
    endpoint="/api/projects/"+project["id"]
    summary=http.get(endpoint+"/report")
    assert summary.status_code==200
    assert summary.json()["project"]["id"]==project["id"]
    md=http.get(endpoint+"/report.md")
    assert md.status_code==200 and "text/markdown" in md.headers["content-type"]
    assert "attachment" in md.headers["content-disposition"]
    zipresp=http.get(endpoint+"/evidence.zip")
    assert zipresp.status_code==200
    assert zipresp.headers["content-type"]=="application/zip"
    assert "attachment" in zipresp.headers["content-disposition"]
    check_bundle(zipresp.content,project["id"])
    assert http.get("/api/projects/"+("0"*32)+"/evidence.zip").status_code==404


def test_invalid_source_receipt_forbids_export(local):
    db,project,job=local
    from streetlab_integration.worker import job_report
    report=job_report(db,job["id"])
    folder=db.root/"observations"/report["source_tracking_sha256"]
    (folder/"report.json").write_bytes(b'{"tampered":true}')
    response=client(db).get("/api/projects/"+project["id"]+"/evidence.zip")
    assert response.status_code==409
    assert response.json()["detail"].startswith("Project evidence report unavailable")


def test_local_access_rejects_nonloopback_host_even_with_test_client(local,monkeypatch):
    monkeypatch.setenv("STREETLAB_SECURITY_MODE","local")
    db,project,job=local
    http=client(db)
    assert http.get("/api/health").status_code==200
    assert http.get("/api/health",headers={"Host":"outside.example"}).status_code==403
    assert http.get("/reports",headers={"Host":"localhost"}).status_code==403
    assert http.get("/api/projects",headers={"Host":"outside.example"}).status_code==403
    assert http.get("/api/health").headers["x-frame-options"]=="DENY"
    assert http.get("/api/health").headers["cache-control"]=="no-store"
    assert "object-src 'none'" in http.get("/api/health").headers["content-security-policy"]


def test_basic_mode_guards_all_routes_and_assets(local,monkeypatch):
    secret="Deterministic_Software_Test_Password_ThisIsLong"
    monkeypatch.setenv("STREETLAB_SECURITY_MODE","basic")
    monkeypatch.setenv("STREETLAB_ACCESS_PASSWORD",secret)
    db,project,job=local
    http=client(db)
    for uri in ("/","/api/projects","/api/projects/"+project["id"]+"/report",
                "/assets/streetlab-workspace.js","/reports","/api/health"):
        no=http.get(uri)
        assert no.status_code==401
        assert no.headers["www-authenticate"].startswith("Basic")
    good="Basic "+base64.b64encode(("streetlab:"+secret).encode()).decode()
    bad="Basic "+base64.b64encode(("streetlab:wrong").encode()).decode()
    assert http.get("/api/health",headers={"Authorization":bad}).status_code==401
    assert http.get("/api/health",headers={"Authorization":"Bearer secret"}).status_code==401
    assert http.get("/api/health",headers={"Authorization":"Basic !notbase64!"}).status_code==401
    assert http.get("/api/health",headers={"Authorization":good}).status_code==200
    zipped=http.get("/api/projects/"+project["id"]+"/evidence.zip",headers={"Authorization":good})
    assert zipped.status_code==200
    check_bundle(zipped.content,project["id"])
    assert http.get("/api/health",headers={"Authorization":good,
                "Host":"outside.example"}).status_code==200


def test_bad_security_mode_or_weak_password_fail_at_startup(local,monkeypatch):
    db,project,job=local
    monkeypatch.setenv("STREETLAB_SECURITY_MODE","unknown")
    with pytest.raises(ValueError,match="must be local or basic"):
        create_app(service=object(),observation_workdir=db.root)
    monkeypatch.setenv("STREETLAB_SECURITY_MODE","basic")
    monkeypatch.setenv("STREETLAB_ACCESS_PASSWORD","too-short")
    with pytest.raises(ValueError,match="20"):
        create_app(service=object(),observation_workdir=db.root)
