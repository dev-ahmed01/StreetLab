"""HTTP and semantic regression for the existing StreetLab research workspace."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from streetlab_integration.research_api import mount_research_routes
from streetlab_integration.research_page import research_page_html


def test_page_has_small_existing_route_hierarchy():
    page=research_page_html()
    assert "font-family:Poppins" in page
    assert "font-family:Manrope" in page
    assert '<main>' in page
    assert 'aria-label="Research sections"' in page
    for path in ('/','/reports','/release','/fieldwork'):
        assert 'href="'+path+'"' in page
    for section in ('overview','diagnostics','simulation','boundaries'):
        assert 'id="'+section+'"' in page
    for key in ('fluid','fluidRecall','chennai','auditF1','auditHeavy','auditBus','auditStatus'):
        assert 'id="'+key+'"' in page
    assert "/api/research/may25-audit" in page
    assert "/api/research/summary" in page


def test_page_keeps_scientific_boundary_visible():
    page=research_page_html()
    assert "Historical regression" in page
    assert "Tuned development" in page
    assert "Synthetic only" in page
    assert "Chennai independent transfer failed" in page
    assert "Not production approved" in page
    assert "Tracker IDs are not independently verified unique vehicles" in page
    assert "Original-source audit unavailable or inconsistent" in page
    assert "91.42%" not in page  # no data displayed without verified saved audit
    assert "real-time inference" not in page.lower()


def test_local_http_never_invents_missing_object_attribution(monkeypatch):
    monkeypatch.delenv("STREETLAB_MAY25_AUDIT_PATH",raising=False)
    app=FastAPI()
    mount_research_routes(app)
    with TestClient(app) as client:
        page=client.get("/research")
        assert page.status_code==200
        assert 'id="auditHeavy"' in page.text
        assert "Research" in page.text
        missing=client.get("/api/research/may25-audit")
        assert missing.status_code==409
        assert "not been configured" in missing.json()["detail"]
