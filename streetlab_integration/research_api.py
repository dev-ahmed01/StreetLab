"""Read-only local research-submission page; does not execute untrusted commands."""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from streetlab_integration.research_audit_display import AuditDisplayError, read_audit
from streetlab_integration.research_page import research_page_html
from streetlab_integration.research_submission import EvidenceError, study


def mount_research_routes(app:FastAPI) -> None:
    root=Path(__file__).resolve().parents[1]

    @app.get("/api/research/summary")
    def research_summary() -> dict:
        try:
            report=study(root,root,False)
        except (EvidenceError,KeyError,ValueError,OSError) as exc:
            # Never return an incorrectly completed summary.
            raise HTTPException(status_code=409,
                detail="Research source unavailable or inconsistent; check repository evidence") from exc
        return {
            "schema_version":1,"track":report["track"],
            "chennai":report["dataset_summaries"]["chennai"],
            "fluid_w04":report["dataset_summaries"]["fluid_w04"],
            "research_workflow_status":report["research_workflow_status"],
            "production_release_approved":False,
            "sources":report["source_manifest"],
        }

    @app.get("/api/research/may25-audit")
    def may25_audit() -> dict:
        # Only an explicit original-source completed artifact can be displayed.
        # No source files or inferred results are silently loaded from Drive.
        import os
        configured = os.environ.get("STREETLAB_MAY25_AUDIT_PATH", "").strip()
        if not configured:
            raise HTTPException(status_code=409,
                detail="Original-source autonomous audit has not been configured")
        try:
            return read_audit(Path(configured))
        except (AuditDisplayError, OSError, ValueError) as exc:
            raise HTTPException(status_code=409,
                detail="Original-source autonomous audit is absent or inconsistent") from exc

    @app.get("/research",response_class=HTMLResponse)
    def research_page() -> str:
        return research_page_html()
