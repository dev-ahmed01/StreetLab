"""M7 read-only reproducible project-report endpoints with no media data export."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from streetlab_integration.video_jobs import VideoStore, JobError
from streetlab_integration.site_inputs import SiteInputError
from streetlab_integration.site_scenarios import ScenarioError
from streetlab_integration.reconstruction import ReconstructionError
from streetlab_integration.evidence_reports import ReportError, report_dict, markdown_report, bundle_bytes

def mount_report_routes(app: FastAPI, workdir: Path) -> None:
    store=VideoStore(workdir)
    def problem(error: Exception) -> HTTPException:
        if isinstance(error,JobError):
            return HTTPException(status_code=404 if "not found" in str(error).lower() else 409,detail=str(error))
        return HTTPException(status_code=409,detail="Project evidence report unavailable: source receipt missing, invalid or exceeds export safety limit")

    @app.get("/reports",response_class=HTMLResponse)
    def reports_page():
        from streetlab_integration.report_ui import report_html
        return report_html()

    @app.get("/api/projects/{project_id}/report")
    def summary(project_id: str):
        try: return report_dict(store,project_id)
        except (JobError,ReportError,SiteInputError,ScenarioError,ReconstructionError,ValueError,TypeError,KeyError,OSError,OverflowError,json.JSONDecodeError) as exc:
            raise problem(exc) from exc

    @app.get("/api/projects/{project_id}/report.md")
    def markdown(project_id: str):
        try:
            report=report_dict(store,project_id)
            return Response(markdown_report(report),media_type="text/markdown; charset=utf-8",headers={"Content-Disposition":'attachment; filename="streetlab-evidence-report.md"'})
        except (JobError,ReportError,SiteInputError,ScenarioError,ReconstructionError,ValueError,TypeError,KeyError,OSError,OverflowError,json.JSONDecodeError) as exc:
            raise problem(exc) from exc

    @app.get("/api/projects/{project_id}/evidence.zip")
    def archive(project_id: str):
        try:
            payload=bundle_bytes(store,project_id)
            return Response(payload,media_type="application/zip",headers={"Content-Disposition":'attachment; filename="streetlab-project-evidence.zip"'})
        except (JobError,ReportError,SiteInputError,ScenarioError,ReconstructionError,ValueError,TypeError,KeyError,OSError,OverflowError,json.JSONDecodeError) as exc:
            raise problem(exc) from exc