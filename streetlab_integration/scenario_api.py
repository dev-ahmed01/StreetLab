"""M5 HTTP endpoints: save checked scenario proposals and read immutable results."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from streetlab_integration.video_jobs import VideoStore, JobError
from streetlab_integration.reconstruction import ReconstructionError
from streetlab_integration.site_inputs import SiteInputError
from streetlab_integration.site_scenarios import ScenarioError
from streetlab_integration.scenario_experiments import create_scenario, latest_scenario, verified_scenario, _folder

class ScenarioRequest(BaseModel):
    baseline_revision: str=Field(min_length=64,max_length=64)
    scenario: dict

def mount_scenario_routes(app: FastAPI, workdir: Path) -> None:
    store=VideoStore(workdir)
    def fail(exc):
        if isinstance(exc,(ScenarioError,SiteInputError,ReconstructionError)):
            return HTTPException(status_code=422,detail=str(exc))
        if isinstance(exc,JobError):
            return HTTPException(status_code=404 if "not found" in str(exc).lower() else 409,detail=str(exc))
        return HTTPException(status_code=409,detail="Simulation evidence failed integrity checks")
    @app.get("/scenarios",response_class=HTMLResponse)
    def page():
        from streetlab_integration.scenario_ui import scenarios_html
        return scenarios_html()
    @app.get("/api/projects/{project_id}/scenario")
    def latest(project_id: str):
        try: return latest_scenario(store,project_id)
        except (ScenarioError,SiteInputError,JobError,ReconstructionError,ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise fail(exc) from exc
    @app.post("/api/projects/{project_id}/scenario",status_code=201)
    def create(project_id: str,body: ScenarioRequest):
        try: return create_scenario(store,project_id,body.baseline_revision,body.scenario)
        except (ScenarioError,SiteInputError,JobError,ReconstructionError,ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise fail(exc) from exc
    @app.get("/api/projects/{project_id}/scenario/{revision}")
    def get_revision(project_id: str,revision: str):
        try: return verified_scenario(store,project_id,revision)
        except (ScenarioError,SiteInputError,JobError,ReconstructionError,ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise fail(exc) from exc
    @app.get("/api/projects/{project_id}/scenario/{revision}/results")
    def results(project_id: str,revision: str):
        try:
            item=verified_scenario(store,project_id,revision)
            if item["runtime"]["status"]=="NOT_EXECUTED": raise ScenarioError("Experiment not executed")
            return item["runtime"]
        except (ScenarioError,SiteInputError,JobError,ReconstructionError,ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise fail(exc) from exc
    @app.get("/api/projects/{project_id}/scenario/{revision}/files/{filename}")
    def file(project_id: str,revision: str,filename: str):
        try:
            item=verified_scenario(store,project_id,revision)
            if filename not in item["runtime"].get("files",{}): raise ScenarioError("File unavailable or not verified")
            path=_folder(store,project_id,revision)/"runtime"/filename
            if path.is_symlink() or not path.is_file(): raise ScenarioError("File missing or untrusted")
            return FileResponse(path,media_type="application/xml" if filename.endswith(".xml") else "text/plain",headers={"Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})
        except (ScenarioError,SiteInputError,JobError,ReconstructionError,ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise fail(exc) from exc