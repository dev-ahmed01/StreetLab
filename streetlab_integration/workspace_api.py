"""M6 project-bound read-only operational overview API."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from streetlab_integration.video_jobs import VideoStore, JobError
from streetlab_integration.reconstruction import ReconstructionError
from streetlab_integration.site_inputs import SiteInputError
from streetlab_integration.site_scenarios import ScenarioError
from streetlab_integration.workspace_read import project_workspace

def mount_workspace_routes(app: FastAPI, workdir: Path) -> None:
    store=VideoStore(workdir)
    @app.get("/api/projects/{project_id}/workspace")
    def read_workspace(project_id: str):
        try:
            return project_workspace(store,project_id)
        except JobError as exc:
            raise HTTPException(status_code=404 if "not found" in str(exc).lower() else 409,detail=str(exc)) from exc
        except (ScenarioError,SiteInputError,ReconstructionError,ValueError,TypeError,KeyError,FileNotFoundError,OSError,json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409,detail="Project evidence could not be verified; inspect source receipts and retry") from exc