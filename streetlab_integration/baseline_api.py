"""Project-specific M4 baseline API: create blueprint, read evidence, export files.

CPU simulation must be launched as separate CLI; POST never invokes SUMO.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from streetlab_integration.video_jobs import VideoStore, JobError
from streetlab_integration.reconstruction import ReconstructionError
from streetlab_integration.site_inputs import SiteInputError
from streetlab_integration.site_baseline import (
    BLUEPRINT,EXEC_FILES,latest_baseline,save_baseline,verified_baseline,_folder,
)


class BaselineRequest(BaseModel):
    spatial_revision: str=Field(min_length=64,max_length=64)
    site: dict


def mount_baseline_routes(app: FastAPI, workdir: Path) -> None:
    store=VideoStore(workdir)

    def error(exc: Exception) -> HTTPException:
        if isinstance(exc,SiteInputError):
            return HTTPException(status_code=422,detail=str(exc))
        if isinstance(exc,JobError):
            return HTTPException(status_code=404 if "not found" in str(exc).lower() else 409,
                                 detail=str(exc))
        if isinstance(exc,ReconstructionError):
            return HTTPException(status_code=422,detail=str(exc))
        return HTTPException(status_code=409,
                             detail="Baseline source or stored evidence integrity validation failed")

    @app.get("/baseline",response_class=HTMLResponse)
    def baseline_page():
        from streetlab_integration.baseline_ui import baseline_html
        return baseline_html()

    @app.get("/api/projects/{project_id}/baseline")
    def latest(project_id: str):
        try:
            return latest_baseline(store,project_id)
        except (SiteInputError,JobError,ReconstructionError,
                ValueError,KeyError,OSError,TypeError,json.JSONDecodeError) as exc:
            raise error(exc) from exc

    @app.post("/api/projects/{project_id}/baseline",status_code=201)
    def create(project_id: str, req: BaselineRequest):
        try:
            return save_baseline(store,project_id,req.spatial_revision,req.site)
        except (SiteInputError,JobError,ReconstructionError,
                ValueError,KeyError,OSError,TypeError,json.JSONDecodeError) as exc:
            raise error(exc) from exc

    @app.get("/api/projects/{project_id}/baseline/{revision}")
    def get_revision(project_id: str,revision: str):
        try:
            return verified_baseline(store,project_id,revision)
        except (SiteInputError,JobError,ReconstructionError,
                ValueError,KeyError,OSError,TypeError,json.JSONDecodeError) as exc:
            raise error(exc) from exc

    @app.get("/api/projects/{project_id}/baseline/{revision}/files/{name}")
    def get_file(project_id: str,revision: str,name: str):
        try:
            verified_baseline(store,project_id,revision)
            if name not in BLUEPRINT|EXEC_FILES:
                raise SiteInputError("File is not an approved baseline artifact")
            folder=_folder(store,project_id,revision)
            file=(folder/name if name in BLUEPRINT else folder/"runtime"/name)
            if not file.is_file() or file.is_symlink():
                raise SiteInputError("Requested artifact has not been generated")
            return FileResponse(file,media_type="application/xml" if name.endswith(".xml")
                                else "text/plain",
                                headers={"Cache-Control":"no-store",
                                         "X-Content-Type-Options":"nosniff"})
        except (SiteInputError,JobError,ReconstructionError,
                ValueError,KeyError,OSError,TypeError,json.JSONDecodeError) as exc:
            raise error(exc) from exc
