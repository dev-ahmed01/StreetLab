"""M3 project-bound guided road reconstruction endpoints.

The frame viewer shows genuine local source pixels; the API never accepts a
client filesystem path. Reconstructed meters are available only after an
independently checked and documented ground-plane calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel, Field

from streetlab_integration.video_jobs import JobError, VideoStore
from streetlab_integration.primary_runner import sha
from streetlab_integration.worker import job_report
from streetlab_integration.reconstruction import (
    ReconstructionError, latest_reconstruction, reconstruct, verified_reconstruction,
)


class ReconstructionSubmission(BaseModel):
    job_id: str = Field(min_length=32, max_length=32)
    model: dict


def mount_spatial_routes(app: FastAPI, workdir: Path) -> None:
    store = VideoStore(workdir)

    def fail(exc: Exception) -> HTTPException:
        message = str(exc)
        if isinstance(exc, JobError):
            return HTTPException(status_code=404 if "not found" in message.lower() else 409,
                                 detail=message)
        if isinstance(exc, ReconstructionError):
            return HTTPException(status_code=422, detail=message)
        return HTTPException(status_code=409,
                             detail="Spatial source or persisted reconstruction failed integrity validation")

    @app.get("/calibration", response_class=HTMLResponse)
    def calibration_page():
        from streetlab_integration.spatial_ui import calibration_html
        return calibration_html()

    @app.get("/api/projects/{project_id}/reconstruction")
    def spatial_latest(project_id: str):
        try:
            return latest_reconstruction(store, project_id)
        except (JobError, ReconstructionError, OSError, ValueError, KeyError,
                TypeError, json.JSONDecodeError) as exc:
            raise fail(exc) from exc

    @app.post("/api/projects/{project_id}/reconstruction", status_code=201)
    def spatial_save(project_id: str, payload: ReconstructionSubmission):
        try:
            return reconstruct(store, project_id, payload.job_id, payload.model)
        except (JobError, ReconstructionError, OSError, ValueError, KeyError,
                TypeError, json.JSONDecodeError) as exc:
            raise fail(exc) from exc

    @app.get("/api/projects/{project_id}/reconstruction/export")
    def export_world(project_id: str):
        try:
            result = latest_reconstruction(store, project_id)
            if result.get("status") == "NOT_CONFIGURED":
                raise ReconstructionError("Save a reviewed calibration first")
            if not result["quality"]["metric_transform_checked"]:
                raise ReconstructionError("Independent metric check is required before world-coordinate export")
            file = (store.root / "projects" / project_id / "reconstructions" /
                    result["revision"] / "source_to_world.csv")
            return FileResponse(file, media_type="text/csv",
                                filename="streetlab_source_to_local_world.csv",
                                headers={"Cache-Control":"no-store"})
        except (JobError, ReconstructionError, OSError, ValueError, KeyError,
                TypeError, json.JSONDecodeError) as exc:
            raise fail(exc) from exc

    @app.get("/api/jobs/{job_id}/source-frame")
    def source_frame(job_id: str, frame: int | None = Query(default=None, ge=0)):
        try:
            job = store.job(job_id)
            if job["status"] != "SUCCEEDED":
                raise JobError("Select a completed observation job for calibration")
            source = store.source(job["source_id"])
            target = job["config"]["first_frame"] if frame is None else frame
            if not job["config"]["first_frame"] <= target <= job["config"]["last_frame"]:
                raise ReconstructionError("Requested frame is outside this verified job interval")
            job_report(store, job_id)
            video = store.source_path(source)
            if sha(video) != source["sha256"]:
                raise ReconstructionError("Source media changed since upload")
            try:
                import cv2
            except ImportError as exc:
                raise JobError("OpenCV required for real source-frame preview") from exc
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                cap.release()
                raise ReconstructionError("Source video can no longer be opened")
            try:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise ReconstructionError("Cannot seek to the requested source frame")
                ok, bgr = cap.read()
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if not ok or bgr is None or abs(pos-target-1) > .51:
                    raise ReconstructionError("Exact source-frame decoding failed")
                if bgr.shape[:2] != (source["metadata"]["height"],
                                    source["metadata"]["width"]):
                    raise ReconstructionError("The source image geometry changed")
                h, w = bgr.shape[:2]
                if w > 1600:
                    bgr = cv2.resize(bgr, (1600, round(h * 1600/w)))
                encoded, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not encoded:
                    raise ReconstructionError("Source-frame JPEG conversion failed")
                return Response(content=data.tobytes(), media_type="image/jpeg",
                                headers={"Cache-Control":"no-store",
                                         "X-Content-Type-Options":"nosniff",
                                         "X-StreetLab-Source-Frame":str(target)})
            finally:
                cap.release()
        except (JobError, ReconstructionError, OSError, ValueError, KeyError,
                TypeError, json.JSONDecodeError) as exc:
            raise fail(exc) from exc
