"""M2 acceptance tests: persistent jobs, source integrity, failure/cancel/retry,
API/UI wiring and an honest synthetic tracking-output worker contract.

These tests DO NOT claim OpenVINO recall or real-road site reconstruction.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from streetlab_integration import video_jobs, worker
from streetlab_integration.primary_runner import sha
from streetlab_integration.video_jobs import JobCancelled, JobError, VideoStore


VIDEO_BYTES = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
METADATA = {"fps": 30.0, "frames": 90, "width": 320, "height": 240,
            "duration_seconds": 3.0}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(video_jobs, "profile_video", lambda _: dict(METADATA))
    return VideoStore(tmp_path / "product")


def project_source(store: VideoStore):
    project = store.create_project("Real road observation test")
    source = store.save_source(project["id"], "test.mp4", io.BytesIO(VIDEO_BYTES))
    return project, source


def test_upload_is_source_sha_bound_and_non_overwriting(store):
    project, src = project_source(store)
    assert src["sha256"] == sha(store.source_path(store.source(src["id"])))
    assert src["size_bytes"] == len(VIDEO_BYTES)
    assert src["metadata"]["frames"] == 90
    with pytest.raises(JobError, match="already"):
        store.save_source(project["id"], "other.mp4", io.BytesIO(VIDEO_BYTES))
    path = store.source_path(store.source(src["id"]))
    path.write_bytes(VIDEO_BYTES + b"x")
    with pytest.raises(JobError, match="changed"):
        store.source_path(store.source(src["id"]))


def test_invalid_source_header_name_and_project(store):
    project = store.create_project("Junction")
    with pytest.raises(JobError, match="container"):
        store.save_source(project["id"], "wrong.mp4", io.BytesIO(b"x" * 200))
    with pytest.raises(JobError, match="formats"):
        store.save_source(project["id"], "source.exe", io.BytesIO(VIDEO_BYTES))
    with pytest.raises(JobError, match="not found"):
        store.project("a" * 32)
    assert store.projects()[0]["id"] == project["id"]


def test_queue_exclusivity_progress_and_durable_restart(store):
    project, _ = project_source(store)
    job = store.queue(project["id"], first_frame=10, last_frame=24)
    assert job["total_frames"] == 15
    assert job["config"]["policy"] == "hard_nms_ios_0.30"
    assert job["config"]["warmup_frames"] == 10
    with pytest.raises(JobError, match="already"):
        store.queue(project["id"])
    again = VideoStore(store.root)
    leased = again.claim("worker-one")
    assert leased["id"] == job["id"]
    assert leased["status"] == "RUNNING"
    assert again.claim("worker-two") is None
    again.progress(job["id"], "worker-one", 7)
    assert store.job(job["id"])["completed_frames"] == 7
    with pytest.raises(JobCancelled):
        again.progress(job["id"], "worker-two", 8)
    done = again.finish(job["id"], "worker-one", "SUCCEEDED", report_sha256="a" * 64)
    assert done["status"] == "SUCCEEDED"
    assert done["completed_frames"] == 15
    with pytest.raises(JobError, match="Only failed"):
        again.retry(job["id"])


def test_cancel_and_retry_preserve_source_and_enforce_state(store):
    project, src = project_source(store)
    one = store.queue(project["id"])
    assert store.cancel(one["id"])["status"] == "CANCELLED"
    assert store.retry(one["id"])["status"] == "QUEUED"
    leased = store.claim("cpu-one")
    store.cancel(leased["id"])
    with pytest.raises(JobCancelled, match="Cancellation"):
        store.progress(leased["id"], "cpu-one", 1)
    done = store.finish(leased["id"], "cpu-one", "CANCELLED")
    assert done["status"] == "CANCELLED"
    assert store.source_path(store.source(src["id"])).is_file()
    assert store.retry(one["id"])["status"] == "QUEUED"


def test_expired_worker_lease_requeues_without_second_live_owner(store, monkeypatch):
    project, _ = project_source(store)
    job = store.queue(project["id"])
    clock = [1000.]
    monkeypatch.setattr(video_jobs, "_now", lambda: clock[0])
    assert store.claim("old-worker")["attempts"] == 1
    assert store.claim("second-worker") is None
    clock[0] += video_jobs.LEASE_SECONDS + 1
    recovered = store.claim("new-worker")
    assert recovered["id"] == job["id"]
    assert recovered["attempts"] == 2
    assert recovered["completed_frames"] == 0
    with pytest.raises(JobCancelled):
        store.progress(job["id"], "old-worker", 3)


def test_api_starts_jobs_but_never_runs_video_inside_request(store, monkeypatch):
    from streetlab_phase2.api import create_app
    app = create_app(service=object(), observation_workdir=store.root)
    client = TestClient(app)
    assert client.get("/api/projects").json()["projects"] == []
    assert 'id="m2ProjectName"' in client.get("/").text
    project = client.post("/api/projects", json={"name": "BTM signal approach"})
    assert project.status_code == 201
    project_id = project.json()["id"]
    bad = client.post(
        f"/api/projects/{project_id}/video", content=b"invalid data",
        headers={"x-streetlab-filename": "test.mp4"})
    assert bad.status_code == 409
    good = client.post(
        f"/api/projects/{project_id}/video", content=VIDEO_BYTES,
        headers={"x-streetlab-filename": "sample.mp4",
                 "content-type": "application/octet-stream"})
    assert good.status_code == 201
    item = client.get(f"/api/projects/{project_id}").json()
    assert item["source"]["sha256"] == good.json()["sha256"]
    created = client.post(f"/api/projects/{project_id}/jobs", json={"last_frame": 19})
    assert created.status_code == 202
    job = created.json()
    assert job["status"] == "QUEUED"
    assert job["completed_frames"] == 0
    assert job["total_frames"] == 20
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "QUEUED"
    assert client.get(f"/api/projects/{project_id}/jobs").json()["jobs"][0]["id"] == job["id"]
    assert client.get(f"/api/jobs/{job['id']}/report").status_code == 409
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "CANCELLED"
    assert client.post(f"/api/jobs/{job['id']}/retry").json()["status"] == "QUEUED"
    assert client.get("/api/jobs/../../etc/passwd").status_code in (404, 405)


def test_worker_imports_native_tracks_and_keeps_real_site_blocked(store, monkeypatch):
    project, source = project_source(store)
    job = store.queue(project["id"], last_frame=2)
    monkeypatch.setattr(worker, "pinned_model", lambda m, p: ("a" * 64, "b" * 64))
    def fake_inference(video, folder, model_dir, config, fps, input_sha, model_sha, progress):
        # A synthetic native contract test, NOT a real detector accuracy test.
        folder.mkdir(parents=True)
        export = folder / "primary_ios030.txt"
        with export.open("w", newline="") as stream:
            csv.writer(stream).writerows([
                [n, 3, 100+n, 80, 20, 12, 100+n, 80, 20, 12, 0, .9, 20, 12]
                for n in range(3)])
        manifest = {
            "schema_version": 1,
            "status": "STREETLAB_M2_OBSERVED_AUTO_PIXEL_ONLY",
            "source_video_sha256": input_sha,
            "model_tree_sha256": model_sha,
            "files": {export.name: sha(export)},
            "overlay_files": [],
        }
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        progress(3)
        return manifest
    assert worker.run_once(store, Path("unused"), Path("unused"), "cpu-1",
                           runner=fake_inference) is True
    outcome = store.job(job["id"])
    assert outcome["status"] == "SUCCEEDED"
    report = worker.job_report(store, job["id"])
    assert report["observed_points"] == 3
    assert report["tracking_identity_count"] == 1
    assert report["study_gate"]["status"] == "NEEDS_DATA"
    assert report["data_readiness"]["real_site_simulation_allowed"] is False
    assert report["source_coordinate_system"] == "SOURCE_IMAGE_PIXELS_UNCALIBRATED"
    assert worker.run_once(store, Path("unused"), Path("unused"), "cpu-1",
                           runner=fake_inference) is False
    from streetlab_phase2.api import create_app
    client = TestClient(create_app(service=object(), observation_workdir=store.root))
    assert client.get("/api/jobs/" + job["id"] + "/report").json()["observed_points"] == 3
    assert client.get("/api/observations/latest").json()["study_gate"]["status"] == "NEEDS_DATA"
    assert client.get("/api/jobs/" + job["id"] + "/overlays").json() == {"overlay_files": []}
    export = store.root / "runs" / job["id"] / "primary_ios030.txt"
    export.write_text("tampered")
    assert client.get("/api/jobs/" + job["id"] + "/overlays").status_code == 409


def test_opencv_generated_media_decoder_smoke(tmp_path):
    """Optional basic decode smoke. Generated frames are not real-world CV evidence."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "toy.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("CI host does not have MJPG video encoding")
    import numpy as np
    for value in (20, 40, 60):
        writer.write(np.full((240, 320, 3), value, dtype=np.uint8))
    writer.release()
    metadata = video_jobs.profile_video(path)
    assert metadata["width"] == 320
    assert metadata["height"] == 240
    assert metadata["frames"] == 3
    assert metadata["fps"] == pytest.approx(10., rel=.1)
