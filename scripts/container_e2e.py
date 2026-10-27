"""Run a genuine container HTTP journey; temporary generated MP4 tests decoder
and queue semantics only, not frozen W04 model inference or field accuracy.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-persisted", action="store_true")
    args = parser.parse_args()
    base = "http://127.0.0.1:8000"
    receipt_path = Path("/app/.streetlab-m5/container_e2e_receipt.json")
    password = os.environ["STREETLAB_ACCESS_PASSWORD"]
    basic = "Basic " + base64.b64encode(("streetlab:" + password).encode()).decode()

    def request(path: str, *, method: str = "GET", body: bytes | None = None,
                headers: dict | None = None, auth: bool = True):
        supplied = {"Accept": "application/json", **(headers or {})}
        if auth:
            supplied["Authorization"] = basic
        req = urllib.request.Request(base + path, data=body, method=method,
                                     headers=supplied)
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read()
            status = error.code
        try:
            return status, json.loads(raw)
        except (ValueError, TypeError):
            return status, raw

    assert request("/api/health", auth=False)[0] == 401
    assert request("/api/health")[1]["status"] == "ok"
    if args.verify_persisted:
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
        pid, jid = saved["project_id"], saved["job_id"]
        assert request("/api/projects/" + pid)[0] == 200
        status, job = request("/api/jobs/" + jid)
        assert status == 200 and job["status"] == "QUEUED", job
        assert request("/api/projects/" + pid + "/workspace")[0] == 200
        print("CONTAINER_RESTART_SQLITE_PERSISTENCE_PASS")
        return

    for page in ("/", "/research", "/reports", "/release", "/fieldwork",
                 "/calibration", "/baseline", "/scenarios"):
        status, _ = request(page)
        assert status == 200, (page, status)
    status, report = request("/api/research/summary")
    assert status == 200, (status, report)
    assert report["chennai"]["holdout_generalization_pass"] is False
    assert report["production_release_approved"] is False
    assert request("/api/research/may25-audit")[0] == 409
    assert request("/api/research/simulation")[0] == 409

    status, created = request("/api/projects", method="POST",
        body=b'{"name":"Container actual MP4 upload check"}',
        headers={"Content-Type": "application/json"})
    assert status == 201, (status, created)
    pid = created["id"]
    upload_path = "/api/projects/" + pid + "/video"
    media_headers = {"x-streetlab-filename": "source.mp4",
                     "Content-Type": "application/octet-stream"}
    assert request(upload_path, method="POST", body=b"not-video",
                   headers=media_headers)[0] == 409

    import cv2
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        video = Path(folder) / "source.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                  10.0, (160, 120))
        assert writer.isOpened(), "MP4 encoder missing"
        try:
            for step in range(12):
                pixels = np.zeros((120, 160, 3), dtype=np.uint8)
                cv2.rectangle(pixels, (15 + step, 36),
                              (60 + step, 72), (255, 255, 255), -1)
                writer.write(pixels)
        finally:
            writer.release()
        payload = video.read_bytes()
    assert payload[4:8] == b"ftyp" and len(payload) > 100
    status, source = request(upload_path, method="POST", body=payload,
                             headers=media_headers)
    assert status == 201, (status, source)
    assert source["metadata"]["frames"] == 12, source
    assert request(upload_path, method="POST", body=payload,
                   headers=media_headers)[0] == 409

    jobs = "/api/projects/" + pid + "/jobs"
    status, job = request(jobs, method="POST", body=b'{"last_frame":11}',
                          headers={"Content-Type": "application/json"})
    assert status == 202, (status, job)
    jid = job["id"]
    assert request(jobs, method="POST", body=b'{}',
                   headers={"Content-Type": "application/json"})[0] == 409
    assert request("/api/jobs/" + jid + "/report")[0] == 409
    status, cancelled = request("/api/jobs/" + jid + "/cancel", method="POST")
    assert status == 200 and cancelled["status"] == "CANCELLED", (status, cancelled)
    status, retried = request("/api/jobs/" + jid + "/retry", method="POST")
    assert status == 202 and retried["status"] == "QUEUED", (status, retried)
    assert request("/api/projects/" + pid + "/workspace")[0] == 200
    receipt_path.write_text(json.dumps({"project_id": pid, "job_id": jid}),
                            encoding="utf-8")
    print("CONTAINER_REAL_MP4_AUTH_PROJECT_QUEUE_CANCEL_RETRY_PASS")


if __name__ == "__main__":
    main()
