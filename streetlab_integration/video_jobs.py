"""SQLite-backed local StreetLab projects, immutable sources, and durable job transitions.

Database rows are metadata only. Video bytes live under UUID-scoped local paths.
The web server does not run CPU inference; a separately started worker claims jobs.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, BinaryIO

CHUNK_BYTES = 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
MAX_VIDEO_SECONDS = 3600
MAX_PENDING = 24
LEASE_SECONDS = 600
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SCHEMA = 1


class JobError(ValueError):
    pass


class JobCancelled(Exception):
    pass


def _uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise JobError("Invalid resource identifier") from exc
    if parsed.hex != value:
        raise JobError("Invalid resource identifier")
    return value


def _now() -> float:
    return time.time()


def _media_header(header: bytes, suffix: str) -> bool:
    if suffix in {".mp4", ".mov"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if suffix == ".avi":
        return header.startswith(b"RIFF") and header[8:12] == b"AVI "
    if suffix == ".mkv":
        return header.startswith(b"\x1a\x45\xdf\xa3")
    return False


def profile_video(path: Path) -> dict[str, Any]:
    """Get source-native metadata, not inferred metrics. Requires real OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise JobError("OpenCV missing; install the M2 video runtime dependencies") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise JobError("Video decoder could not open this source")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
    finally:
        cap.release()
    if (not math.isfinite(fps) or not 1 <= fps <= 120 or frames < 1
        or not 64 <= width <= 7680 or not 64 <= height <= 4320
        or frames / fps > MAX_VIDEO_SECONDS or not ok or frame is None
        or frame.shape[:2] != (height, width)):
        raise JobError("Unsupported FPS, duration, resolution, frame count or undecodable source")
    return {"fps": fps, "frames": frames, "width": width, "height": height,
            "duration_seconds": round(frames / fps, 3)}


class VideoStore:
    def __init__(self, workdir: str | Path):
        self.root = Path(workdir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "video_jobs.sqlite3"
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects(
                  id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sources(
                  id TEXT PRIMARY KEY, project_id TEXT NOT NULL UNIQUE
                    REFERENCES projects(id), relative_path TEXT NOT NULL UNIQUE,
                  sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, metadata TEXT NOT NULL,
                  original_filename TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                  source_id TEXT NOT NULL REFERENCES sources(id), status TEXT NOT NULL,
                  config TEXT NOT NULL, completed_frames INTEGER NOT NULL DEFAULT 0,
                  total_frames INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                  cancel_requested INTEGER NOT NULL DEFAULT 0,
                  lease_until REAL, worker_id TEXT, error TEXT, report_sha256 TEXT,
                  created REAL NOT NULL, updated REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS job_queue ON jobs(status, created);
            """)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(str(self.db), isolation_level=None, timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for field in ("config", "metadata"):
            if field in d:
                d[field] = json.loads(d[field])
        return d

    def create_project(self, name: str) -> dict:
        name = name.strip()
        if not 1 <= len(name) <= 120 or any(ord(c) < 32 for c in name):
            raise JobError("Project name must contain 1–120 printable characters")
        id_ = uuid.uuid4().hex
        with self.connection() as conn:
            conn.execute("INSERT INTO projects VALUES(?,?,?)", (id_, name, _now()))
        return {"id": id_, "name": name}

    def projects(self) -> list[dict]:
        with self.connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT p.id,p.name,p.created,s.id AS source_id,s.metadata AS source_metadata "
                "FROM projects p LEFT JOIN sources s ON s.project_id=p.id "
                "ORDER BY p.created DESC")]
    
    def project(self, project_id: str) -> dict:
        _uuid(project_id)
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise JobError("Project not found")
        return dict(row)

    def source(self, source_id: str) -> dict:
        _uuid(source_id)
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise JobError("Source not found")
        return self._dict(row)

    def save_source(self, project_id: str, filename: str, src: BinaryIO) -> dict:
        self.project(project_id)
        name = Path(filename.replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in EXTENSIONS or not 1 <= len(name) <= 160:
            raise JobError("Supported video formats: MP4, MOV, AVI and MKV")
        with self.connection() as conn:
            if conn.execute("SELECT 1 FROM sources WHERE project_id=?", (project_id,)).fetchone():
                raise JobError("This project already has an immutable video source")
        source_id = uuid.uuid4().hex
        folder = self.root / "projects" / project_id / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / ("." + source_id + ".upload" + suffix)
        dest = folder / (source_id + suffix)
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as output:
                while True:
                    block = src.read(CHUNK_BYTES)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_VIDEO_BYTES:
                        raise JobError("Video exceeds the 2 GiB local ingestion limit")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            with temporary.open("rb") as f:
                if not _media_header(f.read(16), suffix) or size < 16:
                    raise JobError("Video container signature does not match its extension")
            metadata = profile_video(temporary)
            os.replace(temporary, dest)
            rel = dest.relative_to(self.root).as_posix()
            with self.connection() as conn:
                conn.execute(
                    "INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)",
                    (source_id, project_id, rel, digest.hexdigest(), size,
                     json.dumps(metadata), name, _now()))
            return {"id": source_id, "sha256": digest.hexdigest(),
                    "size_bytes": size, "metadata": metadata, "original_filename": name}
        finally:
            temporary.unlink(missing_ok=True)

    def source_path(self, source: dict) -> Path:
        expected = ("projects/" + _uuid(source["project_id"]) + "/sources/" +
                    _uuid(source["id"]) +
                    Path(source["relative_path"]).suffix.lower())
        if source["relative_path"] != expected:
            raise JobError("Untrusted local video location")
        path = self.root / expected
        if path.is_symlink() or not path.is_file() or path.stat().st_size != source["size_bytes"]:
            raise JobError("Stored source is missing or has changed")
        return path

    def queue(self, project_id: str, *, first_frame: int = 0,
              last_frame: int | None = None) -> dict:
        self.project(project_id)
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source_row = conn.execute(
                "SELECT * FROM sources WHERE project_id=?", (project_id,)).fetchone()
            if source_row is None:
                raise JobError("Upload a valid video before starting analysis")
            source = self._dict(source_row)
            if last_frame is None:
                last_frame = source["metadata"]["frames"] - 1
            if (type(first_frame) is not int or type(last_frame) is not int
                or first_frame < 0 or last_frame < first_frame
                or last_frame >= source["metadata"]["frames"]):
                raise JobError("Requested source-frame interval is invalid")
            count = conn.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('QUEUED','RUNNING')").fetchone()[0]
            if count >= MAX_PENDING:
                raise JobError("Local worker queue is full")
            active = conn.execute(
                "SELECT 1 FROM jobs WHERE project_id=? AND status IN ('QUEUED','RUNNING')",
                (project_id,)).fetchone()
            if active:
                raise JobError("This project already has a queued or running analysis")
            job_id = uuid.uuid4().hex
            ts = _now()
            config = {"first_frame": first_frame, "last_frame": last_frame,
                      "policy": "hard_nms_ios_0.30", "warmup_frames": min(30, first_frame),
                      "slice_size": 640, "overlap": 0.20, "confidence": 0.15}
            conn.execute(
                "INSERT INTO jobs(id,project_id,source_id,status,config,total_frames,created,updated) "
                "VALUES(?,?,?,'QUEUED',?,?,?,?)",
                (job_id, project_id, source["id"], json.dumps(config),
                 last_frame - first_frame + 1, ts, ts))
        return self.job(job_id)

    def job(self, job_id: str) -> dict:
        _uuid(job_id)
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobError("Job not found")
        return self._dict(row)

    def jobs(self, project_id: str) -> list[dict]:
        self.project(project_id)
        with self.connection() as conn:
            return [self._dict(r) for r in conn.execute(
                "SELECT * FROM jobs WHERE project_id=? ORDER BY created DESC", (project_id,))]

    def claim(self, worker_id: str) -> dict | None:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = _now()
            # Requeued jobs always restart from the first source frame. Immutable
            # completed run directories are reused by the worker if present.
            conn.execute(
                "UPDATE jobs SET status='QUEUED',worker_id=NULL,lease_until=NULL,"
                "completed_frames=0,updated=? WHERE status='RUNNING' AND lease_until<?",
                (now, now))
            if conn.execute(
                "SELECT 1 FROM jobs WHERE status='RUNNING' AND lease_until>=?",
                (now,)).fetchone():
                return None
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='QUEUED' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE jobs SET status='RUNNING',attempts=attempts+1,worker_id=?,"
                "lease_until=?,cancel_requested=0,updated=? WHERE id=?",
                (worker_id, now + LEASE_SECONDS, now, row["id"]))
            id_ = row["id"]
        return self.job(id_)

    def progress(self, job_id: str, worker_id: str, completed: int) -> bool:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT total_frames,status,cancel_requested,worker_id FROM jobs WHERE id=?",
                (_uuid(job_id),)).fetchone()
            if (not row or row["status"] != "RUNNING" or row["worker_id"] != worker_id):
                raise JobCancelled("Job is no longer owned by this worker")
            if row["cancel_requested"]:
                raise JobCancelled("Cancellation requested")
            if not 0 <= completed <= row["total_frames"]:
                raise JobError("Invalid progress update")
            conn.execute(
                "UPDATE jobs SET completed_frames=?,lease_until=?,updated=? WHERE id=?",
                (completed, _now() + LEASE_SECONDS, _now(), job_id))
        return True

    def cancel(self, job_id: str) -> dict:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (_uuid(job_id),)).fetchone()
            if row is None:
                raise JobError("Job not found")
            if row["status"] == "QUEUED":
                conn.execute("UPDATE jobs SET status='CANCELLED',updated=? WHERE id=?",
                             (_now(), job_id))
            elif row["status"] == "RUNNING":
                conn.execute("UPDATE jobs SET cancel_requested=1,updated=? WHERE id=?",
                             (_now(), job_id))
            else:
                raise JobError("Only active jobs can be cancelled")
        return self.job(job_id)

    def finish(self, job_id: str, worker_id: str, status: str, *,
               report_sha256: str | None = None, error: str | None = None) -> dict:
        if status not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            raise JobError("Invalid terminal job status")
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status,worker_id,cancel_requested,total_frames FROM jobs WHERE id=?",
                (_uuid(job_id),)).fetchone()
            if not row or row["status"] != "RUNNING" or row["worker_id"] != worker_id:
                raise JobError("Job lease is not held")
            if row["cancel_requested"] and status == "SUCCEEDED":
                status, report_sha256 = "CANCELLED", None
            conn.execute(
                "UPDATE jobs SET status=?,report_sha256=?,error=?,worker_id=NULL,"
                "lease_until=NULL,completed_frames=CASE WHEN ?='SUCCEEDED' "
                "THEN total_frames ELSE completed_frames END,updated=? WHERE id=?",
                (status, report_sha256, error, status, _now(), job_id))
        return self.job(job_id)

    def retry(self, job_id: str) -> dict:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (_uuid(job_id),)).fetchone()
            if row is None or row["status"] not in ("FAILED", "CANCELLED"):
                raise JobError("Only failed or cancelled jobs can be retried")
            conn.execute(
                "UPDATE jobs SET status='QUEUED',error=NULL,cancel_requested=0,"
                "completed_frames=0,updated=? WHERE id=?", (_now(), job_id))
        return self.job(job_id)
