"""Independent restartable local CPU worker: python -m streetlab_integration.worker.

HTTP handlers only enqueue jobs. This process owns the GPU/CPU runtime and
persists progress + leases in SQLite, so browser navigation/server reloads
do not interrupt long-running source processing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
import uuid

from streetlab_integration.video_jobs import JobCancelled, VideoStore
from streetlab_integration.primary_runner import run_primary, sha, verified_run

LOG = logging.getLogger("streetlab.worker")


def pinned_model(model_dir: Path, provenance: Path) -> tuple[str, str]:
    """Only the recorded W04 OpenVINO model, never an unpinned replacement."""
    from streetlab_phase3.video.openvino_export import hash_model_tree
    if not provenance.is_file():
        raise ValueError("Frozen model provenance file missing")
    reference = json.loads(provenance.read_text(encoding="utf-8"))
    recorded = reference.get("openvino_model_sha256")
    if (reference.get("eligible_for_promotion") is not False
        or not isinstance(recorded, str) or len(recorded) != 64
        or not model_dir.is_dir() or not list(model_dir.glob("*.xml"))
        or not list(model_dir.glob("*.bin"))):
        raise ValueError("Frozen W04 OpenVINO reference is incomplete")
    actual = hash_model_tree(model_dir)
    if actual != recorded:
        raise ValueError("Worker model does not match the frozen W04 model SHA")
    return actual, sha(provenance)


def _verified_receipt(root: Path, digest: str) -> dict:
    from streetlab_integration.observation_bridge import STATUS
    folder = root / "observations" / digest
    checks = (folder / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    hashes = {}
    for item in checks:
        hex_digest, name = item.split("  ", 1)
        if name not in {"report.json", "track_points_pixels.csv"}:
            raise ValueError("Unexpected observation receipt file")
        hashes[name] = hex_digest
    if set(hashes) != {"report.json", "track_points_pixels.csv"}:
        raise ValueError("Incomplete observation receipt")
    for name, expected in hashes.items():
        file = folder / name
        if file.is_symlink() or sha(file) != expected:
            raise ValueError("Observation receipt integrity failed")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if (report.get("status") != STATUS or report.get("source_tracking_sha256") != digest
        or report.get("data_readiness", {}).get("real_site_simulation_allowed") is not False):
        raise ValueError("Untrusted observation receipt")
    return report


def job_report(store: VideoStore, job_id: str) -> dict:
    job = store.job(job_id)
    if job["status"] != "SUCCEEDED" or not job["report_sha256"]:
        raise ValueError("This job has no completed verified observation")
    return _verified_receipt(store.root, job["report_sha256"])


def run_once(store: VideoStore, model_dir: Path, provenance: Path,
             worker_id: str, *, runner=run_primary) -> bool:
    job = store.claim(worker_id)
    if job is None:
        return False
    try:
        model_hash, provenance_hash = pinned_model(model_dir, provenance)
        source = store.source(job["source_id"])
        path = store.source_path(source)
        if sha(path) != source["sha256"]:
            raise ValueError("Stored source video SHA-256 has changed")
        folder = store.root / "runs" / job["id"]
        config = job["config"]
        def progress(done: int) -> None:
            store.progress(job["id"], worker_id, done)
        if folder.exists():
            manifest = verified_run(folder, source["sha256"])
        else:
            manifest = runner(
                path, folder, model_dir, config, source["metadata"]["fps"],
                source["sha256"], model_hash, progress)
            manifest = verified_run(folder, source["sha256"])
        if manifest.get("model_tree_sha256") != model_hash:
            raise ValueError("Run model SHA differs from the pinned worker model")
        from streetlab_integration.observation_bridge import ingest_track_file
        tracks = folder / "primary_ios030.txt"
        digest = sha(tracks)
        if (store.root / "observations" / digest).exists():
            report = _verified_receipt(store.root, digest)
        else:
            _, report = ingest_track_file(
                tracks, store.root, fps=source["metadata"]["fps"],
                source_label="M2_FROZEN_PRIMARY_OBSERVED_AUTO")
        if report["source_tracking_sha256"] != digest:
            raise ValueError("Imported observation receipt mismatched the native tracking source")
        store.progress(job["id"], worker_id, job["total_frames"])
        store.finish(job["id"], worker_id, "SUCCEEDED", report_sha256=digest)
        LOG.info("Completed job %s with %s source-pixel tracking points",
                 job["id"], report["observed_points"])
    except JobCancelled:
        store.finish(job["id"], worker_id, "CANCELLED",
                     error="Analysis cancelled; source footage was preserved")
        LOG.info("Cancelled job %s", job["id"])
    except Exception:
        LOG.exception("Analysis job %s failed", job["id"])
        # Details (including local path and model internals) go to worker logs,
        # never directly to an HTTP client.
        try:
            store.finish(job["id"], worker_id, "FAILED",
                         error="Video processing failed; check the local worker logs and retry")
        except (ValueError, JobCancelled):
            LOG.exception("Worker lease lost while recording failure")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".streetlab-m5"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frozen-provenance", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 0.2 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 0.2 and 60")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = VideoStore(args.workdir)
    # Fail before accepting any queued work if the operator supplied the wrong model.
    pinned_model(args.model_dir, args.frozen_provenance)
    worker_id = uuid.uuid4().hex
    LOG.info("StreetLab local M2 worker active: %s", worker_id)
    while True:
        busy = run_once(store, args.model_dir, args.frozen_provenance, worker_id)
        if args.once:
            break
        if not busy:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
