"""Single primary-lane video inference using existing frozen OpenVINO / SAHI / ByteTrack.

The research shadow lane, FLUID scoring and holdout restrictions are NOT invoked.
No detector/tracker thresholds are optimized here. All output is source-pixel data.
Heavy video dependencies are optional until a real worker starts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Callable

from streetlab_integration.video_jobs import JobCancelled


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run_primary(video: Path, output: Path, model_dir: Path, config: dict,
                fps: float, source_sha256: str, model_sha256: str,
                progress: Callable[[int], None]) -> dict:
    """A new immutable run directory, never an edit of a W04 research artifact."""
    if output.exists():
        raise FileExistsError("Analysis artifact already exists; refusing overwrite")
    if config.get("policy") != "hard_nms_ios_0.30":
        raise ValueError("Unapproved tracking policy")
    if config.get("slice_size") != 640 or config.get("overlap") != .20:
        raise ValueError("Only frozen original tile configuration is accepted")
    start, end = config["first_frame"], config["last_frame"]
    warmup = config["warmup_frames"]
    first = start - warmup
    if first < 0 or end < start:
        raise ValueError("Invalid source frame interval")
    import cv2
    from sahi import AutoDetectionModel
    from supervision import Detections
    from trackers import ByteTrackTracker
    from streetlab_phase3.video.premerge_box_capture import capture_boxes
    from streetlab_phase3.video.unseen_dual_lane import policy_merge, source_rgb
    from streetlab_phase3.video.continuous_box_tracking_lab import (
        as_tracker_detections, _count_ids,
    )
    from streetlab_phase3.video.sahi_tracker_trial import rows_from_tracks

    from streetlab_phase3.video.openvino_export import hash_model_tree
    if hash_model_tree(model_dir) != model_sha256:
        raise ValueError("Pinned OpenVINO model changed")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(model_dir),
        confidence_threshold=.15, device="cpu", image_size=640)
    tracker = ByteTrackTracker(
        track_activation_threshold=.20, high_conf_det_threshold=.15,
        lost_track_buffer=45, minimum_consecutive_frames=2, frame_rate=fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".stage-", dir=output.parent))
    count = 0
    rows_written = 0
    wall_times = []
    previews = []
    elapsed_start = time.perf_counter()
    try:
        tracks = stage / "primary_ios030.txt"
        with tracks.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            for frame, rgb in source_rgb(video, first, end, cv2_module=cv2):
                progress(max(0, frame - start))
                before = time.perf_counter()
                raw, _ = capture_boxes([(frame, rgb)], model, overlap=.20, slice_size=640)
                merged = policy_merge(raw, "primary")
                detections = as_tracker_detections(merged, Detections)
                tracked = tracker.update(detections)
                if frame >= start:
                    if len(tracked.xyxy) == 0 and getattr(tracked, "tracker_id", None) is None:
                        rows = []
                    else:
                        rows, _ = rows_from_tracks(tracked, frame)
                    if _count_ids(tracked) != len(rows):
                        raise ValueError("Source tracker IDs are inconsistent")
                    writer.writerows(rows)
                    rows_written += len(rows)
                    count += 1
                    wall_times.append(time.perf_counter() - before)
                    # Small source-image overlay samples, not synthetic video frames.
                    if frame in {start, (start + end) // 3, (start + end) * 2 // 3, end}:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        for item in rows:
                            _, tid, x, y, w, h = item[:6]
                            p1 = (int(x-w/2), int(y-h/2))
                            p2 = (int(x+w/2), int(y+h/2))
                            cv2.rectangle(bgr, p1, p2, (32, 210, 140), 2)
                            cv2.putText(bgr, str(tid), (p1[0], max(14, p1[1]-4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, (32, 210, 140), 1)
                        file = stage / ("overlay_%08d.jpg" % frame)
                        if not cv2.imwrite(str(file), bgr):
                            raise IOError("Could not write an overlay preview")
                        previews.append(file.name)
                progress(count)
        if count != end - start + 1:
            raise ValueError("Decoding stopped before the requested last source frame")
        if rows_written == 0:
            raise ValueError("No confirmed tracks; check source quality and model suitability")
        if sha(video) != source_sha256:
            raise ValueError("Source video was modified during inference")
        manifest = {
            "schema_version": 1,
            "status": "STREETLAB_M2_OBSERVED_AUTO_PIXEL_ONLY",
            "source_video_sha256": source_sha256,
            "model_tree_sha256": model_sha256,
            "tracking_policy": "hard_nms_ios_0.30",
            "parameters": config,
            "fps": fps,
            "frames_processed": count,
            "native_tracking_rows": rows_written,
            "total_worker_wall_seconds": round(time.perf_counter()-elapsed_start, 3),
            "per_source_frame_compute_median_seconds": round(statistics.median(wall_times), 4),
            "per_source_frame_compute_p95_seconds": round(
                sorted(wall_times)[min(len(wall_times)-1, int(.95*(len(wall_times)-1)))], 4),
            "limitations": [
                "Timing excludes upload and report import; not an end-to-end benchmark.",
                "Tracker IDs are not a census of distinct physical vehicles.",
                "Geometry, road scale, speeds in m/s and turn demand remain NEEDS_DATA.",
                "An image overlay is observational evidence, not a georeferenced reconstruction.",
            ],
            "files": {f.name: sha(f) for f in [tracks, *(stage / name for name in previews)]},
            "overlay_files": previews,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError("Immutable completed run already exists")
        os.replace(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verified_run(folder: Path, source_hash: str) -> dict:
    """Read only expected run paths and independently validate every artifact."""
    if folder.is_symlink():
        raise ValueError("Untrusted run directory")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    previews = manifest.get("overlay_files", [])
    if (manifest.get("schema_version") != 1
        or manifest.get("status") != "STREETLAB_M2_OBSERVED_AUTO_PIXEL_ONLY"
        or manifest.get("source_video_sha256") != source_hash
        or set(files) != {"primary_ios030.txt", *previews}
        or len(previews) > 4):
        raise ValueError("Untrusted run manifest")
    for name, digest in files.items():
        if name != "primary_ios030.txt" and not (
            name.startswith("overlay_") and name.endswith(".jpg")
            and len(name) == len("overlay_00000000.jpg")):
            raise ValueError("Untrusted run file")
        file = folder / name
        if file.is_symlink() or sha(file) != digest:
            raise ValueError("Run artifact checksum failed")
    return manifest
