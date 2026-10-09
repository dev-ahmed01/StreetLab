"""Experimental SAHI -> Roboflow ByteTrack -> Geo-trax pixel-track export.

Only public SAHI, Supervision, and Trackers APIs are used. No extra dependency
is imported until the trial actually runs. No parameter or model is promoted.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from streetlab_phase3.video.engine_shootout import load_class_map, sha256_file
from streetlab_phase3.video.sahi_detection_audit import _sahi_loader, _sahi_predictors

@dataclass(frozen=True)
class SahiTrackingTrial:
    video: str
    weights: str
    output: str
    start_frame: int
    end_frame: int
    mode: str = "sliced"
    warmup_frames: int = 90
    confidence: float = 0.15
    image_size: int = 1920
    device: str = "cpu"
    class_map_file: str | None = None
    slice_height: int = 640
    slice_width: int = 640
    overlap: float = 0.20
    track_activation_threshold: float = 0.20
    high_conf_det_threshold: float = 0.15
    lost_track_buffer: int = 45
    minimum_consecutive_frames: int = 2

    def validate(self) -> None:
        if self.mode not in {"standard", "sliced"}:
            raise ValueError("mode must be standard or sliced")
        if self.start_frame < 0 or self.end_frame < self.start_frame or self.warmup_frames < 0:
            raise ValueError("Invalid frame interval or warm-up")
        if not 0 < self.confidence <= 1 or not 0 <= self.overlap < 0.5:
            raise ValueError("Invalid confidence or overlap")
        if self.image_size < 320 or self.slice_height < 128 or self.slice_width < 128:
            raise ValueError("Invalid model or slice dimensions")
        if (not 0 < self.high_conf_det_threshold <=
                self.track_activation_threshold <= 1):
            raise ValueError("Tracker confidence thresholds must be ordered and positive")
        if self.confidence > self.high_conf_det_threshold:
            raise ValueError("Detector floor must be <= low tracking threshold")
        if self.lost_track_buffer < 1 or self.minimum_consecutive_frames < 1:
            raise ValueError("Invalid track lifecycle settings")
        for filename in (self.video, self.weights):
            if not Path(filename).is_file():
                raise FileNotFoundError(f"Local trial input missing: {filename}")
        output = Path(self.output)
        for candidate in (output, output.with_suffix(".manifest.json")):
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite trial evidence: {candidate}")


def sahi_to_detections(
    predictions: Sequence[Any], class_map: Mapping[str, int],
    detection_class: Callable[..., Any],
) -> Any:
    """Full-resolution boxes to sv.Detections without implicit COCO ID mapping."""
    boxes: list[list[float]] = []
    scores: list[float] = []
    classes: list[int] = []
    for item in predictions:
        name = str(item.category.name).strip().lower()
        num = str(item.category.id)
        canonical = class_map.get(name, class_map.get(num))
        if canonical is None:
            continue
        bounds = [float(v) for v in item.bbox.to_xyxy()]
        confidence = float(item.score.value)
        if (len(bounds) != 4 or
            not all(math.isfinite(v) for v in (*bounds, confidence)) or
            bounds[0] >= bounds[2] or bounds[1] >= bounds[3] or
            not 0 <= confidence <= 1):
            raise ValueError("Invalid SAHI detection geometry or confidence")
        boxes.append(bounds)
        scores.append(confidence)
        classes.append(canonical)
    return detection_class(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.asarray(classes, dtype=np.int32),
    )


def rows_from_tracks(tracked: Any, frame: int) -> tuple[list[list[object]], int]:
    """Convert confirmed tracker IDs to frozen 14-column Geo-trax raw-pixel schema."""
    ids = getattr(tracked, "tracker_id", None)
    if ids is None:
        raise ValueError("Tracker did not return identity IDs")
    boxes = np.asarray(tracked.xyxy)
    classes = np.asarray(tracked.class_id)
    confidences = np.asarray(tracked.confidence)
    ids = np.asarray(ids)
    if not (len(boxes) == len(classes) == len(confidences) == len(ids)):
        raise ValueError("Tracker output arrays have inconsistent lengths")
    if len(boxes) and boxes.shape != (len(boxes), 4):
        raise ValueError("Tracker bbox array must have shape (N, 4)")
    output: list[list[object]] = []
    ignored = 0
    seen: set[int] = set()
    for (x1, y1, x2, y2), klass, confidence, tid in zip(
        boxes, classes, confidences, ids
    ):
        if not all(math.isfinite(float(v)) for v in (x1, y1, x2, y2, klass, confidence, tid)):
            raise ValueError("Non-finite tracking result")
        if int(tid) == -1:
            ignored += 1
            continue
        if (int(tid) < 0 or int(tid) in seen or
            int(klass) not in (0, 1, 2, 3) or
            x2 <= x1 or y2 <= y1 or not 0 <= confidence <= 1):
            raise ValueError("Invalid or duplicate confirmed track ID, class, bbox, or score")
        seen.add(int(tid))
        x, y = float((x1 + x2) / 2), float((y1 + y2) / 2)
        w, h = float(x2 - x1), float(y2 - y1)
        output.append([frame, int(tid), x, y, w, h, x, y, w, h,
                       int(klass), float(confidence), w, h])
    return output, ignored


def run_sahi_tracking(
    trial: SahiTrackingTrial,
    *, model_loader: Callable[..., Any] | None = None,
    predictors: tuple[Callable[..., Any], Callable[..., Any]] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    detection_class: Callable[..., Any] | None = None,
    cv2_module: Any = None,
) -> dict[str, Any]:
    trial.validate()
    class_map = load_class_map(trial.class_map_file)
    if model_loader is None:
        model_loader = _sahi_loader
    if predictors is None:
        predictors = _sahi_predictors()
    if tracker_factory is None:
        from trackers import ByteTrackTracker
        tracker_factory = ByteTrackTracker
    if detection_class is None:
        from supervision import Detections
        detection_class = Detections
    if cv2_module is None:
        import cv2
        cv2_module = cv2
    model = model_loader(trial.weights, trial.confidence, trial.device, trial.image_size)
    # Tracker lifecycle uses one update per decoded source frame, including
    # warm-up and frames without detections. Buffer units are tracker updates.
    tracker = tracker_factory(
        track_activation_threshold=trial.track_activation_threshold,
        high_conf_det_threshold=trial.high_conf_det_threshold,
        lost_track_buffer=trial.lost_track_buffer,
        minimum_consecutive_frames=trial.minimum_consecutive_frames,
        frame_rate=30.0,  # 45 lost-buffer units = 45 consecutive calls
    )
    cap = cv2_module.VideoCapture(trial.video)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Video backend could not open {trial.video}")
    first = max(0, trial.start_frame - trial.warmup_frames)
    output = Path(trial.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    frames = predicted_count = confirmed_count = ignored_count = 0
    t0 = time.perf_counter()
    try:
        if first and not cap.set(cv2_module.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError("Video seek to warm-up start failed")
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".txt",
                                          prefix=".sahi-trial-", encoding="utf-8",
                                          dir=output.parent, delete=False) as stream:
            temporary = stream.name
            writer = csv.writer(stream)
            for frame in range(first, trial.end_frame + 1):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Unexpected video EOF at absolute frame {frame}")
                decoded_position = float(cap.get(cv2_module.CAP_PROP_POS_FRAMES))
                if not math.isfinite(decoded_position) or abs(decoded_position - (frame + 1)) > 0.51:
                    raise RuntimeError(
                        f"Video decode/frame alignment failure at {frame}: "
                        f"next position was {decoded_position}")
                frame_rgb = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB)
                if trial.mode == "sliced":
                    result = predictors[1](
                        frame_rgb, model, slice_height=trial.slice_height,
                        slice_width=trial.slice_width,
                        overlap_height_ratio=trial.overlap,
                        overlap_width_ratio=trial.overlap,
                        perform_standard_pred=False, postprocess_type="NMS",
                        postprocess_match_metric="IOU",
                        postprocess_match_threshold=0.5, verbose=0)
                else:
                    result = predictors[0](frame_rgb, model, verbose=0)
                detections = sahi_to_detections(
                    result.object_prediction_list, class_map, detection_class)
                tracked = tracker.update(detections)
                frames += 1
                if frame < trial.start_frame:
                    continue
                predicted_count += len(detections.xyxy)
                rows, ignored = rows_from_tracks(tracked, frame)
                writer.writerows(rows)
                confirmed_count += len(rows)
                ignored_count += ignored
        if confirmed_count == 0:
            raise RuntimeError("No confirmed vehicle tracks; trial is not eligible for comparison")
        versions: dict[str, str] = {}
        for name in ("sahi", "ultralytics", "trackers", "supervision"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "injected-test-double"
        manifest: dict[str, Any] = {
            "provider": "sahi_plus_roboflow_bytetrack",
            "status": "EXPERIMENTAL_NOT_PROMOTED",
            **asdict(trial),
            "model_sha256": sha256_file(trial.weights),
            "versions": versions,
            "evaluation_frame_offset": 1,
            "first_decoded_frame": first,
            "processed_frames": frames,
            "evaluation_frames": trial.end_frame - trial.start_frame + 1,
            "evaluation_raw_detections": predicted_count,
            "evaluation_confirmed_tracks": confirmed_count,
            "evaluation_unconfirmed_detections": ignored_count,
            "tracker_timebase": "one update per actual video frame",
            "tracker_internal_frame_rate": 30.0,
            "stabilized_coordinates": False,
            "geo_referenced": False,
            "elapsed_seconds": time.perf_counter() - t0,
            "class_map": class_map,
        }
        manifest_path = output.with_suffix(".manifest.json")
        if manifest_path.exists() or output.exists():
            raise FileExistsError("Completed output or manifest already exists")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            os.replace(temporary, output)
            temporary = None
        except BaseException:
            manifest_path.unlink(missing_ok=True)
            raise
        return manifest
    finally:
        cap.release()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
