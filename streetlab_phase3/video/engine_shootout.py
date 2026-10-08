"""Reversible Ultralytics tracker trials with Geo-trax-compatible pixel exports.

The dependency is deliberately optional: importing this module does not import
Ultralytics, torch, OpenCV, or the installed Geo-trax package.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

# Canonical Geo-trax export class IDs. Bicycle is intentionally NOT motorcycle.
DEFAULT_CLASS_MAP: dict[str, int] = {
    "car": 0, "automobile": 0, "bus": 1,
    "truck": 2, "lorry": 2, "heavy vehicle": 2, "heavy_vehicle": 2,
    "motorcycle": 3, "motorbike": 3, "moped": 3, "motor": 3,
}
CANONICAL_CLASSES = {0, 1, 2, 3}


@dataclass(frozen=True)
class Trial:
    video: str
    weights: str
    tracker: str
    output: str
    start_frame: int
    end_frame: int
    warmup_frames: int = 90
    confidence: float = 0.10
    image_size: int = 1920
    device: str = "cpu"
    class_map_file: str | None = None

    def validate(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("Invalid evaluation frame interval")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be nonnegative")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if not 320 <= self.image_size <= 4096:
            raise ValueError("image_size must be between 320 and 4096")
        if not self.weights.strip():
            raise ValueError("Explicit model weights are required")
        if self.tracker not in {"bytetrack.yaml", "botsort.yaml"}:
            tracker = Path(self.tracker)
            if not tracker.is_file() or tracker.suffix.lower() != ".yaml":
                raise ValueError("Tracker must be bytetrack.yaml, botsort.yaml, or an existing YAML file")
        if not Path(self.video).is_file():
            raise FileNotFoundError(f"Video not available: {self.video}")
        if Path(self.output).exists():
            raise FileExistsError(f"Refusing to overwrite a completed trial: {self.output}")


def load_class_map(path: str | None) -> dict[str, int]:
    if path is None:
        return dict(DEFAULT_CLASS_MAP)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("class_map must be a nonempty JSON object")
    parsed: dict[str, int] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip() or type(value) is not int or value not in CANONICAL_CLASSES:
            raise ValueError(f"Invalid class mapping: {name!r} -> {value!r}")
        parsed[name.strip().lower()] = value
    return parsed


def model_class_ids(model_names: Mapping[int, str] | list[str], class_map: Mapping[str, int]) -> dict[int, int]:
    names = model_names.items() if isinstance(model_names, Mapping) else enumerate(model_names)
    mapping = {int(i): class_map[str(name).strip().lower()] for i, name in names
               if str(name).strip().lower() in class_map}
    if not mapping:
        raise ValueError("No model classes map to StreetLab vehicle classes; provide --class-map")
    return mapping


def points_from_boxes(boxes: Any, frame: int, class_ids: Mapping[int, int]) -> list[list[object]]:
    """Convert tracked YOLO xywh boxes to the EXACT 14-column Geo-trax pixel shape.

    A YOLO detection without a persistent tracker ID is omitted rather than
    assigned a fabricated identity. Unmapped classes are omitted.
    """
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    xywh = boxes.xywh.cpu().tolist()
    track_ids = boxes.id.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    if not (len(xywh) == len(track_ids) == len(classes) == len(scores)):
        raise ValueError("Inconsistent YOLO box arrays")
    rows: list[list[object]] = []
    for (x, y, w, h), tid, cls, score in zip(xywh, track_ids, classes, scores):
        canonical = class_ids.get(int(cls))
        if canonical is None:
            continue
        if not all(math.isfinite(float(v)) for v in (x, y, w, h, tid, score)):
            raise ValueError("Non-finite tracker output")
        if w <= 0 or h <= 0 or int(tid) < 0 or not 0 <= float(score) <= 1:
            raise ValueError("Invalid YOLO tracker geometry, ID, or score")
        # Geo-trax P3B schema: frame, track, center, dimensions,
        # stabilized center/dimensions, class, confidence, dimensions.
        rows.append([frame, int(tid), x, y, w, h, x, y, w, h,
                     canonical, score, w, h])
    return rows


def sha256_file(path: str) -> str | None:
    file = Path(path)
    if not file.is_file():
        return None  # named weights may be resolved/downloaded by Ultralytics
    digest = hashlib.sha256()
    with file.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_trial(trial: Trial, *, model_factory: Any = None, cv2_module: Any = None) -> dict[str, object]:
    """Run one independent video window. No output is published on failure."""
    trial.validate()
    class_map = load_class_map(trial.class_map_file)
    if model_factory is None:
        from ultralytics import YOLO  # optional external engine
        model_factory = YOLO
    if cv2_module is None:
        import cv2
        cv2_module = cv2
    model = model_factory(trial.weights)
    ids = model_class_ids(model.names, class_map)
    cap = cv2_module.VideoCapture(trial.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {trial.video}")
    first = max(0, trial.start_frame - trial.warmup_frames)
    output = Path(trial.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    processed = exported = 0
    try:
        # One seek only. Every subsequent tracker update sees consecutive frames.
        if not cap.set(cv2_module.CAP_PROP_POS_FRAMES, first):
            if first != 0:
                raise RuntimeError("Video backend cannot seek to warm-up start")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix=".trial-",
                                          dir=output.parent, newline="", delete=False,
                                          encoding="utf-8") as tmp:
            temp_path = tmp.name
            writer = csv.writer(tmp)
            frame = first
            while frame <= trial.end_frame:
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"Unexpected EOF at frame {frame}; expected through {trial.end_frame}")
                results = model.track(img, persist=True, tracker=trial.tracker,
                                      conf=trial.confidence, imgsz=trial.image_size,
                                      device=trial.device, classes=sorted(ids),
                                      verbose=False)
                if not results:
                    raise RuntimeError(f"YOLO returned no result for frame {frame}")
                processed += 1
                if frame >= trial.start_frame:
                    rows = points_from_boxes(results[0].boxes, frame, ids)
                    writer.writerows(rows)
                    exported += len(rows)
                frame += 1
        if exported == 0:
            raise RuntimeError("No supported tracked vehicles; candidate not eligible for scoring")
        # Validate expected window, retain a unique run provenance manifest.
        import importlib.metadata
        try:
            ul_version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError:
            ul_version = "injected-test-double"
        manifest: dict[str, object] = {
            "provider": "ultralytics_external_trial",
            "status": "EXPERIMENTAL_NOT_PROMOTED",
            **asdict(trial),
            "ultralytics_version": ul_version,
            "weights_sha256": sha256_file(trial.weights),
            "effective_class_ids": {str(k): v for k, v in ids.items()},
            "first_processed_frame": first,
            "processed_frames": processed,
            "exported_points": exported,
            "pixel_coordinate_origin": "absolute_video_frame_zero_based",
            "stabilized_coordinates": False,
            "geo_referenced": False,
        }
        manifest_file = output.with_suffix(".manifest.json")
        if manifest_file.exists():
            raise FileExistsError(f"Manifest exists: {manifest_file}")
        # Publish manifest first then track file; a failed run cannot overwrite a prior trial.
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_path, output)
        temp_path = None
        return manifest
    finally:
        cap.release()
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def promotion_gate(baseline: Mapping[str, object], candidate: Mapping[str, object],
                   baseline_identity: Mapping[str, object] | None = None,
                   candidate_identity: Mapping[str, object] | None = None) -> dict[str, object]:
    """Fail closed: exact scoring window and frame offset + metric guardrails."""
    reasons: list[str] = []
    for key in ("truth_points", "truth_tracks", "evaluation_frame_start",
                "evaluation_frame_end", "max_distance_px"):
        if baseline.get(key) != candidate.get(key):
            reasons.append(f"{key}: not the same evaluation cohort")
    if baseline.get("frame_offset") != 1 or candidate.get("frame_offset") != 1:
        reasons.append("frame offset must be frozen at +1")
    for key, tolerance, higher_better in (
        ("point_recall", 0.0, True), ("point_precision", 0.01, True),
        ("track_coverage", 0.01, True), ("class_agreement", 0.01, True),
        ("pixel_mae", 1.0, False)):
        base, new = baseline.get(key), candidate.get(key)
        if not isinstance(base, (int, float)) or not isinstance(new, (int, float)) or not math.isfinite(new):
            reasons.append(f"{key}: missing or nonnumeric")
        elif (new < base - tolerance if higher_better else new > base + tolerance):
            reasons.append(f"{key}: regression beyond guardrail")
    if baseline_identity is None or candidate_identity is None:
        reasons.append("identity benchmark evidence missing")
    else:
        key = "fraction_truth_tracks_fragmented"
        a, b = baseline_identity.get(key), candidate_identity.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            reasons.append("identity fragmentation data missing")
        elif b > a + 0.02:
            reasons.append("identity fragmentation regression")
        for key in ("truth_tracks", "max_distance_px", "frame_offset"):
            if baseline_identity.get(key) != candidate_identity.get(key):
                reasons.append(f"identity {key}: not the same cohort")
    # Require a measurable win, not merely no decline.
    a, b = baseline.get("point_recall"), candidate.get("point_recall")
    ca, cb = baseline_identity or {}, candidate_identity or {}
    identity_win = isinstance(ca.get("fraction_truth_tracks_fragmented"), (int, float)) and isinstance(cb.get("fraction_truth_tracks_fragmented"), (int, float)) and cb["fraction_truth_tracks_fragmented"] < ca["fraction_truth_tracks_fragmented"] - 0.01
    recall_win = isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a + 0.01
    if not recall_win and not identity_win:
        reasons.append("no meaningful (>1 pp) recall or fragmentation improvement")
    return {"eligible_for_promotion": not reasons, "reasons": reasons,
            "reproduced_comparison": not any("cohort" in x or "offset" in x for x in reasons)}
