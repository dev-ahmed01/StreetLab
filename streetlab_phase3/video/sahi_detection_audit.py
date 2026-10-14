"""Detection-only comparison of SAHI slicing vs ordinary aerial YOLO inference.

No track identities are fabricated, no Geo-trax baseline is modified, and
truth denominators come from a predeclared set of frames rather than detections.
SAHI/OpenCV/Ultralytics are optional runtime dependencies.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from streetlab_phase3.pixel_benchmark import FluidPixelTruth, normalize_fluid_pixel_truth
from streetlab_phase3.video.engine_shootout import load_class_map, sha256_file
from streetlab_phase3.video.openvino_export import validate_export_source, hash_model_tree

CLASSES = ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
CANONICAL_IDS = {0: "CAR", 1: "BUS", 2: "HEAVY_VEHICLE", 3: "MOTORCYCLE"}


@dataclass(frozen=True)
class DetectorAudit:
    video: str
    fluid_tracks: str
    weights: str
    output_dir: str
    start_frame: int
    end_frame: int
    sample_step: int = 10
    frame_offset: int = 1
    confidence: float = 0.15
    image_size: int = 1920
    device: str = "cpu"
    class_map_file: str | None = None
    slice_height: int = 640
    slice_width: int = 640
    overlap: float = 0.20
    max_pixel_distance: float = 50.0
    runtime_model_path: str | None = None

    def validate(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame or self.sample_step < 1:
            raise ValueError("Invalid fixed sampled frame window or step")
        if self.frame_offset != 1 or self.max_pixel_distance != 50.0:
            raise ValueError("FLUID comparison requires frozen +1 offset and 50px distance")
        if not 0 < self.confidence <= 1 or not 0 <= self.overlap < 0.5:
            raise ValueError("Invalid confidence or slice overlap")
        if self.image_size < 320 or self.slice_height < 128 or self.slice_width < 128:
            raise ValueError("Invalid detector or slice size")
        for path in (self.video, self.fluid_tracks, self.weights):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Local input not found: {path}")
        if Path(self.output_dir).exists():
            raise FileExistsError(f"Audit directory already exists: {self.output_dir}")
        if self.runtime_model_path is not None:
            validate_export_source(Path(self.runtime_model_path),
                                   Path(self.weights), self.image_size)

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(range(self.start_frame, self.end_frame + 1, self.sample_step))


@dataclass(frozen=True, slots=True)
class Detection:
    frame: int
    x_px: float
    y_px: float
    vehicle_class: str
    confidence: float


def convert_sahi_predictions(predictions: Sequence[Any], frame: int,
                             class_map: Mapping[str, int]) -> list[Detection]:
    result: list[Detection] = []
    for item in predictions:
        category = item.category
        name_key = str(category.name).strip().lower()
        index_key = str(category.id)
        # Numeric IDs never apply implicitly to COCO. Only an explicitly supplied
        # class map with numeric keys can enable the Geo-trax four-class ontology.
        label = class_map.get(name_key, class_map.get(index_key))
        if label is None:
            continue
        bbox = item.bbox.to_xyxy()
        if len(bbox) != 4:
            raise ValueError("SAHI bbox must have four coordinates")
        left, top, right, bottom = (float(x) for x in bbox)
        score = float(item.score.value)
        if (not all(math.isfinite(v) for v in (left, top, right, bottom, score))
                or left >= right or top >= bottom or not 0 <= score <= 1):
            raise ValueError("Invalid SAHI prediction coordinates or score")
        result.append(Detection(frame, (left + right) / 2, (top + bottom) / 2,
                                CANONICAL_IDS[label], score))
    return result


def _accepted_matches(predicted: Sequence[Detection], truth: Sequence[FluidPixelTruth],
                      max_distance_px: float) -> list[float]:
    """Class-aware maximum-cardinality spatial matching within a fixed gate.

    Dummy nodes make a valid in-gate match preferable to leaving two unmatched
    objects; unlike post-filtering ordinary Hungarian distance cost, this
    cannot sacrifice a valid pair to reduce the sum of other pair distances.
    """
    if not predicted or not truth:
        return []
    p = np.asarray([(d.x_px, d.y_px) for d in predicted], dtype=float)
    t = np.asarray([(d.x_px, d.y_px) for d in truth], dtype=float)
    distances = np.sqrt(np.sum((p[:, None, :] - t[None, :, :]) ** 2, axis=2))
    n, m = distances.shape
    penalty = max_distance_px + 1.0
    impossible = (n + m + 1) * penalty * 4.0
    cost = np.full((n + m, n + m), impossible, dtype=float)
    cost[:n, :m] = np.where(distances <= max_distance_px, distances, impossible)
    cost[:n, m:] = penalty
    cost[n:, :m] = penalty
    cost[n:, m:] = 0.0
    ii, jj = linear_sum_assignment(cost)
    return [float(distances[i, j]) for i, j in zip(ii, jj)
            if i < n and j < m and distances[i, j] <= max_distance_px]


def score_detections(predictions: Mapping[int, Sequence[Detection]],
                     truth: Sequence[FluidPixelTruth], frames: Sequence[int],
                     *, frame_offset: int = 1,
                     max_distance_px: float = 50.0) -> dict[str, Any]:
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("Sampled frames must be a nonempty set of unique positions")
    truth_by_frame_class: dict[tuple[int, str], list[FluidPixelTruth]] = defaultdict(list)
    valid_truth_frames = {frame + frame_offset for frame in frames}
    for item in truth:
        if item.frame in valid_truth_frames and item.vehicle_class in CLASSES:
            truth_by_frame_class[(item.frame, item.vehicle_class)].append(item)
    pred_by_frame_class: dict[tuple[int, str], list[Detection]] = defaultdict(list)
    for frame, detections in predictions.items():
        if frame not in frames:
            raise ValueError(f"Out-of-sample detections in frame {frame}")
        for detection in detections:
            if detection.frame != frame or detection.vehicle_class not in CLASSES:
                raise ValueError("Inconsistent detection frame or class")
            pred_by_frame_class[(frame, detection.vehicle_class)].append(detection)
    totals = {"truth": 0, "predicted": 0, "matched": 0}
    per_class: dict[str, dict[str, Any]] = {}
    for canonical in CLASSES:
        gt = pred = matched = 0
        errors: list[float] = []
        for frame in frames:
            gt_points = truth_by_frame_class[(frame + frame_offset, canonical)]
            detections = pred_by_frame_class[(frame, canonical)]
            gt += len(gt_points)
            pred += len(detections)
            errors.extend(_accepted_matches(detections, gt_points, max_distance_px))
        matched = len(errors)
        totals["truth"] += gt
        totals["predicted"] += pred
        totals["matched"] += matched
        per_class[canonical] = {
            "truth": gt, "predicted": pred, "matched": matched,
            "false_positives": pred - matched, "misses": gt - matched,
            "recall": matched / gt if gt else None,
            "precision": matched / pred if pred else None,
        }
    return {
        "sampled_frames": len(frames),
        "truth_points": totals["truth"],
        "predicted_points": totals["predicted"],
        "matched_points": totals["matched"],
        "recall": totals["matched"] / totals["truth"] if totals["truth"] else None,
        "precision": totals["matched"] / totals["predicted"] if totals["predicted"] else None,
        "per_class": per_class,
        "frame_offset": frame_offset,
        "max_pixel_distance": max_distance_px,
        "track_identity_available": False,
    }


def detector_gate(control: Mapping[str, Any], sliced: Mapping[str, Any],
                  *, max_latency_ratio: float = 5.0) -> dict[str, Any]:
    """Allows progression to a tracking trial; never promotes sliced inference."""
    reasons = []
    for key in ("sampled_frames", "truth_points", "frame_offset", "max_pixel_distance"):
        if control.get(key) != sliced.get(key):
            reasons.append(f"{key} differs between cohorts")
    for label in CLASSES:
        if control.get("per_class", {}).get(label, {}).get("truth") != sliced.get("per_class", {}).get(label, {}).get("truth"):
            reasons.append(f"{label}: different truth cohort")
    def finite(x: Any) -> bool:
        return type(x) in (float, int) and math.isfinite(x)
    for label, a, b, tolerance in (
        ("overall precision", control.get("precision"), sliced.get("precision"), -0.02),
        ("overall recall", control.get("recall"), sliced.get("recall"), -0.01),
        ("car recall", control.get("per_class", {}).get("CAR", {}).get("recall"),
         sliced.get("per_class", {}).get("CAR", {}).get("recall"), -0.02),
    ):
        if not finite(a) or not finite(b) or b < a + tolerance:
            reasons.append(f"{label}: unavailable or outside regression limit")
    old_mc = control.get("per_class", {}).get("MOTORCYCLE", {}).get("recall")
    new_mc = sliced.get("per_class", {}).get("MOTORCYCLE", {}).get("recall")
    if not finite(old_mc) or not finite(new_mc) or new_mc < old_mc + 0.02:
        reasons.append("motorcycle recall did not improve by >=2 percentage points")
    lat_standard, lat_sliced = control.get("latency_median_s"), sliced.get("latency_median_s")
    if not finite(lat_standard) or not finite(lat_sliced) or lat_standard <= 0:
        reasons.append("latency comparison is missing")
    elif lat_sliced / lat_standard > max_latency_ratio:
        reasons.append("sliced inference exceeds latency budget")
    return {"eligible_for_tracking_trial": not reasons,
            "eligible_for_production": False,
            "reasons": reasons}


def _sahi_loader(weights: str, confidence: float, device: str, image_size: int) -> Any:
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights,
        confidence_threshold=confidence, device=device, image_size=image_size)


def _sahi_predictors() -> tuple[Callable[..., Any], Callable[..., Any]]:
    from sahi.predict import get_prediction, get_sliced_prediction
    return get_prediction, get_sliced_prediction


def run_detector_audit(
    trial: DetectorAudit, *,
    model_loader: Callable[..., Any] | None = None,
    predictors: tuple[Callable[..., Any], Callable[..., Any]] | None = None,
    cv2_module: Any = None,
    timer: Callable[[], float] | None = None,
) -> dict[str, Any]:
    trial.validate()
    class_map = load_class_map(trial.class_map_file)
    with Path(trial.fluid_tracks).open(encoding="utf-8-sig", newline="") as fh:
        truth = normalize_fluid_pixel_truth(csv.DictReader(fh))
    frames = trial.frames
    if not any(x.vehicle_class == "MOTORCYCLE" and x.frame in {f + 1 for f in frames} for x in truth):
        raise ValueError("No annotated motorcycles in selected frames: trial cannot assess motorcycle recall")
    if model_loader is None:
        model_loader = _sahi_loader
    if predictors is None:
        predictors = _sahi_predictors()
    if cv2_module is None:
        import cv2
        cv2_module = cv2
    if timer is None:
        timer = time.perf_counter
    runtime_path = trial.runtime_model_path or trial.weights
    model = model_loader(runtime_path, trial.confidence, trial.device, trial.image_size)
    standard_fn, sliced_fn = predictors
    cap = cv2_module.VideoCapture(trial.video)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open video for detector audit")
    results: dict[str, dict[int, list[Detection]]] = {"standard": {}, "sliced": {}}
    timings: dict[str, list[float]] = {"standard": [], "sliced": []}
    try:
        for frame in frames:
            if not cap.set(cv2_module.CAP_PROP_POS_FRAMES, frame):
                raise RuntimeError(f"OpenCV failed seeking to frame {frame}")
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Missing requested sample frame {frame}")
            position = cap.get(cv2_module.CAP_PROP_POS_FRAMES)
            if abs(float(position) - (frame + 1)) > 0.51:
                raise RuntimeError(f"Decode misalignment at frame {frame}: got position {position}")
            # OpenCV decodes BGR but SAHI expects RGB for ndarray sources.
            rgb = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB)
            # Alternate evaluation order to reduce warmup/order advantage.
            modes = ("standard", "sliced") if frame % 2 == 0 else ("sliced", "standard")
            for mode in modes:
                begin = timer()
                if mode == "standard":
                    response = standard_fn(rgb, model, verbose=0)
                else:
                    response = sliced_fn(
                        rgb, model,
                        slice_height=trial.slice_height, slice_width=trial.slice_width,
                        overlap_height_ratio=trial.overlap,
                        overlap_width_ratio=trial.overlap,
                        perform_standard_pred=False,
                        postprocess_type="NMS", postprocess_match_metric="IOU",
                        postprocess_match_threshold=0.5, verbose=0,
                    )
                elapsed = timer() - begin
                if elapsed < 0:
                    raise RuntimeError("Nonmonotonic timer observed")
                timings[mode].append(elapsed)
                results[mode][frame] = convert_sahi_predictions(
                    response.object_prediction_list, frame, class_map)
    finally:
        cap.release()
    scored = {mode: score_detections(detections, truth, frames, frame_offset=1)
              for mode, detections in results.items()}
    for mode in ("standard", "sliced"):
        scored[mode]["latency_median_s"] = statistics.median(timings[mode])
        scored[mode]["latency_mean_s"] = statistics.mean(timings[mode])
    gate = detector_gate(scored["standard"], scored["sliced"])
    versions = {}
    for name in ("sahi", "ultralytics", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed / mocked"
    payload: dict[str, Any] = {
        "status": "EXPERIMENTAL_DETECTION_ONLY",
        "trial": asdict(trial),
        "sample_frames": list(frames),
        "model_sha256": sha256_file(trial.weights),
        "ground_truth_sha256": sha256_file(trial.fluid_tracks),
        "versions": versions,
        "class_map": class_map,
        "runtime_backend": "openvino" if trial.runtime_model_path else "pytorch",
        "runtime_model_sha256": (
            hash_model_tree(Path(trial.runtime_model_path))
            if trial.runtime_model_path else sha256_file(trial.weights)
        ),
        "standard": scored["standard"],
        "sliced": scored["sliced"],
        "gate": gate,
    }
    # Assemble all evidence in a unique temporary directory, then atomically
    # rename it only after successful completion. A failed export is discarded.
    output = Path(trial.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",
                                      dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        for mode in ("standard", "sliced"):
            with (temporary / f"{mode}_detections.csv").open(
                "w", encoding="utf-8", newline=""
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow(("frame", "x_px", "y_px", "vehicle_class", "confidence"))
                for frame in frames:
                    for point in results[mode][frame]:
                        writer.writerow((point.frame, point.x_px, point.y_px,
                                         point.vehicle_class, point.confidence))
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return payload
