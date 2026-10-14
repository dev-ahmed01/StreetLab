"""Render local CCTV crops for the existing OpenVINO 21-frame precision audit.

ONLY BGR video decoding and drawing of cached center points. Neither the
Ultralytics detector nor ByteTrack is called. Only user-local JPGs are saved;
no footage or annotations are uploaded. This is NOT raw bounding-box NMS.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
from streetlab_phase3.video.cached_detector_subset import _detections, _read_audit
from streetlab_phase3.video.sahi_detection_audit import CLASSES, score_detections


def select_visual_targets(review: dict[str, Any], *, max_items: int = 16) -> list[dict[str, Any]]:
    if not 5 <= max_items <= 40:
        raise ValueError("max_items must be 5..40")
    if review.get("status") != "EXPLORATORY_DETECTOR_UNMATCHED_REVIEW_NOT_FALSE_POSITIVE_VERDICT":
        raise ValueError("Expected W04 detector-only precision review")
    if review.get("eligible_for_promotion") is not False:
        raise ValueError("Review report must be non-promotion")
    targets = [dict(x) for x in review["review_queue"]]
    # The default 12-case queue may be truncated before *any* BUS examples.
    # Add class-representative unmatched observations, never inferred FPs.
    for klass, desired in (("BUS", 2), ("HEAVY_VEHICLE", 2)):
        have = sum(t["class"] == klass for t in targets)
        extras = sorted(
            (r for r in review["unmatched_rows"] if r["class"] == klass),
            key=lambda r: (-r["confidence"], r["video_frame"], r["x_px"], r["y_px"]),
        )
        seen = {
            (t.get("video_frame"), t.get("x_px"), t.get("y_px"), t["class"])
            for t in targets
        }
        for row in extras:
            key = (row["video_frame"], row["x_px"], row["y_px"], row["class"])
            if have >= desired:
                break
            if key in seen:
                continue
            targets.append({"review_reason": f"unmatched_{klass.lower()}_class_audit", **row})
            seen.add(key)
            have += 1
    # Preserve at least the three current FLUID parity disagreements and a bus
    # even under user-configured output caps.
    important = [t for t in targets
                 if t["review_reason"] == "pytorch_openvino_truth_observation_disagreement"]
    selected = important + [t for t in targets if t not in important]
    return selected[:max_items]


def _draw_centers(cv2: Any, image: Any, rows: list[Any],
                  x0: int, y0: int, color: tuple[int, int, int], label: str) -> None:
    for p in rows:
        x, y = int(round(p.x_px)) - x0, int(round(p.y_px)) - y0
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 4, color, 2)
            cv2.putText(image, f"{label}:{p.vehicle_class[:1]}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .28, color, 1)


def render_detector_review(
    *, review_json: Path, reference_dir: Path, openvino_dir: Path,
    output_dir: Path, max_items: int = 16, crop_size: int = 384,
    cv2_module: Any = None,
) -> dict[str, Any]:
    if not 192 <= crop_size <= 1024:
        raise ValueError("crop_size must be 192..1024")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace local visual evidence: {output_dir}")
    review = json.loads(review_json.read_text(encoding="utf-8"))
    targets = select_visual_targets(review, max_items=max_items)
    ref = _read_audit(reference_dir)
    ov = _read_audit(openvino_dir)
    if ref.get("runtime_backend", "pytorch") != "pytorch" or ov.get("runtime_backend") != "openvino":
        raise ValueError("Expected PyTorch and OpenVINO detector-only audit reports")
    for key in ("video", "fluid_tracks", "weights", "confidence", "image_size",
                "frame_offset", "max_pixel_distance", "slice_height",
                "slice_width", "overlap", "device", "class_map_file"):
        if ref["trial"].get(key) != ov["trial"].get(key):
            raise ValueError(f"Source trial settings diverge: {key}")
    if ref["model_sha256"] != ov["model_sha256"] or ref["ground_truth_sha256"] != ov["ground_truth_sha256"]:
        raise ValueError("Checkpoint or FLUID annotations diverge")
    if ref["class_map"] != ov["class_map"]:
        raise ValueError("Detector class mapping diverges")
    frames = ov["sample_frames"]
    if (not set(frames).issubset(ref["sample_frames"])
        or review["frames"] != frames
        or Path(review["prediction_source"]).resolve()
        != (openvino_dir / "sliced_detections.csv").resolve()):
        raise ValueError("Review is not from the supplied OpenVINO sample")
    if (review["matched_predictions"] != ov["sliced"]["matched_points"]
        or review["unmatched_predictions"] !=
            ov["sliced"]["predicted_points"] - ov["sliced"]["matched_points"]):
        raise ValueError("Precision review totals disagree with OpenVINO sample")
    if not targets:
        raise ValueError("No review targets")

    source_video = Path(ov["trial"]["video"])
    truth_source = Path(ov["trial"]["fluid_tracks"])
    if not source_video.is_file():
        raise FileNotFoundError(f"Original video unavailable: {source_video}")
    if cv2_module is None:
        import cv2
        cv2_module = cv2
    with truth_source.open(encoding="utf-8-sig", newline="") as fh:
        truth = normalize_fluid_pixel_truth(csv.DictReader(fh))
    truth_by_frame = defaultdict(list)
    for p in truth:
        if p.frame - 1 in frames and p.vehicle_class in CLASSES:
            truth_by_frame[p.frame - 1].append(p)
    source_cache = {}
    for label, folder, report in (
        ("pytorch", reference_dir, ref), ("openvino", openvino_dir, ov)
    ):
        preds = _detections(folder, "sliced", report["sample_frames"])
        checked = score_detections(preds, truth, report["sample_frames"])
        for k in ("truth_points", "predicted_points", "matched_points"):
            if checked[k] != report["sliced"][k]:
                raise ValueError(f"Cached {label} predictions do not reproduce {k}")
        source_cache[label] = preds

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-",
                                dir=output_dir.parent))
    cap = None
    try:
        cap = cv2_module.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open local source video: {source_video}")
        decoded = {}
        entries = []
        for idx, case in enumerate(targets, 1):
            frame = case.get("source_frame", case.get("video_frame"))
            if type(frame) is not int or frame not in frames:
                raise ValueError(f"Review case outside source frames: {frame}")
            x = case.get("annotation_x_px", case.get("x_px"))
            y = case.get("annotation_y_px", case.get("y_px"))
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError("Invalid review coordinate fields")
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Non-finite review coordinates")
            if frame not in decoded:
                if not cap.set(cv2_module.CAP_PROP_POS_FRAMES, frame):
                    raise RuntimeError(f"Video seek failed for frame {frame}")
                ok, image = cap.read()
                decoded_position = float(cap.get(cv2_module.CAP_PROP_POS_FRAMES))
                if not ok or not math.isfinite(decoded_position) or abs(decoded_position - frame - 1) > .51:
                    raise RuntimeError(f"Decode frame alignment mismatch at frame {frame}")
                decoded[frame] = image
            image = decoded[frame]
            height, width = image.shape[:2]
            cx, cy = int(round(x)), int(round(y))
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Review target outside source image at {frame}: {(x, y)}")
            x0 = max(0, min(cx - crop_size // 2, max(0, width - crop_size)))
            y0 = max(0, min(cy - crop_size // 2, max(0, height - crop_size)))
            raw = image[y0:min(y0 + crop_size, height),
                        x0:min(x0 + crop_size, width)].copy()
            overlay = raw.copy()
            # OpenCV BGR: annotations gold; PyTorch blue; OpenVINO green.
            _draw_centers(cv2_module, overlay, truth_by_frame[frame], x0, y0,
                          (0, 210, 255), "GT")
            _draw_centers(cv2_module, overlay, source_cache["pytorch"][frame], x0, y0,
                          (255, 130, 0), "PT")
            _draw_centers(cv2_module, overlay, source_cache["openvino"][frame], x0, y0,
                          (0, 240, 0), "OV")
            cv2_module.circle(overlay, (cx - x0, cy - y0), 13, (255, 0, 255), 2)
            # Each JPEG has an unannotated crop on the left; the right side
            # shows the SAME source region with only center-point markers.
            display = cv2_module.hconcat([cv2_module.resize(raw, (raw.shape[1] * 2, raw.shape[0] * 2)),
                                          cv2_module.resize(overlay, (raw.shape[1] * 2, raw.shape[0] * 2))])
            cv2_module.putText(
                display, f"RAW | {case['review_reason']} | FRAME {frame}",
                (8, 23), cv2_module.FONT_HERSHEY_SIMPLEX, .55,
                (255, 255, 255), 2)
            filename = f"{idx:02d}_frame{frame}_{case['class']}.jpg"
            if not cv2_module.imwrite(str(temp / filename), display,
                                      [cv2_module.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"Failed to write visual review JPEG {filename}")
            entries.append({"image_file": filename, "source_frame": frame,
                            "review_reason": case["review_reason"],
                            "class": case["class"], "x_px": x, "y_px": y,
                            "confidence": case.get("confidence"),
                            "fluid_track_id": case.get("fluid_track_id"),
                            "found_by": case.get("found_by")})
        index = {
            "status": "LOCAL_ONLY_OPENVINO_DETECTOR_VISUAL_REVIEW_NOT_PHYSICAL_FP_LABELS",
            "eligible_for_promotion": False,
            "source_video": str(source_video),
            "source_annotation_file": str(truth_source),
            "matched_fluid_offset": 1,
            "source_review": str(review_json),
            "frames": frames,
            "image_count": len(entries),
            "overlay": {
                "left": "unaltered source-video crop",
                "right": "same pixels plus overlay",
                "GT": "gold FLUID annotated center",
                "PT": "blue PyTorch cached detection center",
                "OV": "green OpenVINO cached detection center",
                "selected_case": "magenta ring",
            },
            "limitations": (
                "Only cached detection CENTERS are available, not raw "
                "slice bounding boxes or suppression group IDs. Proximity "
                "does not establish same-object duplicates; class mismatches "
                "may arise from annotations. This is not production labeling, "
                "does not compute new metrics or change tracker/FLUID outputs."
            ),
            "entries": entries,
        }
        (temp / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Refusing to overwrite already existing review output")
        os.replace(temp, output_dir)
        return index
    finally:
        if cap is not None:
            cap.release()
        if temp.exists():
            shutil.rmtree(temp)
