"""Local-only W04 visual audit: annotate EXISTING tracks on decoded CCTV crops.

No detector inference, no cloud upload, no changes to model/tracking scores.
The resulting crops are review prompts, not verified false-positive labels.
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

from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth


def review_targets(report: dict[str, Any], *, max_items: int = 22) -> list[dict[str, Any]]:
    if max_items < 1:
        raise ValueError("max_items must be positive")
    if report.get("status") != "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE":
        raise ValueError("Expected the existing experiment's fixed-window report")
    proximity = report["sliced_unmatched_proximity"]
    paired = report["paired_error_audit"]
    if (proximity["status"] != "UNMATCHED_PROXIMITY_HEURISTIC_NOT_PROOF_OF_DUPLICATES"
        or paired["status"] != "PAIRED_ERROR_AUDIT_TUNING_ONLY_NOT_PRODUCTION_EVIDENCE"):
        raise ValueError("Invalid provenance for visual audit")
    first, last = report["evaluation_start_frame"], report["evaluation_end_frame"]
    if (proximity["video_frame_start"], proximity["video_frame_end"]) != (first, last):
        raise ValueError("Proximity data covers a different frame interval")
    targets = []
    candidates = proximity["examples"]
    high_cars = sorted(
        (x for x in candidates if x["predicted_class"] == "CAR"
         and x["confidence"] >= 0.5),
        key=lambda x: (-x["confidence"], x["video_frame"], x["predicted_track_id"]),
    )
    near_bikes = sorted(
        (x for x in candidates if x["predicted_class"] == "MOTORCYCLE"
         and x["near_matched_prediction"]),
        key=lambda x: (x["video_frame"], x["predicted_track_id"]),
    )
    for group, label in ((high_cars, "high_confidence_unmatched_car"),
                         (near_bikes, "motorcycle_unmatched_near_another_match")):
        for point in group:
            targets.append({
                "reason": label,
                "video_frame": point["video_frame"],
                "x_px": float(point["pixel_center"][0]),
                "y_px": float(point["pixel_center"][1]),
                "predicted_track_id": point["predicted_track_id"],
                "confidence": point["confidence"],
            })
    motorcycle = paired["by_truth_class"]["MOTORCYCLE"]
    # Sample unique truth vehicles rather than 3 repetitions of the same ID.
    for section, reason in (
        ("spatially_rescued_by_sliced", "motorcycle_rescued_by_sliced"),
        ("spatially_lost_by_sliced", "motorcycle_lost_by_sliced"),
    ):
        seen = set()
        for point in motorcycle[section]:
            ident = point["fluid_track_id"]
            if ident in seen:
                continue
            seen.add(ident)
            targets.append({"reason": reason, "video_frame": point["video_frame"],
                            "fluid_track_id": ident})
    selected = targets[:max_items]
    if not selected:
        raise ValueError("No visual review targets in selected interval")
    if any(not first <= x["video_frame"] <= last for x in selected):
        raise ValueError("Review target outside scored interval")
    return selected


def _draw_markers(cv2: Any, crop: Any, locations: list[Any],
                  x0: int, y0: int, *, color: tuple[int, int, int],
                  prefix: str) -> None:
    for p in locations:
        x = int(round(p.x_px)) - x0
        y = int(round(p.y_px)) - y0
        if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
            cv2.circle(crop, (x, y), 5, color, 2)
            cv2.putText(crop, f"{prefix}:{p.vehicle_class[:1]}", (x + 6, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1)


def write_visual_review(
    report: dict[str, Any], *, video: Path, fluid_tracks: Path,
    standard_tracks: Path, sliced_tracks: Path, output_dir: Path,
    max_items: int = 22, crop_size: int = 360, cv2_module: Any = None,
) -> dict[str, Any]:
    if crop_size < 128 or crop_size > 1024:
        raise ValueError("Crop size must be within [128,1024] pixels")
    if output_dir.exists():
        raise FileExistsError(f"Visual audit output exists: {output_dir}")
    targets = review_targets(report, max_items=max_items)
    if (standard_tracks.resolve() != Path(report["results"]["standard"]["tracks"]).resolve()
        or sliced_tracks.resolve() != Path(report["results"]["sliced"]["tracks"]).resolve()):
        raise ValueError("Track paths do not match the comparison report")
    for file in (video, fluid_tracks, standard_tracks, sliced_tracks):
        if not file.is_file():
            raise FileNotFoundError(f"Visual review source not available: {file}")
    if cv2_module is None:
        import cv2
        cv2_module = cv2

    first, last = report["evaluation_start_frame"], report["evaluation_end_frame"]
    with fluid_tracks.open("r", newline="", encoding="utf-8-sig") as fh:
        truth = normalize_fluid_pixel_truth(csv.DictReader(fh))
    truth_by_frame: dict[int, list[Any]] = defaultdict(list)
    for item in truth:
        if first + 1 <= item.frame <= last + 1:
            truth_by_frame[item.frame - 1].append(item)
    pred_by_frame = {}
    for name, filename in (("standard", standard_tracks), ("sliced", sliced_tracks)):
        points = load_geotrax_pixel_tracks(filename)
        grouped: dict[int, list[Any]] = defaultdict(list)
        for point in points:
            if first <= point.frame <= last:
                grouped[point.frame].append(point)
        pred_by_frame[name] = grouped

    # Resolve rescued truth IDs to actual camera coordinates.
    for target in targets:
        if "fluid_track_id" in target:
            hits = [x for x in truth_by_frame[target["video_frame"]]
                    if x.track_id == target["fluid_track_id"]]
            if len(hits) != 1:
                raise ValueError("Rescued/lost truth coordinate is absent or duplicated")
            target["x_px"], target["y_px"] = hits[0].x_px, hits[0].y_px
        if not math.isfinite(target["x_px"]) or not math.isfinite(target["y_px"]):
            raise ValueError("Target pixel coordinates must be finite")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-",
                                      dir=output_dir.parent))
    cap = cv2_module.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        shutil.rmtree(temporary)
        raise RuntimeError(f"Cannot decode review video: {video}")
    snapshots: dict[int, Any] = {}
    entries = []
    try:
        for idx, target in enumerate(targets, start=1):
            frame_id = target["video_frame"]
            if frame_id not in snapshots:
                if not cap.set(cv2_module.CAP_PROP_POS_FRAMES, frame_id):
                    raise RuntimeError(f"Seek failed at frame {frame_id}")
                ok, image = cap.read()
                pos = cap.get(cv2_module.CAP_PROP_POS_FRAMES)
                if not ok or not math.isfinite(pos) or abs(pos - (frame_id + 1)) > .51:
                    raise RuntimeError(f"Video decode misalignment at frame {frame_id}")
                snapshots[frame_id] = image
            frame = snapshots[frame_id]
            h, w = frame.shape[:2]
            cx, cy = int(round(target["x_px"])), int(round(target["y_px"]))
            x0 = max(0, min(cx - crop_size // 2, max(0, w - crop_size)))
            y0 = max(0, min(cy - crop_size // 2, max(0, h - crop_size)))
            raw = frame[y0:min(h, y0 + crop_size), x0:min(w, x0 + crop_size)].copy()
            if raw.size == 0:
                raise ValueError("Review target is outside the physical video frame")
            annotated = raw.copy()
            # BGR: truth gold, standard blue, sliced green, target magenta.
            _draw_markers(cv2_module, annotated, truth_by_frame[frame_id],
                          x0, y0, color=(0, 220, 255), prefix="GT")
            _draw_markers(cv2_module, annotated, pred_by_frame["standard"][frame_id],
                          x0, y0, color=(255, 128, 0), prefix="STD")
            _draw_markers(cv2_module, annotated, pred_by_frame["sliced"][frame_id],
                          x0, y0, color=(0, 240, 0), prefix="SAHI")
            cv2_module.circle(annotated, (cx - x0, cy - y0), 13, (255, 0, 255), 2)
            scale = 2
            sz = (raw.shape[1] * scale, raw.shape[0] * scale)
            left = cv2_module.resize(raw, sz)
            right = cv2_module.resize(annotated, sz)
            side_by_side = cv2_module.hconcat([left, right])
            cv2_module.putText(side_by_side, f"RAW | {target['reason']} | FRAME {frame_id}",
                               (8, 24), cv2_module.FONT_HERSHEY_SIMPLEX,
                               .55, (255, 255, 255), 2)
            name = f"{idx:02d}_frame{frame_id}_{target['reason']}.jpg"
            if not cv2_module.imwrite(str(temporary / name), side_by_side,
                                      [cv2_module.IMWRITE_JPEG_QUALITY, 88]):
                raise RuntimeError(f"Failed writing review image: {name}")
            entries.append({"image_file": name, **target})
        manifest = {
            "status": "LOCAL_ONLY_VISUAL_REVIEW_NOT_GROUND_TRUTH_CORRECTION",
            "eligible_for_promotion": False,
            "video": str(video),
            "fixed_video_frame_window": [first, last],
            "image_count": len(entries),
            "overlay": {
                "GT": "yellow", "STD": "blue",
                "SAHI": "green", "selected_target": "magenta"
            },
            "caution": (
                "Yellow annotations are the user's FLUID labels, not absolute "
                "proof of object presence. Unmatched predictions and close centers "
                "require human interpretation. Crops are local and do not "
                "recompute benchmark metrics."
            ),
            "entries": entries,
        }
        (temporary / "index.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite: {output_dir}")
        os.replace(temporary, output_dir)
        return manifest
    finally:
        cap.release()
        if temporary.exists():
            shutil.rmtree(temporary)
