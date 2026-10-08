"""Read-only audit of RAW FLUID label vocabulary for actual W04 review crops.

Retains original type strings, including classes intentionally omitted by the
calibrated pixel benchmark. This NEVER corrects annotations or alters scores.
Only nearest annotation CENTER distances, not physical object association.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from streetlab_phase3.class_mapping import map_vehicle_class


def raw_label_audit(
    *, gallery_index: Path, precision_review: Path, fluid_tracks: Path,
    neighbors: int = 3,
) -> dict[str, Any]:
    if not 1 <= neighbors <= 10:
        raise ValueError("neighbors must be between 1 and 10")
    gallery = json.loads(gallery_index.read_text(encoding="utf-8"))
    source = json.loads(precision_review.read_text(encoding="utf-8"))
    if (gallery.get("status")
            != "LOCAL_ONLY_OPENVINO_DETECTOR_VISUAL_REVIEW_NOT_PHYSICAL_FP_LABELS"
        or gallery.get("eligible_for_promotion") is not False):
        raise ValueError("Expected unpromoted local W04 visual-review index")
    if (source.get("status")
            != "EXPLORATORY_DETECTOR_UNMATCHED_REVIEW_NOT_FALSE_POSITIVE_VERDICT"
        or source.get("eligible_for_promotion") is not False):
        raise ValueError("Expected unpromoted W04 unmatched-detection audit")
    if (Path(gallery["source_review"]).resolve() != precision_review.resolve()
        or Path(gallery["source_annotation_file"]).resolve()
            != fluid_tracks.resolve()):
        raise ValueError("Gallery provenance differs from selected review or FLUID labels")
    entries = gallery.get("entries")
    if (not isinstance(entries, list) or not entries
        or len(entries) != gallery.get("image_count")):
        raise ValueError("Invalid visual gallery entries")
    frames = set(source["frames"])
    for entry in entries:
        if (type(entry.get("source_frame")) is not int
            or entry["source_frame"] not in frames
            or entry.get("class") not in ("MOTORCYCLE", "CAR", "BUS", "HEAVY_VEHICLE")
            or any(not isinstance(entry.get(k), (int, float))
                   or not math.isfinite(entry[k]) for k in ("x_px", "y_px"))):
            raise ValueError("Invalid review case coordinate or frame")

    target_fluid_frames = {e["source_frame"] + 1 for e in entries}
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with fluid_tracks.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"frame", "id", "cx", "cy", "type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("FLUID CSV lacks required frame/id/cx/cy/type columns")
        for row in reader:
            # Do not normalize/exclude any vehicle family: that's the point.
            frame = int(float(row["frame"]))
            if frame not in target_fluid_frames or not row["cx"] or not row["cy"]:
                continue
            x, y = float(row["cx"]), float(row["cy"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Nonfinite FLUID coordinate in frame {frame}")
            mapped = map_vehicle_class(row["type"])
            rows_by_frame[frame].append({
                "fluid_frame": frame, "fluid_id": str(row["id"]),
                "raw_fluid_type": row["type"],
                "normalized_fluid_class": mapped.canonical.value,
                "x_px": x, "y_px": y,
            })
    records = []
    for entry in entries:
        frame, x, y = entry["source_frame"], entry["x_px"], entry["y_px"]
        sorted_neighbors = sorted(
            (
                {**row,
                 "distance_px": math.hypot(row["x_px"] - x, row["y_px"] - y)}
                for row in rows_by_frame[frame + 1]
            ),
            key=lambda r: (r["distance_px"], r["fluid_id"]),
        )
        close = sorted_neighbors[:neighbors]
        records.append({
            "image_file": entry["image_file"],
            "source_frame": frame,
            "selected_class": entry["class"],
            "selected_x_px": x, "selected_y_px": y,
            "review_reason": entry["review_reason"],
            "nearest_raw_fluid_labels": close,
            "nearest_within_50px": bool(close and close[0]["distance_px"] <= 50),
            "nearest_within_25px": bool(close and close[0]["distance_px"] <= 25),
            "nearest_raw_class_equals_prediction": (
                close[0]["normalized_fluid_class"] == entry["class"]
                if close else None
            ),
        })
    return {
        "status": "W04_RAW_FLUID_LABEL_ONTOLOGY_REVIEW_NOT_RELABELED",
        "eligible_for_promotion": False,
        "source_gallery_index": str(gallery_index),
        "source_precision_review": str(precision_review),
        "original_fluid_tracks": str(fluid_tracks),
        "source_frame_offset": 1,
        "cases": records,
        "case_count": len(records),
        "caution": (
            "The nearest annotation is not proof it belongs to the same physical "
            "vehicle. Original FLUID type strings are preserved exactly; no labels "
            "are overwritten, classes remapped for scoring, or benchmark figures "
            "updated. Cases are cherry-picked from a previously tuned W04 review, "
            "not a representative estimate of annotation incompleteness or "
            "physical false-positive rates."
        ),
    }
