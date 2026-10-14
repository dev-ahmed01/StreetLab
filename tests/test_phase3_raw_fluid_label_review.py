"""Ensure raw FLUID type audit never guesses class or edits the original CSV."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.video.raw_fluid_label_review import raw_label_audit


def source_files(tmp_path):
    fluid = tmp_path / "original_truth.csv"
    fluid.write_text(
        "frame,id,cx,cy,type\n"
        "1,b,100,100,bus\n1,t,300,100,truck\n"
        "2,l,100,100,van\n2,m,350,200,moped\n",
        encoding="utf-8")
    review = tmp_path / "precision.json"
    review.write_text(json.dumps({
        "status": "EXPLORATORY_DETECTOR_UNMATCHED_REVIEW_NOT_FALSE_POSITIVE_VERDICT",
        "eligible_for_promotion": False,
        "frames": [0, 1],
    }))
    gallery = tmp_path / "index.json"
    entries = [
        {"image_file": "01_frame0_BUS.jpg", "source_frame": 0,
         "class": "BUS", "x_px": 101.0, "y_px": 99.0,
         "review_reason": "unmatched_bus_class_audit"},
        {"image_file": "02_frame1_CAR.jpg", "source_frame": 1,
         "class": "CAR", "x_px": 102.0, "y_px": 98.0,
         "review_reason": "high_confidence_unmatched_car"},
    ]
    gallery.write_text(json.dumps({
        "status": "LOCAL_ONLY_OPENVINO_DETECTOR_VISUAL_REVIEW_NOT_PHYSICAL_FP_LABELS",
        "eligible_for_promotion": False,
        "source_review": str(review),
        "source_annotation_file": str(fluid),
        "image_count": 2, "entries": entries,
    }))
    return gallery, review, fluid


def test_retains_raw_fluid_types_even_if_class_not_scored(tmp_path):
    gallery, review, fluid = source_files(tmp_path)
    before = fluid.read_bytes()
    r = raw_label_audit(gallery_index=gallery, precision_review=review,
                        fluid_tracks=fluid)
    assert r["eligible_for_promotion"] is False
    assert r["case_count"] == 2
    a, b = r["cases"]
    assert a["nearest_raw_fluid_labels"][0]["raw_fluid_type"] == "bus"
    assert a["nearest_raw_fluid_labels"][0]["normalized_fluid_class"] == "BUS"
    assert a["nearest_within_50px"] is True
    assert b["nearest_raw_fluid_labels"][0]["raw_fluid_type"] == "van"
    assert b["nearest_raw_fluid_labels"][0]["normalized_fluid_class"] == "LIGHT_COMMERCIAL"
    assert b["nearest_raw_class_equals_prediction"] is False
    assert len(b["nearest_raw_fluid_labels"]) == 2
    assert fluid.read_bytes() == before


def test_refuses_mismatched_source_provenance(tmp_path):
    gallery, review, fluid = source_files(tmp_path)
    data = json.loads(gallery.read_text())
    data["source_annotation_file"] = str(tmp_path / "different.csv")
    gallery.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        raw_label_audit(gallery_index=gallery, precision_review=review,
                        fluid_tracks=fluid)


def test_cli_does_not_overwrite_original_review(tmp_path):
    gallery, review, fluid = source_files(tmp_path)
    output = tmp_path / "new.json"
    cmd = [sys.executable,
           str(Path(__file__).resolve().parents[1] /
               "scripts" / "phase3_raw_fluid_label_review.py"),
           "--index", str(gallery), "--precision-review", str(review),
           "--fluid-tracks", str(fluid), "--output", str(output)]
    p = subprocess.run(cmd, text=True, capture_output=True, check=True)
    assert json.loads(p.stdout)["cases"][1]["closest_fluid_label"][
        "raw_fluid_type"] == "van"
    assert output.is_file()
    again = subprocess.run(cmd, text=True, capture_output=True)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
