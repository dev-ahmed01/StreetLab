"""Class-agnostic RAW FLUID shadow links must never rewrite class-aware truth."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streetlab_phase3.pixel_benchmark import normalize_fluid_pixel_truth
from streetlab_phase3.video.openvino_export import export_isolated_openvino
from streetlab_phase3.video.sahi_detection_audit import Detection, score_detections
from streetlab_phase3.video.shadow_raw_fluid_ontology import shadow_raw_fluid_ontology


def fake_export(weights, imgsz):
    assert imgsz == 640
    folder = weights.with_name("checkpoint_openvino_model")
    folder.mkdir()
    (folder / "model.xml").write_text("<model/>")
    (folder / "model.bin").write_bytes(b"converted-not-int8")
    return folder


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"frozen")
    truth = tmp_path / "labels.csv"
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "1,original_car,100,50,car\n"
        "1,original_pedestrian,200,50,pedestrian\n"
        "1,original_moped,300,50,moped\n"
        "2,other_car,100,50,car\n"
        "2,other_van,400,50,van\n",
        encoding="utf-8")
    trial = tmp_path / "output"
    trial.mkdir()
    export = export_isolated_openvino(
        weights=weights, output_dir=tmp_path / "openvino",
        image_size=640, exporter=fake_export)
    sliced = {
        0: [
            Detection(0, 100, 50, "BUS", .88),
            Detection(0, 200, 50, "MOTORCYCLE", .72),
            Detection(0, 300, 50, "MOTORCYCLE", .96),
            Detection(0, 700, 50, "CAR", .25),
        ],
        1: [
            Detection(1, 100, 50, "CAR", .91),
            Detection(1, 400, 50, "CAR", .76),
            Detection(1, 700, 50, "HEAVY_VEHICLE", .30),
        ],
    }
    empty = {0: [], 1: []}
    with truth.open(newline="", encoding="utf-8") as fh:
        gt = normalize_fluid_pixel_truth(csv.DictReader(fh))
    results = {mode: score_detections(pred, gt, [0,1])
               for mode,pred in (("standard",empty),("sliced",sliced))}
    for mode, pred in (("standard",empty),("sliced",sliced)):
        with (trial / f"{mode}_detections.csv").open(
            "w",encoding="utf-8",newline=""
        ) as fh:
            w = csv.writer(fh)
            w.writerow(("frame","x_px","y_px","vehicle_class","confidence"))
            for frame in [0,1]:
                for d in pred[frame]:
                    w.writerow((frame,d.x_px,d.y_px,d.vehicle_class,d.confidence))
    payload = {
        "status": "EXPERIMENTAL_DETECTION_ONLY",
        "sample_frames": [0,1],
        "model_sha256": sha(weights),
        "ground_truth_sha256": sha(truth),
        "runtime_backend": "openvino",
        "runtime_model_sha256": export["export_sha256"],
        "trial": {
            "video": "not-needed.mp4", "weights": str(weights),
            "fluid_tracks": str(truth), "confidence": .15,
            "image_size": 640, "slice_width": 640, "slice_height": 640,
            "overlap": .2, "device":"cpu", "class_map_file":None,
            "runtime_model_path": export["model_path"],
            "frame_offset":1,"max_pixel_distance":50.,
            "start_frame":0,"end_frame":1,"sample_step":1,
        },
        "class_map":{"car":0,"motorcycle":3,"bus":1,"truck":2},
        **results,
    }
    (trial / "report.json").write_text(json.dumps(payload))
    return trial,truth


def test_shadow_rescues_only_unused_annotation_centers_and_locks_official(tmp_path):
    audit, truth = make_fixture(tmp_path)
    frozen_bytes = (audit / "report.json").read_bytes()
    truth_bytes = truth.read_bytes()
    r = shadow_raw_fluid_ontology(audit)
    assert r["eligible_for_promotion"] is False
    official = r["original_frozen_class_aware"]
    assert official == {
        "truth":3, "predicted":7, "matched":2, "unmatched":5,
        "precision":2/7, "recall":2/3,
    }
    assert r["additional_shadow_cross_class_spatial_pairs"] == 3
    assert r["pairs_to_labels_outside_original_four_classes"] == 2
    assert sum(r["remaining_without_unused_FLUID_center_within_50px_by_class"].values()) == 2
    tup = {(row["prediction_class"],row["mapped_fluid_class"],row["original_fluid_type"])
           for row in r["shadow_pairs"]}
    assert tup == {("BUS","CAR","car"), ("MOTORCYCLE","PEDESTRIAN","pedestrian"),
                   ("CAR","LIGHT_COMMERCIAL","van")}
    assert r["all_original_fluid_annotations_by_mapped_type"]["PEDESTRIAN"] == 1
    assert (audit/"report.json").read_bytes() == frozen_bytes
    assert truth.read_bytes() == truth_bytes


def test_rejects_corrupted_frozen_detector_score(tmp_path):
    audit, truth = make_fixture(tmp_path)
    p = audit / "report.json"
    obj = json.loads(p.read_text())
    obj["sliced"]["matched_points"] = 3
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="Frozen OpenVINO scoring mismatch"):
        shadow_raw_fluid_ontology(audit)


def test_rejects_modified_annotations_sha(tmp_path):
    audit, truth = make_fixture(tmp_path)
    with truth.open("a") as f:
        f.write("3,extra,1,1,car\n")
    with pytest.raises(ValueError,match="Truth annotations differ"):
        shadow_raw_fluid_ontology(audit)


def test_cli_is_new_output_only(tmp_path):
    audit,_ = make_fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_shadow_raw_fluid_ontology.py"
    output = tmp_path / "shadow.json"
    cmd = [sys.executable,str(script),"--audit-dir",str(audit),"--output",str(output)]
    run = subprocess.run(cmd,check=True,capture_output=True,text=True)
    assert json.loads(run.stdout)["additional_cross_class_center_pairings"] == 3
    assert len(json.loads(output.read_text())["shadow_pairs"]) == 3
    again = subprocess.run(cmd,capture_output=True,text=True)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
