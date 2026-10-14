"""Tests precision review on real cached-schema synthetic fixtures without video inference."""
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
from streetlab_phase3.video.sahi_unmatched_detection_review import (
    accepted_prediction_indices, audit_unmatched_detections,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_export(source: Path, image_size: int) -> Path:
    directory = source.with_name("checkpoint_openvino_model")
    directory.mkdir()
    (directory / "model.xml").write_text("<model/>")
    (directory / "model.bin").write_bytes(b"fakeconverted")
    return directory


def make_audit(tmp_path):
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"frozen original")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video test")
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "frame,id,cx,cy,type\n"
        "1,m,10,20,moped\n1,c,100,20,car\n"
        "2,m,15,20,moped\n2,c,105,20,car\n",
        encoding="utf-8")
    exported = export_isolated_openvino(
        weights=weights, output_dir=tmp_path / "openvino",
        image_size=640, exporter=fake_export)
    audit = tmp_path / "audit"
    audit.mkdir()
    observations = {
        0: [
            Detection(0, 10., 20., "MOTORCYCLE", .9),
            Detection(0, 12., 20., "MOTORCYCLE", .7),
            Detection(0, 100., 20., "CAR", .9),
            Detection(0, 10., 20., "CAR", .8),
        ],
        1: [
            Detection(1, 15., 20., "MOTORCYCLE", .9),
            Detection(1, 200., 20., "MOTORCYCLE", .85),
            Detection(1, 105., 20., "CAR", .9),
            Detection(1, 105., 20., "BUS", .6),
        ],
    }
    with truth.open(newline="", encoding="utf-8") as stream:
        ground_truth = normalize_fluid_pixel_truth(csv.DictReader(stream))
    baseline = score_detections(observations, ground_truth, [0, 1])
    baseline["latency_median_s"] = 1.1
    original = {0: [], 1: []}
    reference = score_detections(original, ground_truth, [0, 1])
    reference["latency_median_s"] = .2
    for mode, preds in (("standard", original), ("sliced", observations)):
        with (audit / f"{mode}_detections.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(("frame", "x_px", "y_px", "vehicle_class", "confidence"))
            for frame in (0, 1):
                for item in preds[frame]:
                    writer.writerow((frame, item.x_px, item.y_px,
                                     item.vehicle_class, item.confidence))
    manifest = {
        "status": "EXPERIMENTAL_DETECTION_ONLY",
        "sample_frames": [0, 1],
        "model_sha256": sha(weights),
        "ground_truth_sha256": sha(truth),
        "class_map": {"car": 0, "motorcycle": 3},
        "runtime_backend": "openvino",
        "runtime_model_sha256": exported["export_sha256"],
        "trial": {
            "video": str(video), "fluid_tracks": str(truth),
            "weights": str(weights), "image_size": 640, "device": "cpu",
            "confidence": .15, "frame_offset": 1, "max_pixel_distance": 50.,
            "runtime_model_path": exported["model_path"],
            "class_map_file": None,
            "start_frame": 0, "end_frame": 1, "sample_step": 1,
            "slice_width": 640, "slice_height": 640, "overlap": .20,
        },
        "standard": reference, "sliced": baseline,
    }
    (audit / "report.json").write_text(json.dumps(manifest))
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({
        "status": "PAIRED_DETECTOR_TRUTH_OBSERVATION_AUDIT_NOT_TRACKING",
        "samples": [0, 1],
        "same_checkpoint_sha256": sha(weights),
        "candidate_backend": "openvino",
        "total_annotation_observation_disagreements": 1,
        "by_class": {
            cls: {
                "annotated_vehicle_observation_disagreements":
                ([{"video_frame": 0, "fluid_track_id": "m",
                   "found_by": "pytorch_only"}]
                 if cls == "MOTORCYCLE" else [])
            } for cls in ("CAR", "BUS", "HEAVY_VEHICLE", "MOTORCYCLE")
        },
    }))
    return audit, parity


def test_nearest_matching_diagnostics_do_not_suppress_real_nearby_motos(tmp_path):
    audit, parity = make_audit(tmp_path)
    result = audit_unmatched_detections(audit, parity_file=parity)
    assert result["eligible_for_promotion"] is False
    assert result["annotated_truth_points"] == 4
    assert result["matched_predictions"] == 4
    assert result["unmatched_predictions"] == 4
    assert result["classes"]["MOTORCYCLE"]["unmatched"] == 2
    assert result["classes"]["MOTORCYCLE"]["near_matched_same_class_prediction_25px"] == 1
    assert result["classes"]["MOTORCYCLE"]["confidence_bands"]["[.5,1]"] == 2
    assert result["classes"]["CAR"]["unmatched"] == 1
    assert result["classes"]["CAR"]["near_other_class_truth_50px"] == 1
    assert result["classes"]["BUS"]["unmatched"] == 1
    assert result["classes"]["BUS"]["near_other_class_truth_50px"] == 1
    assert result["parity_disagreements"][0]["annotation_x_px"] == 10
    assert result["review_queue"][0]["review_reason"] == (
        "pytorch_openvino_truth_observation_disagreement")
    assert len(result["unmatched_rows"]) == 4
    assert len(result["review_queue"]) <= 12


def test_frozen_prediction_score_must_agree(tmp_path):
    audit, _ = make_audit(tmp_path)
    path = audit / "report.json"
    source = json.loads(path.read_text())
    source["sliced"]["per_class"]["MOTORCYCLE"]["matched"] += 1
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Cached sliced report differs"):
        audit_unmatched_detections(audit)


def test_parity_provenance_must_agree(tmp_path):
    audit, parity = make_audit(tmp_path)
    raw = json.loads(parity.read_text())
    raw["same_checkpoint_sha256"] = "0" * 64
    parity.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="parity provenance"):
        audit_unmatched_detections(audit, parity_file=parity)


def test_cli_output_is_immutable(tmp_path):
    audit, parity = make_audit(tmp_path)
    cli = (Path(__file__).resolve().parents[1] / "scripts"
           / "phase3_sahi_unmatched_detection_review.py")
    output = tmp_path / "review.json"
    args = [sys.executable, str(cli), "--audit-dir", str(audit),
            "--parity", str(parity), "--output", str(output)]
    p = subprocess.run(args, capture_output=True, text=True, check=True)
    summary = json.loads(p.stdout)
    assert summary["unmatched_predictions"] == 4
    assert summary["classes"]["BUS"]["unmatched"] == 1
    assert output.is_file()
    duplicate = subprocess.run(args, capture_output=True, text=True)
    assert duplicate.returncode != 0
    assert "FileExistsError" in duplicate.stderr

def test_local_visual_gallery_uses_cached_centers_and_preserves_sources(tmp_path):
    import shutil
    import numpy as np
    from streetlab_phase3.video.sahi_detector_visual_review import (
        render_detector_review, select_visual_targets,
    )
    audit, parity = make_audit(tmp_path)
    summary = audit_unmatched_detections(audit, parity_file=parity)
    visual = tmp_path / "review.json"
    visual.write_text(json.dumps(summary), encoding="utf-8")
    ref = tmp_path / "pytorch"
    ref.mkdir()
    source = json.loads((audit / "report.json").read_text())
    source["runtime_backend"] = "pytorch"
    source["runtime_model_sha256"] = source["model_sha256"]
    source["trial"]["runtime_model_path"] = None
    (ref / "report.json").write_text(json.dumps(source))
    for mode in ("sliced", "standard"):
        shutil.copyfile(audit / f"{mode}_detections.csv",
                        ref / f"{mode}_detections.csv")
    class FakeCapture:
        def __init__(self, path):
            self.pos = 0
            self.released = False
            self.seeks = []
        def isOpened(self): return True
        def set(self, _key, frame):
            self.seeks.append(frame)
            self.pos = int(frame)
            return True
        def read(self):
            self.pos += 1
            return True, np.zeros((280, 350, 3), dtype=np.uint8)
        def get(self, _key): return float(self.pos)
        def release(self): self.released = True

    class CV2:
        CAP_PROP_POS_FRAMES = 1
        FONT_HERSHEY_SIMPLEX = 2
        IMWRITE_JPEG_QUALITY = 3
        cap = None
        @classmethod
        def VideoCapture(cls, path):
            cls.cap = FakeCapture(path)
            return cls.cap
        @staticmethod
        def circle(img, center, radius, color, thickness):
            img[0,0,0] = 1
        @staticmethod
        def putText(img, txt, xy, font, scale, color, thickness):
            return img
        @staticmethod
        def resize(img, dim):
            assert dim[0] == img.shape[1] * 2
            assert dim[1] == img.shape[0] * 2
            return np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
        @staticmethod
        def hconcat(images): return np.concatenate(images, axis=1)
        @staticmethod
        def imwrite(path, image, flags):
            assert image.shape[1] >= 1000
            Path(path).write_bytes(b"test-image")
            return True
    selected = select_visual_targets(summary)
    assert any(x["class"] == "BUS" for x in selected)
    assert selected[0]["review_reason"] == (
        "pytorch_openvino_truth_observation_disagreement")
    output = tmp_path / "gallery"
    before = (audit / "sliced_detections.csv").read_bytes()
    result = render_detector_review(
        review_json=visual, reference_dir=ref, openvino_dir=audit,
        output_dir=output, cv2_module=CV2, crop_size=256)
    assert result["eligible_for_promotion"] is False
    assert result["image_count"] >= 5
    assert len(list(output.glob("*.jpg"))) == result["image_count"]
    assert (output / "index.json").is_file()
    assert (audit / "sliced_detections.csv").read_bytes() == before
    assert CV2.cap.released is True
    assert set(CV2.cap.seeks) == {0, 1}
    with pytest.raises(FileExistsError):
        render_detector_review(
            review_json=visual, reference_dir=ref, openvino_dir=audit,
            output_dir=output, cv2_module=CV2, crop_size=256)


def test_local_visual_gallery_refuses_tampered_review_source(tmp_path):
    from streetlab_phase3.video.sahi_detector_visual_review import render_detector_review
    audit, parity = make_audit(tmp_path)
    summary = audit_unmatched_detections(audit, parity_file=parity)
    summary["prediction_source"] = str(tmp_path / "different.csv")
    review = tmp_path / "review.json"
    review.write_text(json.dumps(summary))
    import shutil
    ref = tmp_path / "pytorch"
    ref.mkdir()
    source = json.loads((audit / "report.json").read_text())
    source["runtime_backend"] = "pytorch"
    source["runtime_model_sha256"] = source["model_sha256"]
    source["trial"]["runtime_model_path"] = None
    (ref / "report.json").write_text(json.dumps(source))
    for mode in ("sliced", "standard"):
        shutil.copyfile(audit / f"{mode}_detections.csv",
                        ref / f"{mode}_detections.csv")
    with pytest.raises(ValueError, match="Review is not"):
        render_detector_review(
            review_json=review, reference_dir=ref,
            openvino_dir=audit, output_dir=tmp_path / "wrong")
