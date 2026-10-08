"""An OpenVINO export is a conversion candidate, not a validated new model."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from streetlab_phase3.video.openvino_export import (
    export_isolated_openvino, hash_model_tree, validate_export_source,
)
from streetlab_phase3.video.sahi_detection_audit import DetectorAudit


def fake_exporter(weights: Path, imgsz: int) -> Path:
    assert imgsz == 640
    assert weights.name == "checkpoint.pt"
    assert weights.read_bytes() == b"frozen model weights"
    exported = weights.with_name("checkpoint_openvino_model")
    exported.mkdir()
    (exported / "model.xml").write_text("<model/>", encoding="utf-8")
    (exported / "model.bin").write_bytes(b"fake weights")
    (exported / "metadata.yaml").write_text("names: {0: car}", encoding="utf-8")
    return exported


def test_isolated_openvino_export_is_immutable_and_sha_linked(tmp_path):
    original = tmp_path / "checkpoint.pt"
    original.write_bytes(b"frozen model weights")
    target = tmp_path / "ov_trial"
    report = export_isolated_openvino(
        weights=original, output_dir=target,
        image_size=640, exporter=fake_exporter)
    assert report["eligible_for_promotion"] is False
    assert report["int8"] is False
    assert original.read_bytes() == b"frozen model weights"
    model_dir = Path(report["model_path"])
    assert model_dir.is_dir()
    assert model_dir.name == "checkpoint_openvino_model"
    assert len(hash_model_tree(model_dir)) == 64
    meta = validate_export_source(model_dir, original, 640)
    assert meta["export_sha256"] == report["export_sha256"]
    with pytest.raises(FileExistsError):
        export_isolated_openvino(weights=original, output_dir=target,
                                 image_size=640, exporter=fake_exporter)

    (model_dir / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified"):
        validate_export_source(model_dir, original, 640)


def test_export_fails_closed_on_wrong_checkpoint_and_imgsize(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"frozen model weights")
    report = export_isolated_openvino(
        weights=weights, output_dir=tmp_path / "out",
        image_size=640, exporter=fake_exporter)
    model_dir = Path(report["model_path"])
    weights.write_bytes(b"not the original model")
    with pytest.raises(ValueError, match="does not match"):
        validate_export_source(model_dir, weights, 640)
    with pytest.raises(ValueError, match="does not match"):
        validate_export_source(model_dir, weights, 1280)


def test_detector_audit_uses_optional_runtime_dir_without_altering_checkpoint(tmp_path):
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"frozen model weights")
    created = export_isolated_openvino(
        weights=weights, output_dir=tmp_path / "ov",
        exporter=fake_exporter)
    video, truth = tmp_path / "video.mp4", tmp_path / "truth.csv"
    video.write_bytes(b"video")
    truth.write_text("frame,id,cx,cy,type\n1,m,10,10,moped\n")
    trial = DetectorAudit(
        video=str(video), fluid_tracks=str(truth),
        weights=str(weights), output_dir=str(tmp_path / "audit"),
        start_frame=0, end_frame=0, image_size=640,
        runtime_model_path=created["model_path"])
    trial.validate()
    different = DetectorAudit(
        video=str(video), fluid_tracks=str(truth),
        weights=str(weights), output_dir=str(tmp_path / "other"),
        start_frame=0, end_frame=0, image_size=1280,
        runtime_model_path=created["model_path"])
    with pytest.raises(ValueError, match="does not match"):
        different.validate()
    # Metadata-only checks cannot establish prediction parity; real CPU run must.

def test_detector_audit_passes_openvino_path_and_records_true_runtime_sha(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from streetlab_phase3.video.sahi_detection_audit import run_detector_audit

    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"frozen model weights")
    exported = export_isolated_openvino(
        weights=source, output_dir=tmp_path / "ov",
        exporter=fake_exporter)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    truth = tmp_path / "truth.csv"
    truth.write_text("frame,id,cx,cy,type\n1,m,10,10,moped\n")
    audit = DetectorAudit(
        video=str(video), fluid_tracks=str(truth),
        weights=str(source), output_dir=str(tmp_path / "audit"),
        start_frame=0, end_frame=0, image_size=640,
        runtime_model_path=exported["model_path"])
    received = []
    def loader(path, confidence, device, image_size):
        received.append((path, confidence, device, image_size))
        return object()

    class FakeCapture:
        def __init__(self, path):
            self.pos = 0
        def isOpened(self): return True
        def set(self, _prop, frame): self.pos = int(frame); return True
        def read(self):
            self.pos += 1
            return True, np.zeros((32, 32, 3), dtype=np.uint8)
        def get(self, _prop): return float(self.pos)
        def release(self): pass

    class FakeCV:
        CAP_PROP_POS_FRAMES = 1
        COLOR_BGR2RGB = 2
        VideoCapture = FakeCapture
        @staticmethod
        def cvtColor(arr, _code): return arr

    def motor():
        return SimpleNamespace(
            category=SimpleNamespace(name="motorcycle", id=3),
            bbox=SimpleNamespace(to_xyxy=lambda: (5., 5., 15., 15.)),
            score=SimpleNamespace(value=.8))

    def predict(_im, _model, **kw):
        return SimpleNamespace(object_prediction_list=[motor()])

    out = run_detector_audit(
        audit, model_loader=loader, predictors=(predict, predict),
        cv2_module=FakeCV)
    assert received == [(exported["model_path"], .15, "cpu", 640)]
    assert out["runtime_backend"] == "openvino"
    assert out["runtime_model_sha256"] == exported["export_sha256"]
    assert out["model_sha256"] == exported["source_sha256"]
    assert out["sliced"]["per_class"]["MOTORCYCLE"]["matched"] == 1
    assert out["gate"]["eligible_for_production"] is False
