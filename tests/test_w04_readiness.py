"""W04 readiness guards: synthetic fixture bytes only; no real video is scored."""
from __future__ import annotations
import hashlib
import json
import pytest

from streetlab_integration.w04_readiness import inspect_assets, main, file_sha256
from streetlab_phase3.video.openvino_export import hash_model_tree


def make_assets(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-test-video")
    truth = tmp_path / "traj.csv"
    truth.write_bytes(b"synthetic-test-labels")
    model = tmp_path / "model"
    model.mkdir()
    (model / "checkpoint.xml").write_text("synthetic xml", encoding="utf-8")
    (model / "checkpoint.bin").write_bytes(b"synthetic bin")
    provenance = tmp_path / "source_provenance.json"
    provenance.write_text(json.dumps({
        "video_sha256": file_sha256(video),
        "fluid_sha256": file_sha256(truth),
        "openvino_model_sha256": hash_model_tree(model),
        "eligible_for_promotion": False,
        "video_frame_window": [10750, 10950],
    }), encoding="utf-8")
    return video, truth, model, provenance


def test_verified_development_assets_not_a_blind_holdout(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    report = inspect_assets(video=video, model_dir=model, provenance=provenance, truth=truth)
    assert report["status"] == "W04_LOCAL_DEVELOPMENT_ASSETS_VERIFIED"
    assert report["source_video_and_model_verified"] is True
    assert report["matching_fluid_truth_verified"] is True
    assert report["independent_holdout_pass"] is False
    assert report["new_video_inference_performed"] is False
    assert report["eligible_for_production"] is False


def test_altered_video_refused(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    video.write_bytes(b"changed")
    report = inspect_assets(video=video, model_dir=model, provenance=provenance, truth=truth)
    assert report["status"] == "W04_LOCAL_ASSET_AUDIT_BLOCKED"
    assert report["files"]["original_video"]["status"] == "SHA_MISMATCH"
    assert video.read_bytes() == b"changed"


def test_altered_model_refused(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    (model / "checkpoint.bin").write_bytes(b"changed")
    report = inspect_assets(video=video, model_dir=model, provenance=provenance, truth=truth)
    assert report["files"]["openvino_model"]["status"] == "SHA_MISMATCH"
    assert report["source_video_and_model_verified"] is False


def test_missing_truth_does_not_fake_scoring(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    report = inspect_assets(video=video, model_dir=model, provenance=provenance)
    assert report["source_video_and_model_verified"] is True
    assert report["matching_fluid_truth_verified"] is False
    assert report["new_metric_scoring_performed"] is False
    assert "Matching frozen FLUID labels not verified locally" in report["blockers"]


def test_promoted_provenance_refused(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["eligible_for_promotion"] = True
    provenance.write_text(json.dumps(data), encoding="utf-8")
    report = inspect_assets(video=video, model_dir=model, provenance=provenance)
    assert report["status"] == "W04_LOCAL_ASSET_AUDIT_BLOCKED"
    assert report["eligible_for_production"] is False


def test_modified_development_window_refused(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["video_frame_window"] = [0, 200]
    provenance.write_text(json.dumps(data), encoding="utf-8")
    report = inspect_assets(video=video, model_dir=model, provenance=provenance)
    assert report["source_video_and_model_verified"] is False
    assert any("window differs" in item for item in report["blockers"])


def test_fresh_output_only_and_no_source_mutation(tmp_path):
    video, truth, model, provenance = make_assets(tmp_path)
    orig = (file_sha256(video), file_sha256(truth), file_sha256(provenance))
    output = tmp_path / "readiness.json"
    argv = [
        "--video", str(video), "--truth", str(truth),
        "--model-dir", str(model), "--provenance", str(provenance),
        "--output", str(output),
    ]
    assert main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["eligible_for_production"] is False
    assert orig == (file_sha256(video), file_sha256(truth), file_sha256(provenance))
    with pytest.raises(FileExistsError):
        main(argv)
