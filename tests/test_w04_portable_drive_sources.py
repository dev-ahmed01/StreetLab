"""Synthetic-only portable Google Drive source remapping security tests.

The real May25 MP4 is user-private and is NEVER checked into Git.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from streetlab_phase3.video.holdout_validation import verify_batch
from streetlab_integration.w04_visual_review_packet import (
    ReviewError, check_sources,
)
from test_phase3_holdout_validation import fixture_batch
from test_w04_visual_review_packet import original_receipt


def _copies(tmp_path, video, truth):
    mirror = tmp_path / "drive_mirror"
    mirror.mkdir()
    copied_video = mirror / "20250525_video.mp4"
    copied_fluid = mirror / "20250525_video_Traj.csv"
    shutil.copyfile(video, copied_video)
    shutil.copyfile(truth, copied_fluid)
    return {"video_path": copied_video, "fluid_path": copied_fluid}


def test_frozen_batch_verification_accepts_original_bytes_at_drive_paths(tmp_path):
    folder, video, truth = fixture_batch(tmp_path, "original")
    relocated = _copies(tmp_path, video, truth)
    video.unlink()
    truth.unlink()
    with pytest.raises(ValueError, match="Original source missing or altered"):
        verify_batch(folder)
    proof = verify_batch(folder, original_paths=relocated)
    assert proof["original_source_sha256_verified"] is True
    assert proof["provenance"]["video_path"] == str(video)
    assert proof["provenance"]["fluid_path"] == str(truth)
    assert json.loads((folder/"source_provenance.json").read_text())[
        "video_path"] == str(video)


def test_both_verified_original_files_required_and_tampering_denied(tmp_path):
    folder, video, truth = fixture_batch(tmp_path, "original")
    sources = _copies(tmp_path, video, truth)
    with pytest.raises(ValueError, match="BOTH"):
        verify_batch(folder, original_paths={"video_path": sources["video_path"]})
    with pytest.raises(ValueError, match="BOTH"):
        verify_batch(folder, verify_originals=False, original_paths=sources)
    sources["video_path"].write_bytes(b"different source")
    with pytest.raises(ValueError, match="Original source missing or altered"):
        verify_batch(folder, original_paths=sources)


def test_original_forensic_queue_accepts_mirrored_unchanged_bytes(tmp_path):
    run, review, video, truth = original_receipt(tmp_path)
    sources = _copies(tmp_path, video, truth)
    video.unlink()
    truth.unlink()
    with pytest.raises(ReviewError, match="Original May25"):
        check_sources(run_dir=run, forensics_dir=review)
    decoded, hashes, events = check_sources(
        run_dir=run, forensics_dir=review, original_paths=sources)
    assert decoded == sources["video_path"]
    assert hashes["original_video_sha256"] == json.loads(
        (run/"batch"/"source_provenance.json").read_text())["video_sha256"]
    assert len(events) == 24


def test_forensic_queue_rejects_fake_fluid_and_incomplete_portable_map(tmp_path):
    run, review, video, truth = original_receipt(tmp_path)
    sources = _copies(tmp_path, video, truth)
    with pytest.raises(ReviewError, match="Both original"):
        check_sources(run_dir=run, forensics_dir=review,
                      original_paths={"video_path": sources["video_path"]})
    sources["fluid_path"].write_text("fabricated independent labels")
    with pytest.raises(ReviewError, match="Original May25"):
        check_sources(run_dir=run, forensics_dir=review, original_paths=sources)
