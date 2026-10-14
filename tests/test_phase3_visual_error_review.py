"""Unit tests of deterministic target selection and no-inference review rendering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from streetlab_phase3.video.visual_error_review import (
    review_targets, write_visual_review,
)


def report(standard_path="standard.txt", sliced_path="sliced.txt"):
    return {
        "status": "EXPLORATORY_SAME_FRAME_COMPARISON_NOT_PRODUCTION_EVIDENCE",
        "evaluation_start_frame": 10,
        "evaluation_end_frame": 12,
        "sliced_unmatched_proximity": {
            "status": "UNMATCHED_PROXIMITY_HEURISTIC_NOT_PROOF_OF_DUPLICATES",
            "video_frame_start": 10,
            "video_frame_end": 12,
            "examples": [
                {"video_frame": 11, "predicted_track_id": "99",
                 "predicted_class": "CAR", "confidence": .9,
                 "pixel_center": [160, 160],
                 "near_matched_prediction": False},
                {"video_frame": 10, "predicted_track_id": "88",
                 "predicted_class": "CAR", "confidence": .48,
                 "pixel_center": [160, 160],
                 "near_matched_prediction": True},
                {"video_frame": 10, "predicted_track_id": "90",
                 "predicted_class": "MOTORCYCLE", "confidence": .32,
                 "pixel_center": [60, 60],
                 "near_matched_prediction": True},
            ],
        },
        "paired_error_audit": {
            "status": "PAIRED_ERROR_AUDIT_TUNING_ONLY_NOT_PRODUCTION_EVIDENCE",
            "by_truth_class": {
                "MOTORCYCLE": {
                    "spatially_rescued_by_sliced": [
                        {"video_frame": 10, "fluid_track_id": "m1"},
                        {"video_frame": 11, "fluid_track_id": "m1"},
                    ],
                    "spatially_lost_by_sliced": [
                        {"video_frame": 11, "fluid_track_id": "m2"},
                    ],
                }
            },
        },
        "results": {
            "standard": {"tracks": standard_path},
            "sliced": {"tracks": sliced_path},
        },
    }


class FakeCap:
    def __init__(self, path):
        self.frame = 0
        self.closed = False
        FakeCV.last = self

    def isOpened(self): return True
    def set(self, prop, frame):
        self.frame = int(frame)
        return True
    def read(self):
        self.frame += 1
        return True, np.zeros((320, 320, 3), dtype=np.uint8)
    def get(self, prop): return float(self.frame)
    def release(self): self.closed = True


class FakeCV:
    CAP_PROP_POS_FRAMES = 1
    FONT_HERSHEY_SIMPLEX = 2
    IMWRITE_JPEG_QUALITY = 3
    VideoCapture = FakeCap
    last = None

    @staticmethod
    def circle(*args): pass

    @staticmethod
    def putText(*args): pass

    @staticmethod
    def resize(a, size): return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    @staticmethod
    def hconcat(images): return np.hstack(images)

    @staticmethod
    def imwrite(name, image, flags):
        Path(name).write_bytes(b"fakejpeg")
        return True


def test_selection_prioritizes_high_conf_cars_and_distinct_rescued_motorcycles():
    targets = review_targets(report())
    assert len(targets) == 4
    assert [x["reason"] for x in targets] == [
        "high_confidence_unmatched_car",
        "motorcycle_unmatched_near_another_match",
        "motorcycle_rescued_by_sliced",
        "motorcycle_lost_by_sliced",
    ]
    assert targets[2]["fluid_track_id"] == "m1"
    assert review_targets(report(), max_items=1)[0]["reason"] == "high_confidence_unmatched_car"


def test_visual_review_is_local_atomic_and_never_overwrites(tmp_path):
    video, fluid = tmp_path / "video.mp4", tmp_path / "truth.csv"
    std, sliced = tmp_path / "standard.txt", tmp_path / "sliced.txt"
    video.write_bytes(b"dummy video")
    fluid.write_text("frame,id,cx,cy,type\n11,m1,100,100,moped\n"
                     "12,m2,110,100,moped\n", encoding="utf-8")
    std.write_text("10,1,100,100,20,10,100,100,20,10,3,.9,20,10\n")
    sliced.write_text("10,2,100,100,20,10,100,100,20,10,3,.9,20,10\n")
    outdir = tmp_path / "review"
    payload = write_visual_review(
        report(str(std), str(sliced)),
        video=video, fluid_tracks=fluid,
        standard_tracks=std, sliced_tracks=sliced, output_dir=outdir,
        crop_size=128, cv2_module=FakeCV)
    assert payload["eligible_for_promotion"] is False
    assert payload["image_count"] == 4
    assert (outdir / "index.json").exists()
    assert len(list(outdir.glob("*.jpg"))) == 4
    assert FakeCV.last.closed
    assert len(json.loads((outdir / "index.json").read_text())["entries"]) == 4
    with pytest.raises(FileExistsError):
        write_visual_review(report(str(std), str(sliced)), video=video,
                            fluid_tracks=fluid, standard_tracks=std,
                            sliced_tracks=sliced, output_dir=outdir,
                            crop_size=128, cv2_module=FakeCV)


def test_wrong_track_paths_refused_before_image_writing(tmp_path):
    p = tmp_path / "notthere.txt"
    r = report()
    with pytest.raises(ValueError, match="do not match"):
        write_visual_review(r, video=p, fluid_tracks=p, standard_tracks=p,
                            sliced_tracks=p, output_dir=tmp_path / "output",
                            cv2_module=FakeCV)
    assert not (tmp_path / "output").exists()
