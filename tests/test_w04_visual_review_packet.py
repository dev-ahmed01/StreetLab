"""Offline visual review packet tests with FAKE local video and synthetic events.

No original FLUID video, image or reviewer verdict is used in GitHub CI.
"""
from __future__ import annotations

from collections import Counter
import csv
import json

import numpy as np
from PIL import Image
import pytest

from streetlab_integration.w04_visual_review_packet import (
    PACKET_STATUS, ReviewError, EVENT_FIELDS, check_sources,
    make_plan, render_packet,
)
from streetlab_integration.w04_holdout_intake import digest


def event(frame, typ, truth_id=None, pred_id=None):
    return {
        "priority": 5 if typ == "HEAVY_WRONG_CLASS" else 1,
        "video_frame": frame, "fluid_frame": frame+1,
        "event": typ,
        "truth_class": "HEAVY_VEHICLE" if truth_id else None,
        "predicted_class": "CAR" if pred_id else None,
        "truth_track_id": truth_id, "predicted_track_id": pred_id,
        "truth_cx": 52. if truth_id else None,
        "truth_cy": 51. if truth_id else None,
        "predicted_cx": 55. if pred_id else None,
        "predicted_cy": 52. if pred_id else None,
        "matching_distance_px": 3.162 if truth_id and pred_id else None,
        "annotation_status": "FLUID_LABEL_ONLY_NOT_PHYSICALLY_ADJUDICATED",
    }


def sample_events():
    all_events = []
    for frame in (60, 70, 80, 100, 120, 130, 150, 179):
        all_events.append(event(frame, "HEAVY_WRONG_CLASS", "36", "29"))
        all_events.append(event(frame, "HEAVY_WRONG_CLASS", "7", "26"))
    all_events.extend([
        event(110, "HEAVY_TRUTH_UNMATCHED", "40"),
        event(130, "HEAVY_TRUTH_UNMATCHED", "34"),
        event(150, "BUS_PRED_MATCHED_NONBUS_TRUTH", "93", "136"),
        event(100, "BUS_PRED_UNMATCHED", None, "110"),
        event(140, "BUS_PRED_UNMATCHED", None, "97"),
        event(120, "MOTORCYCLE_TRUTH_UNMATCHED", "47"),
        event(130, "MOTORCYCLE_TRUTH_UNMATCHED", "60"),
        event(170, "MOTORCYCLE_TRUTH_UNMATCHED", "51"),
    ])
    return all_events


def test_repeated_heavy_tracks_get_early_middle_late_instead_of_14_duplicates():
    p = make_plan(sample_events())
    assert len(p) == 14
    counts = Counter(x["event"] for x in p)
    assert counts["HEAVY_WRONG_CLASS"] == 6
    assert counts["HEAVY_TRUTH_UNMATCHED"] == 2
    assert counts["BUS_PRED_MATCHED_NONBUS_TRUTH"] == 1
    assert counts["BUS_PRED_UNMATCHED"] == 2
    assert counts["MOTORCYCLE_TRUTH_UNMATCHED"] == 3
    for tid in ("7", "36"):
        selected = sorted(x["video_frame"] for x in p
                          if x["truth_track_id"] == tid)
        assert selected == [60, 100, 179]
        assert all(x["observed_repetitions_in_18_review_frames"] == 8
                   for x in p if x["truth_track_id"] == tid)
    assert make_plan(sample_events()) == p
    assert all(x["original_frame_not_physical_object_adjudication"] is True for x in p)


def test_packet_renders_fake_frames_and_leaves_two_blank_review_sheets(tmp_path):
    samples = make_plan(sample_events())
    indices = sorted({x["video_frame"] for x in samples})
    seen = []
    def frame_stream():
        for idx in indices:
            seen.append(idx)
            yield idx, np.zeros((120, 180, 3), dtype=np.uint8)
    output = tmp_path / "result"
    completed = render_packet(
        samples=samples, source_frames=frame_stream(),
        frame_size=(180, 120), output_dir=output,
    )
    assert seen == indices
    assert len(completed) == 14
    assert (output/"review_contact_sheet.jpg").is_file()
    with Image.open(output/"review_contact_sheet.jpg") as image:
        assert image.width == 1620
        assert image.height > 1000
    with (output/"reviewer_1_blank.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(samples)
    assert all(not r["physical_vehicle_type"] and not r["FLUID_class_accurate"] and
               not r["model_class_accurate"] for r in rows)
    assert (output/"reviewer_2_blank.csv").read_bytes() == (
        output/"reviewer_1_blank.csv").read_bytes()
    for item in completed:
        assert digest(output/item["image_file"]) == item["image_sha256"]
    with pytest.raises(FileExistsError):
        render_packet(samples=samples, source_frames=frame_stream(),
                      frame_size=(180,120), output_dir=output)


def test_absent_decoded_frames_or_mismatched_frame_dimensions_block(tmp_path):
    samples = make_plan(sample_events(), max_samples=2)
    indices = {x["video_frame"] for x in samples}
    with pytest.raises(ReviewError, match="Missing original video frames"):
        render_packet(samples=samples, source_frames=[],
                      frame_size=(180, 120), output_dir=tmp_path/"missing")
    one = next(iter(indices))
    with pytest.raises(ReviewError, match="size differs"):
        render_packet(samples=samples,
                      source_frames=[(one, np.zeros((100,180,3), dtype=np.uint8))],
                      frame_size=(180,120), output_dir=tmp_path/"bad")


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(fluid_frame=r["video_frame"]+2),
    lambda r: r.update(annotation_status="HUMAN_REVIEWED"),
    lambda r: r.update(truth_cx=float("inf")),
    lambda r: r.update(predicted_cy=None),
    lambda r: r.update(video_frame=180),
])
def test_forged_or_incomplete_candidate_events_fail_closed(mutation):
    events = sample_events()
    mutation(events[0])
    with pytest.raises(ReviewError):
        make_plan(events)


def original_receipt(tmp_path):
    run = tmp_path/"historical_run"
    review = tmp_path/"prior_forensics"
    (run/"batch").mkdir(parents=True)
    review.mkdir()
    video = tmp_path/"20250525_video.mp4"
    truth = tmp_path/"20250525_video_Traj.csv"
    video.write_bytes(b"synthetic source bytes")
    truth.write_bytes(b"synthetic trajectory rows")
    vsha, tsha = digest(video), digest(truth)
    summary = {
        "status": "W04_MAY25_CROSSDATE_REGRESSION_RECORDED_NOT_BLIND_HOLDOUT",
        "source_video_sha256": vsha,
        "original_fluid_sha256": tsha,
        "source_window": [60,179], "evaluation_frames": 120,
        "source_video_dimensions": [180,120],
        "source_frame_rate_reported": 10.,
    }
    preflight = {"source_video_sha256": vsha, "source_truth_sha256": tsha}
    batch = {"eligible_for_promotion": False}
    origin = {
        "eligible_for_promotion": False,
        "source_is_independent_holdout": False,
        "video_sha256": vsha, "fluid_sha256": tsha,
        "video_path": str(video), "fluid_path": str(truth),
    }
    for path, data in (
        (run/"cross_date_summary.json", summary),
        (run/"preflight.json", preflight),
        (run/"batch"/"batch_report.json", batch),
        (run/"batch"/"source_provenance.json", origin),
    ):
        path.write_text(json.dumps(data))
    events = sample_events()
    forensic = {
        "status": "W04_MAY25_REAL_REGRESSION_REPLAYED_WITH_REVIEW_QUEUE_NOT_BLIND",
        "original_video_and_FLUID_SHA_verified_from_saved_batch": True,
        "original_saved_scores_recomputed_from_tracks_and_FLUID": True,
        "selected_policy": "hard_nms_ios_0.30",
        "independent_holdout_pass": False,
        "production_release_approved": False,
        "saved_regression_summary_sha256": digest(run/"cross_date_summary.json"),
        "preflight_sha256": digest(run/"preflight.json"),
        "saved_regression_batch_sha256": digest(run/"batch"/"batch_report.json"),
        "review": {"events": events, "exported_candidate_events": len(events)},
    }
    summary["original_saved_batch_sha256"] = digest(run/"batch"/"batch_report.json")
    (run/"cross_date_summary.json").write_text(json.dumps(summary))
    forensic["saved_regression_summary_sha256"] = digest(run/"cross_date_summary.json")
    (review/"regression_forensics.json").write_text(json.dumps(forensic))
    with (review/"review_events.csv").open("w",newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
        w.writeheader()
        w.writerows(events)
    return run, review, video, truth


def test_original_source_hashes_and_json_csv_event_consistency(tmp_path):
    run, review, video, truth = original_receipt(tmp_path)
    path, sources, rows = check_sources(run_dir=run, forensics_dir=review)
    assert path == video
    assert sources["original_video_sha256"] == digest(video)
    assert len(rows) == len(sample_events())
    assert sources["source_fps"] == 10


def test_mutated_source_video_refused(tmp_path):
    run, review, video, truth = original_receipt(tmp_path)
    video.write_bytes(b"tampered fake source")
    with pytest.raises(ReviewError, match="Original May25"):
        check_sources(run_dir=run, forensics_dir=review)


def test_reviewer_csv_drift_refused(tmp_path):
    run, review, video, truth = original_receipt(tmp_path)
    path = review/"review_events.csv"
    text = path.read_text()
    path.write_text(text.replace("HEAVY_WRONG_CLASS", "OTHER", 1))
    with pytest.raises(ReviewError, match="does not exactly match"):
        check_sources(run_dir=run, forensics_dir=review)


def test_promoted_status_cannot_generate_visual_packet(tmp_path):
    run, review, video, truth = original_receipt(tmp_path)
    p = review/"regression_forensics.json"
    data = json.loads(p.read_text())
    data["production_release_approved"] = True
    p.write_text(json.dumps(data))
    with pytest.raises(ReviewError, match="falsely promoted"):
        check_sources(run_dir=run, forensics_dir=review)
