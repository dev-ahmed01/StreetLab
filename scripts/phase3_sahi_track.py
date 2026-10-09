"""Run SAHI + Roboflow ByteTrack continuously on a FLUID video window.

A fixed independent FLUID evaluation is compulsory. A candidate cannot
be declared superior without an explicitly supplied same-window T000 track file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3_engine_shootout import score_trial
from streetlab_phase3.video.engine_shootout import promotion_gate
from streetlab_phase3.video.sahi_tracker_trial import SahiTrackingTrial, run_sahi_tracking


def dump_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fluid-tracks", required=True)
    parser.add_argument("--output", required=True, help="New unique Geo-trax-format track path")
    parser.add_argument("--baseline-tracks", help="Frozen T000 track file from SAME video interval")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--mode", choices=("standard", "sliced"), default="sliced")
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--class-map")
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-width", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--track-activation-threshold", type=float, default=0.20)
    parser.add_argument("--high-conf-det-threshold", type=float, default=0.15)
    parser.add_argument("--lost-track-buffer", type=int, default=45)
    parser.add_argument("--minimum-consecutive-frames", type=int, default=2)
    args = parser.parse_args(argv)
    trial = SahiTrackingTrial(
        video=args.video, weights=args.weights, output=args.output,
        start_frame=args.start_frame, end_frame=args.end_frame,
        mode=args.mode, warmup_frames=args.warmup_frames,
        confidence=args.confidence, image_size=args.image_size, device=args.device,
        class_map_file=args.class_map, slice_height=args.slice_height,
        slice_width=args.slice_width, overlap=args.overlap,
        track_activation_threshold=args.track_activation_threshold,
        high_conf_det_threshold=args.high_conf_det_threshold,
        lost_track_buffer=args.lost_track_buffer,
        minimum_consecutive_frames=args.minimum_consecutive_frames,
    )
    output = Path(args.output)
    manifest = run_sahi_tracking(trial)
    candidate_pixel, candidate_identity = score_trial(
        tracks=output, fluid_tracks=Path(args.fluid_tracks),
        max_distance_px=50.0, start_frame=args.start_frame,
        end_frame=args.end_frame)
    dump_new(output.with_suffix(".pixel.json"), candidate_pixel)
    dump_new(output.with_suffix(".identity.json"), candidate_identity)
    print(json.dumps({"manifest": manifest, "candidate_pixel": candidate_pixel,
                      "candidate_identity": candidate_identity}, indent=2))
    if not args.baseline_tracks:
        print("NOT PROMOTED: same-window T000 baseline track file is required.")
        return 0
    baseline_pixel, baseline_identity = score_trial(
        tracks=Path(args.baseline_tracks), fluid_tracks=Path(args.fluid_tracks),
        max_distance_px=50.0, start_frame=args.start_frame,
        end_frame=args.end_frame)
    gate = promotion_gate(baseline_pixel, candidate_pixel,
                          baseline_identity, candidate_identity)
    dump_new(output.with_suffix(".baseline_pixel.json"), baseline_pixel)
    dump_new(output.with_suffix(".baseline_identity.json"), baseline_identity)
    dump_new(output.with_suffix(".gate.json"), gate)
    print("Same-window T000 promotion gate:", json.dumps(gate, indent=2))
    if not gate["eligible_for_promotion"]:
        print("REJECTED: candidate remains isolated; production Geo-trax unchanged.")
        return 2
    print("PRELIMINARY SAME-WINDOW WIN ONLY. Untouched holdout required before promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
