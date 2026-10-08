"""Run one independently scored engine trial on a fixed FLUID video window."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.engine_shootout import Trial, promotion_gate, run_trial


def score_trial(*, tracks: Path, fluid_tracks: Path, max_distance_px: float) -> tuple[dict, dict]:
    from streetlab_phase3.geotrax_pixel import load_geotrax_pixel_tracks
    from streetlab_phase3.identity_benchmark import IdentityBenchmark
    from streetlab_phase3.pixel_benchmark import PixelBenchmark, normalize_fluid_pixel_truth
    from streetlab_phase3.serialization import package_to_dict

    with fluid_tracks.open("r", encoding="utf-8-sig", newline="") as fh:
        truth = normalize_fluid_pixel_truth(csv.DictReader(fh))
    predicted = load_geotrax_pixel_tracks(tracks)
    # Stage-A contract requires the known +1 frame alignment: do not re-optimize.
    pixel = package_to_dict(PixelBenchmark(max_distance_px).evaluate(
        predicted, truth, frame_offsets=(1,)))
    identity = IdentityBenchmark(max_distance_px).evaluate(
        predicted, truth, frame_offsets=(1,)).scorecard_dict()
    return pixel, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Local MP4 (not stored on GitHub)")
    parser.add_argument("--weights", required=True, help="YOLO .pt checkpoint, preferably Geo-trax aerial detector weights")
    parser.add_argument("--tracker", required=True, help="bytetrack.yaml, botsort.yaml or a custom YAML path")
    parser.add_argument("--output", required=True, help="Unique .txt path for 14-column Geo-trax-format tracks")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--class-map", help="Optional JSON mapping detector class NAME to canonical Geo-trax ID 0..3")
    parser.add_argument("--fluid-tracks", help="Optional FLUID ground truth CSV for evaluation")
    parser.add_argument("--baseline-pixel", help="Same-window T000 pixel scorecard JSON")
    parser.add_argument("--baseline-identity", help="Same-window T000 identity scorecard JSON")
    parser.add_argument("--max-pixel-distance", type=float, default=50.0)
    args = parser.parse_args(argv)
    if (args.baseline_pixel or args.baseline_identity) and not args.fluid_tracks:
        parser.error("Baseline gating requires --fluid-tracks")
    if args.max_pixel_distance != 50.0:
        parser.error("Stage-A spatial threshold is frozen at 50 px")
    trial = Trial(video=args.video, weights=args.weights, tracker=args.tracker,
                  output=args.output, start_frame=args.start_frame,
                  end_frame=args.end_frame, warmup_frames=args.warmup_frames,
                  confidence=args.confidence, image_size=args.image_size,
                  device=args.device, class_map_file=args.class_map)
    manifest = run_trial(trial)
    print("Extracted:", json.dumps(manifest, indent=2))

    if not args.fluid_tracks:
        print("NOT BENCHMARKED: --fluid-tracks was not provided.")
        return 0

    output = Path(args.output)
    pixel, identity = score_trial(
        tracks=output, fluid_tracks=Path(args.fluid_tracks),
        max_distance_px=args.max_pixel_distance)
    output.with_suffix(".pixel.json").write_text(
        json.dumps(pixel, indent=2), encoding="utf-8")
    output.with_suffix(".identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8")
    print("FLUID pixel scorecard:", json.dumps(pixel, indent=2))
    print("FLUID identity scorecard:", json.dumps(identity, indent=2))

    if args.baseline_pixel and args.baseline_identity:
        baseline_pixel = json.loads(Path(args.baseline_pixel).read_text(encoding="utf-8"))
        baseline_identity = json.loads(Path(args.baseline_identity).read_text(encoding="utf-8"))
        gate = promotion_gate(baseline_pixel, pixel, baseline_identity, identity)
        output.with_suffix(".gate.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8")
        print("Promotion gate:", json.dumps(gate, indent=2))
        if not gate["eligible_for_promotion"]:
            print("REJECTED: candidate remains isolated; Geo-trax baseline is unchanged.")
            return 2
        return 0
    print("NOT PROMOTED: both same-window baseline scorecards are required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
