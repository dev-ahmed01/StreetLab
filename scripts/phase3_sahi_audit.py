"""Compare standard and SAHI sliced aerial detection on exactly the same FLUID frames.

Detection-only diagnostic: does not produce or claim vehicle track identities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.sahi_detection_audit import DetectorAudit, run_detector_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--fluid-tracks", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--runtime-model", help="SHA-verified isolated OpenVINO export directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--class-map")
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-width", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    args = parser.parse_args(argv)
    report = run_detector_audit(DetectorAudit(
        video=args.video, fluid_tracks=args.fluid_tracks,
        weights=args.weights, output_dir=args.output_dir,
        start_frame=args.start_frame, end_frame=args.end_frame,
        sample_step=args.sample_step, confidence=args.confidence,
        image_size=args.image_size, device=args.device,
        class_map_file=args.class_map, slice_height=args.slice_height,
        slice_width=args.slice_width, overlap=args.overlap,
        runtime_model_path=args.runtime_model))
    print(json.dumps({"output_dir": args.output_dir,
                      "runtime_backend": report["runtime_backend"],
                      "runtime_model_sha256": report["runtime_model_sha256"],
                      "standard": report["standard"],
                      "sliced": report["sliced"], "gate": report["gate"]}, indent=2))
    if not report["gate"]["eligible_for_tracking_trial"]:
        print("Sliced detector rejected for tracking trial; Geo-trax baseline untouched.")
        return 2
    print("Promising detection candidate ONLY; run tracking and held-out validation before promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
