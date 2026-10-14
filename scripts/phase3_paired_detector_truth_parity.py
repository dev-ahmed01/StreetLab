"""Show whether OpenVINO and PyTorch recovered the SAME annotated motorcycles.

No video decoding, inference or ByteTrack runs. Compares exact FLUID truth
observation IDs on the same five cached frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.paired_detector_truth_parity import (
    compare_detector_truth_identity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_detector_truth_identity(
        args.reference_dir, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({
        "status": result["status"],
        "sample_frames": result["samples"],
        "motorcycle": {
            k: v for k, v in result["by_class"]["MOTORCYCLE"].items()
            if k != "annotated_vehicle_observation_disagreements"
        },
        "by_class": {
            name: {
                k: v for k, v in row.items()
                if k != "annotated_vehicle_observation_disagreements"
            } for name, row in result["by_class"].items()
        },
        "observation_disagreements": result["total_annotation_observation_disagreements"],
        "output": str(args.output),
        "caution": result["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
