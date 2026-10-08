"""Export checkpoint to a SHA-linked experimental OpenVINO CPU model.

Install 'ultralytics[export-openvino]' separately in the existing venv.
The source weights in the Hugging Face cache are never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.openvino_export import export_isolated_openvino


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=640)
    args = ap.parse_args(argv)
    result = export_isolated_openvino(
        weights=args.weights, output_dir=args.output_dir,
        image_size=args.image_size)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
