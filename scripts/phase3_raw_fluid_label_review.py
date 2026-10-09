"""Inspect ORIGINAL unnormalized FLUID class labels near W04 visual cases.

No inference, video decoding, benchmark mutation, or label correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.raw_fluid_label_review import raw_label_audit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--precision-review", type=Path, required=True)
    p.add_argument("--fluid-tracks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = raw_label_audit(
        gallery_index=args.index, precision_review=args.precision_review,
        fluid_tracks=args.fluid_tracks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({
        "status": report["status"],
        "cases": [
            {**{k: e[k] for k in ("image_file","source_frame","selected_class",
                                  "review_reason","nearest_within_50px")},
             "closest_fluid_label": e["nearest_raw_fluid_labels"][0]
                 if e["nearest_raw_fluid_labels"] else None}
            for e in report["cases"]
        ],
        "output": str(args.output),
        "caution": report["caution"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
