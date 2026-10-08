"""Inspect missed FLUID vehicles using an EXISTING tracker output; no video inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.class_diagnostics import diagnose_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--fluid-tracks", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--output", help="Optional new JSON path; never overwritten")
    args = parser.parse_args(argv)
    report = diagnose_files(Path(args.tracks), Path(args.fluid_tracks),
                            start_frame=args.start_frame, end_frame=args.end_frame)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
