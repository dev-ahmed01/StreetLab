"""Import an existing native 14-column primary ByteTrack export into Decision Lab.

This is the FIRST product integration milestone, not a new detector benchmark.
Produces local image-pixel observation counts, trajectory CSV, immutable SHA
receipt, and the genuine Phase 2 M6 NEEDS_DATA real-site study result.
It NEVER turns uncalibrated pixel motion into m/s, route demand or SUMO geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_integration.observation_bridge import ingest_track_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True,
                   help="Native 14-column genuine tracker export, source-pixel coordinates")
    p.add_argument("--workdir", type=Path, default=Path(".streetlab-m5"),
                   help="The same workdir used by the Decision Lab API")
    p.add_argument("--fps", type=float, help="Source FPS for duration only, not physical speed")
    p.add_argument("--label", default="native_bytetrack",
                   help="Provenance label; not a claim of ground-truth quality")
    args = p.parse_args(argv)
    location, report = ingest_track_file(args.tracks, args.workdir,
                                         fps=args.fps, source_label=args.label)
    print(json.dumps({
        "milestone": "INTEGRATION_M1_VIDEO_TRACK_TO_DECISION_LAB",
        "receipt": str(location),
        "tracks_sha256": report["source_tracking_sha256"],
        "observed_track_ids": report["tracking_identity_count"],
        "source_points": report["observed_points"],
        "classes": report["tracks_by_class"],
        "real_site_study_status": report["study_gate"]["status"],
        "missing_real_site_inputs": report["study_gate"]["missing_evidence"],
        "web_endpoint": "GET /api/observations/latest",
        "network_source": "DEMO_SYNTHETIC_NOT_OBSERVED",
    }, indent=2))


if __name__ == "__main__":
    main()
