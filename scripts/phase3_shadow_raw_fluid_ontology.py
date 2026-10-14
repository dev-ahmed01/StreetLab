"""Compute conservative class-agnostic spatial-only links to untouched RAW FLUID labels.

Zero detector inference. Preserves every official class-aware match.
Every new cross-class pairing is a diagnostic hypothesis, not a correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetlab_phase3.video.shadow_raw_fluid_ontology import shadow_raw_fluid_ontology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = shadow_raw_fluid_ontology(args.audit_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({
        "status": report["status"],
        "original_frozen_class_aware": report["original_frozen_class_aware"],
        "raw_fluid_type_coverage":
            report["all_original_fluid_annotations_by_mapped_type"],
        "additional_cross_class_center_pairings":
            report["additional_shadow_cross_class_spatial_pairs"],
        "pairings_to_fluid_types_omitted_by_frozen_four_class_scorer":
            report["pairs_to_labels_outside_original_four_classes"],
        "shadow_class_pair_counts": report["shadow_class_pair_counts"],
        "remaining_without_unused_fluid_center_within_50px_by_class":
            report["remaining_without_unused_FLUID_center_within_50px_by_class"],
        "output": str(args.output),
        "caution": report["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
