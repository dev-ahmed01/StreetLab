from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from streetlab_phase1.interactions import (
    InteractionConfig,
    aggregate_behavior_features,
    reconstruct_interactions,
)


def run(role: str, processed_dir: Path, cfg: InteractionConfig) -> dict:
    prefix = role.lower()
    clean_path = processed_dir / f"{prefix}_clean.parquet"
    features_path = processed_dir / f"{prefix}_vehicle_features.parquet"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}. Run Phase 1A profiling first.")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run Phase 1A profiling first.")

    clean = pd.read_parquet(clean_path)
    vehicle_features = pd.read_parquet(features_path)

    interactions, report = reconstruct_interactions(clean, cfg)
    behavior = aggregate_behavior_features(interactions, vehicle_features)

    interactions.to_parquet(processed_dir / f"{prefix}_interactions.parquet", index=False)
    behavior.to_parquet(processed_dir / f"{prefix}_behavior_features.parquet", index=False)
    (processed_dir / f"{prefix}_interaction_qa.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\n{role}")
    print(json.dumps(report, indent=2))
    print(f"Behavior-feature vehicles: {len(behavior)}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="StreetLab Sprint 1B interaction reconstruction")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--lateral-safety-margin", type=float, default=0.20)
    p.add_argument("--max-leader-distance", type=float, default=50.0)
    p.add_argument("--edge-buffer", type=float, default=10.0)
    args = p.parse_args()

    cfg = InteractionConfig(
        lateral_safety_margin_m=args.lateral_safety_margin,
        max_leader_distance_m=args.max_leader_distance,
        edge_buffer_m=args.edge_buffer,
    )

    for role in ("CALIBRATION", "VALIDATION"):
        run(role, args.processed, cfg)

    print("\nSprint 1B complete. Next: feature sanity checks and persona-vs-continuous-distribution experiment.")


if __name__ == "__main__":
    main()
