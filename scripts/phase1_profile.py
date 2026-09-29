from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase1.cleaning import clean_trajectories
from streetlab_phase1.features import class_profile, extract_vehicle_features
from streetlab_phase1.loaders import load_trajectory_xlsx
from streetlab_phase1.qa import build_warnings


def run(path: Path, role: str, output_dir: Path, expected_dt: float, min_track_s: float, drop_flagged: bool) -> dict:
    df, mapping = load_trajectory_xlsx(path)
    clean, report = clean_trajectories(df, drop_flagged_rows=drop_flagged)
    report["warnings"] = build_warnings(clean, report["median_sample_seconds"], expected_dt)
    report["path"] = str(path)
    report["role"] = role
    report["column_mapping"] = mapping
    report["vehicle_class_counts"] = {
        str(k): int(v) for k, v in clean.groupby("vehicle_class")["vehicle_id"].nunique().to_dict().items()
    }

    features = extract_vehicle_features(clean, minimum_track_seconds=min_track_s)
    profile = class_profile(features)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_dir / f"{role.lower()}_clean.parquet", index=False)
    features.to_parquet(output_dir / f"{role.lower()}_vehicle_features.parquet", index=False)
    profile.to_csv(output_dir / f"{role.lower()}_class_profile.csv", index=False)
    (output_dir / f"{role.lower()}_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="StreetLab Phase 1A trajectory profiler")
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--output", default=Path("data/processed"), type=Path)
    p.add_argument("--expected-dt", default=0.5, type=float)
    p.add_argument("--min-track-seconds", default=3.0, type=float)
    p.add_argument("--keep-flagged", action="store_true")
    args = p.parse_args()

    for path, role in [(args.calibration, "CALIBRATION"), (args.validation, "VALIDATION")]:
        report = run(path, role, args.output, args.expected_dt, args.min_track_seconds, not args.keep_flagged)
        print(f"\n{role}: {path}")
        print(json.dumps(report, indent=2))

    print("\nPhase 1A profiling complete. Next: feature validity checks, then persona-vs-continuous-distribution experiment.")


if __name__ == "__main__":
    main()
