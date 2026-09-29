from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase1.sumo_bridge import (
    MappingConfig,
    load_profiles,
    physical_profiles,
    low_speed_gap_priors,
    persona_relative_vectors,
    seed_sumo_parameters,
    make_vtype_seed_xml,
    build_report,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--models", default="data/models")
    args = ap.parse_args()

    processed = Path(args.processed)
    models = Path(args.models)
    models.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(processed / "persona_profiles.csv")
    physical = physical_profiles(processed / "calibration_clean.parquet")
    gaps = low_speed_gap_priors(processed / "calibration_interactions.parquet", MappingConfig())
    relative = persona_relative_vectors(profiles)
    priors = seed_sumo_parameters(relative, MappingConfig())

    relative.to_csv(models / "sumo_persona_relative_vectors.csv", index=False)
    priors.to_csv(models / "sumo_persona_priors.csv", index=False)
    physical.to_csv(models / "sumo_physical_profiles.csv", index=False)
    gaps.to_csv(models / "sumo_low_speed_gap_priors.csv", index=False)

    xml = make_vtype_seed_xml(priors, physical, gaps)
    (models / "sumo_vtypes_seed.add.xml").write_text(xml, encoding="utf-8")

    report = build_report(
        profiles, relative, priors, physical, gaps, MappingConfig()
    )
    (processed / "sprint1d_mapping_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print("SPRINT 1D — SUMO PARAMETER BRIDGE")
    print(json.dumps(report, indent=2, default=str))
    print("\nSaved:")
    for p in [
        models / "sumo_persona_relative_vectors.csv",
        models / "sumo_persona_priors.csv",
        models / "sumo_physical_profiles.csv",
        models / "sumo_low_speed_gap_priors.csv",
        models / "sumo_vtypes_seed.add.xml",
        processed / "sprint1d_mapping_report.json",
    ]:
        print(p)
    print("\nIMPORTANT: These are optimizer SEEDS/BOUNDS, not calibrated SUMO parameters.")


if __name__ == "__main__":
    main()
