from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_demand_audit import run_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_audit(args.project_root)
    print("SPRINT 1E-A4 — DEMAND / GEOMETRY AUDIT")
    print(json.dumps(report, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a4_demand_geometry_audit.json")
    print("\nDo not run another SUMO calibration yet. Review this audit first.")


if __name__ == "__main__":
    main()
