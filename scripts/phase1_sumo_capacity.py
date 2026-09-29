from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_capacity import run_capacity_diagnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_capacity_diagnostic(args.project_root)
    print("\nSPRINT 1E-A0 CAPACITY DIAGNOSTIC")
    print(json.dumps(report, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a0_capacity_report.json")
    print("\nDo NOT run speed calibration yet. Review which capacity case passes first.")


if __name__ == "__main__":
    main()
