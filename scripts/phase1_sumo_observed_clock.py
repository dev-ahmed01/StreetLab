from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_observed_clock import run_observed_clock_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_observed_clock_experiment(args.project_root)
    print("\nSPRINT 1E-A5 — OBSERVED 55 m CROSSING CLOCK")
    print(json.dumps(report, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a5_observed_clock_report.json")
    print("\nDo not tune further until this report is reviewed.")


if __name__ == "__main__":
    main()
