from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_continuous_dna_tau import run_continuous_dna_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_continuous_dna_tau(args.project_root)

    compact = {
        "sprint": report["sprint"],
        "hard_stop": report["hard_stop"],
        "dna_mapping": report["dna_mapping"],
        "fixed_parameters": report["fixed_parameters"],
        "results": report["results"],
        "selection": report["selection"],
        "phase1_next_step": report["phase1_next_step"],
    }

    print("\nSPRINT 1E-A11 — CONTINUOUS DNA -> TAU")
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a11_continuous_dna_tau.json")
    print("\nThis is the final calibration experiment before held-out validation.")


if __name__ == "__main__":
    main()
