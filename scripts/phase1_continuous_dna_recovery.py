from __future__ import annotations

import argparse
import json

from streetlab_phase1.continuous_dna_recovery import run_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_recovery(args.project_root)

    compact = {
        "sprint": report["sprint"],
        "candidate_sources_found": report["candidate_sources_found"],
        "candidate_sources": report["candidate_sources"],
        "best_candidate": report["best_candidate"],
        "recommendation": report["recommendation"],
        "decision_rules": report["decision_rules"],
    }

    print("\nSPRINT 1E-A10.1 — CONTINUOUS DNA RECOVERY")
    print(json.dumps(compact, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a10_1_continuous_dna_recovery.json")
    print("\nDo not run the next SUMO heterogeneity model until this recovery audit is reviewed.")


if __name__ == "__main__":
    main()
