from __future__ import annotations

import argparse
import json

from streetlab_phase1.sumo_longitudinal_audit import run_longitudinal_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    report = run_longitudinal_audit(args.project_root)

    print("\nSPRINT 1E-A7 — LONGITUDINAL SPACING AUDIT")
    print(json.dumps(report, indent=2))
    print("\nSaved: data\\processed\\sprint1e_a7_longitudinal_spacing_audit.json")
    print("\nDo not freeze minGap from sensitivity alone; review empirical gaps and gate behavior first.")


if __name__ == "__main__":
    main()
