from __future__ import annotations

import argparse

from streetlab_phase2.decision_lab import run_decision_lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workdir", default=".streetlab-m1")
    ap.add_argument("--output", default="artifacts/phase2_m1_decision_lab.json")
    ap.add_argument("--decision-at-s", type=float, default=60.0)
    ap.add_argument("--horizon-s", type=float, default=180.0)
    args = ap.parse_args()

    report = run_decision_lab(
        project_root=args.project_root,
        workdir=args.workdir,
        output=args.output,
        decision_at_s=args.decision_at_s,
        horizon_s=args.horizon_s,
    )

    print("\nSTREETLAB PHASE 2 M1 — DECISION LAB")
    print(f"Decision point: {report['branching']['decision_time_s']:.1f}s")
    print(f"Shared snapshot: {report['branching']['snapshot']}")
    print()
    print(
        f"{'Branch':<24} {'Arrivals':>8} {'Rerouted':>9} "
        f"{'MaxQ':>6} {'MeanSpd':>9} {'Wait':>8}"
    )
    for b in report["branches"]:
        m = b["metrics"]
        print(
            f"{b['name']:<24} "
            f"{m['arrivals_after_decision']:>8d} "
            f"{m['rerouted_vehicles']:>9d} "
            f"{m['max_approach_queue_vehicles']:>6d} "
            f"{m['mean_network_speed_mps']:>9.3f} "
            f"{m['mean_instant_waiting_time_s']:>8.3f}"
        )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
