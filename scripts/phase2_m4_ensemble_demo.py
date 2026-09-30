from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase2.decision_lab import _write_routes
from streetlab_phase2.demo_network import write_demo_network, write_vtypes
from streetlab_phase2.ensemble import EnsembleRunner
from streetlab_phase2.models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from streetlab_phase2.personas import load_phase1_personas, profiles_by_class
from streetlab_phase2.runtime import RuntimeSimulation


def _decision(
    policy: ResponsePolicy,
    at_s: float,
    *,
    metadata: dict | None = None,
) -> DecisionEvent:
    return DecisionEvent(
        DecisionType.BLOCK_TURN,
        at_s=at_s,
        duration_s=120.0,
        from_edge="WJ",
        blocked_edge="JN",
        response_policy=policy,
        metadata=metadata or {},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workdir", default=".streetlab-m4")
    ap.add_argument("--output", default="artifacts/phase2_m4_ensemble.json")
    ap.add_argument("--decision-at-s", type=float, default=60.0)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--guided-share", type=float, default=0.50)
    ap.add_argument("--ensemble-seed", default="streetlab-m4-demo")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = root / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    profiles = load_phase1_personas(
        root / "data" / "models" / "sumo_persona_priors.csv"
    )
    grouped = profiles_by_class(profiles)
    net = write_demo_network(workdir)
    types = write_vtypes(workdir, profiles)
    routes = _write_routes(workdir, grouped)

    runtime = RuntimeSimulation(
        net=net,
        routes=routes,
        types=types,
        workdir=workdir / "snapshots",
        simulation_id="m4-demo",
    )

    try:
        runtime.start()
        runtime.step_until(args.decision_at_s)
        runtime.pause()
        snapshot = runtime.capture_snapshot("decision_point")

        branches = [
            BranchSpec("BASELINE", None, "No intervention."),
            BranchSpec(
                "NATURAL_RESPONSE",
                _decision(
                    ResponsePolicy.NATURAL_REROUTE,
                    snapshot.simulation_time_s,
                ),
                "All affected drivers react locally near the closure.",
            ),
            BranchSpec(
                "GUIDED_DIVERSION",
                _decision(
                    ResponsePolicy.GUIDED_DETOUR,
                    snapshot.simulation_time_s,
                ),
                "All affected drivers accept upstream diversion.",
            ),
            BranchSpec(
                "MIXED_RESPONSE",
                _decision(
                    ResponsePolicy.HETEROGENEOUS_RESPONSE,
                    snapshot.simulation_time_s,
                    metadata={
                        "response_assumptions": {
                            "guided_share": args.guided_share,
                            "seed": "replaced-by-ensemble",
                            "local_trigger_position_m": 150.0,
                            "provenance": "ASSUMED",
                        }
                    },
                ),
                (
                    "Mixed guided/local response with member-specific assignment "
                    "under a fixed macro assumption."
                ),
            ),
        ]

        ensemble = EnsembleRunner(runtime).run(
            snapshot_id=snapshot.snapshot_id,
            branches=branches,
            horizon_s=args.horizon_s,
            members=args.members,
            base_seed=args.ensemble_seed,
        )

        report = {
            "milestone": "PHASE2_M4_SCENARIO_ENSEMBLES",
            "objective": (
                "Show the range of plausible counterfactual outcomes under "
                "explicit assumptions so an authority can judge robustness "
                "without treating one run as an exact prediction."
            ),
            "decision": {
                "type": DecisionType.BLOCK_TURN.value,
                "from_edge": "WJ",
                "blocked_edge": "JN",
                "at_s": snapshot.simulation_time_s,
                "duration_s": 120.0,
            },
            "snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "simulation_time_s": snapshot.simulation_time_s,
                "sha256": snapshot.sha256,
            },
            "ensemble": ensemble,
            "guardrails": {
                "phase1_retuned": False,
                "microscopic_parameters_varied": False,
                "routing_behavior_empirically_learned": False,
                "uncertainty_is_probability_forecast": False,
                "authority_remains_final_decision_maker": True,
                "claim": (
                    "StreetLab evaluates a range of plausible counterfactual "
                    "outcomes under explicit assumptions; it does not predict "
                    "one exact future or recommend a policy."
                ),
            },
        }
    finally:
        runtime.close()

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSTREETLAB PHASE 2 M4 — SCENARIO ENSEMBLES")
    print(
        f"Snapshot: {snapshot.simulation_time_s:.1f}s "
        f"{snapshot.sha256}"
    )
    print(f"Members: {args.members}")
    for name, summary in ensemble["branch_summaries"].items():
        queue = summary["metrics"]["max_approach_queue_vehicles"]
        speed = summary["metrics"]["mean_network_speed_mps"]
        print(
            f"{name:<20} "
            f"queue median={queue['median']:.2f} "
            f"p10-p90={queue['p10']:.2f}-{queue['p90']:.2f} "
            f"speed median={speed['median']:.3f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
