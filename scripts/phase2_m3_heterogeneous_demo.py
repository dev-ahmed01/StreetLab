from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetlab_phase2.decision_lab import _write_routes
from streetlab_phase2.demo_network import write_demo_network, write_vtypes
from streetlab_phase2.models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from streetlab_phase2.personas import load_phase1_personas, profiles_by_class
from streetlab_phase2.runtime import RuntimeSimulation


def _decision(policy: ResponsePolicy, at_s: float, metadata=None) -> DecisionEvent:
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
    ap.add_argument("--workdir", default=".streetlab-m3")
    ap.add_argument("--output", default="artifacts/phase2_m3_heterogeneous_response.json")
    ap.add_argument("--decision-at-s", type=float, default=60.0)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    ap.add_argument("--guided-share", type=float, default=0.50)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = root / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    profiles = load_phase1_personas(root / "data" / "models" / "sumo_persona_priors.csv")
    grouped = profiles_by_class(profiles)
    net = write_demo_network(workdir)
    types = write_vtypes(workdir, profiles)
    routes = _write_routes(workdir, grouped)

    runtime = RuntimeSimulation(
        net=net,
        routes=routes,
        types=types,
        workdir=workdir / "snapshots",
        simulation_id="m3-demo",
    )

    try:
        runtime.start()
        runtime.step_until(args.decision_at_s)
        runtime.pause()
        snapshot = runtime.capture_snapshot("decision_point")

        mixed_metadata = {
            "response_assumptions": {
                "guided_share": args.guided_share,
                "seed": "streetlab-m3-demo",
                "local_trigger_position_m": 150.0,
                "provenance": "ASSUMED",
            }
        }

        comparison = runtime.run_branches(
            snapshot_id=snapshot.snapshot_id,
            horizon_s=args.horizon_s,
            branches=[
                BranchSpec("BASELINE", None, "No intervention."),
                BranchSpec(
                    "NATURAL_RESPONSE",
                    _decision(ResponsePolicy.NATURAL_REROUTE, snapshot.simulation_time_s),
                    "All affected drivers react locally near the closure.",
                ),
                BranchSpec(
                    "GUIDED_DIVERSION",
                    _decision(ResponsePolicy.GUIDED_DETOUR, snapshot.simulation_time_s),
                    "All affected drivers accept upstream diversion.",
                ),
                BranchSpec(
                    "MIXED_RESPONSE",
                    _decision(
                        ResponsePolicy.HETEROGENEOUS_RESPONSE,
                        snapshot.simulation_time_s,
                        mixed_metadata,
                    ),
                    "Mixed guided/local response under explicit scenario assumptions.",
                ),
            ],
        )

        report = {
            "milestone": "PHASE2_M3_HETEROGENEOUS_RESPONSE",
            "objective": (
                "Compare plausible traffic outcomes when affected drivers respond "
                "differently to the same intervention, without changing the "
                "authority's role as final decision-maker."
            ),
            "snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "simulation_time_s": snapshot.simulation_time_s,
                "sha256": snapshot.sha256,
            },
            "comparison": comparison,
            "guardrails": {
                "phase1_retuned": False,
                "routing_behavior_empirically_learned": False,
                "response_assumptions_provenance": "ASSUMED",
                "claim": (
                    "StreetLab evaluates plausible counterfactual outcomes under "
                    "explicit assumptions; it does not predict one exact future."
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

    print("\nSTREETLAB PHASE 2 M3 — HETEROGENEOUS RESPONSE")
    print(f"Snapshot: {snapshot.simulation_time_s:.1f}s {snapshot.sha256}")
    for branch in comparison["branches"]:
        m = branch["metrics"]
        print(
            f"{branch['name']:<20} arrivals={m['arrivals_after_decision']:>3} "
            f"rerouted={m['rerouted_vehicles']:>3} maxQ={m['max_approach_queue_vehicles']:>3} "
            f"guided={m['heterogeneous_guided_responders']:>3} "
            f"local={m['heterogeneous_local_responders']:>3}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
