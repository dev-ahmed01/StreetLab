from __future__ import annotations

from pathlib import Path
from typing import Callable

from .decision_lab import _write_routes
from .demo_network import write_demo_network, write_vtypes
from .ensemble import EnsembleRunner
from .models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from .personas import load_phase1_personas, profiles_by_class
from .runtime import RuntimeSimulation


class DecisionLabService:
    """Application service connecting the web surface to M2-M4.

    The live runtime is observation state. Decision evaluation pauses it,
    captures one immutable snapshot, and runs counterfactual branches from that
    snapshot. No evaluated branch is applied back to the live runtime.
    """

    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        workdir: str | Path = ".streetlab-m5",
        runtime_factory: Callable[[], object] | None = None,
        ensemble_factory: Callable[[object], object] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.workdir = Path(workdir)
        if not self.workdir.is_absolute():
            self.workdir = self.project_root / self.workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        self._runtime_factory = runtime_factory or self._build_runtime
        self._ensemble_factory = ensemble_factory or EnsembleRunner
        self._runtime = None
        self._evaluation_index = 0

    def _build_runtime(self) -> RuntimeSimulation:
        profiles = load_phase1_personas(
            self.project_root / "data" / "models" / "sumo_persona_priors.csv"
        )
        grouped = profiles_by_class(profiles)

        net = write_demo_network(self.workdir)
        types = write_vtypes(self.workdir, profiles)
        routes = _write_routes(self.workdir, grouped)

        return RuntimeSimulation(
            net=net,
            routes=routes,
            types=types,
            workdir=self.workdir / "snapshots",
            simulation_id="m5-decision-lab",
        )

    @staticmethod
    def _status_value(runtime) -> str:
        status = getattr(runtime, "status", None)
        return str(getattr(status, "value", status or "UNKNOWN"))

    def start(self) -> dict:
        if self._runtime is None:
            self._runtime = self._runtime_factory()

        status = self._status_value(self._runtime)
        if status == "CREATED":
            return self._runtime.start()
        if status in {"RUNNING", "PAUSED"}:
            return self._runtime.state()
        raise RuntimeError(f"Cannot start simulation from {status}")

    def state(self) -> dict:
        if self._runtime is None:
            return {
                "simulation_id": "m5-decision-lab",
                "status": "CREATED",
                "simulation_time_s": 0.0,
                "active_vehicles": 0,
                "mean_speed_mps": 0.0,
                "approach_queue_vehicles": 0,
                "injected_decisions": 0,
                "active_decisions": [],
            }
        return self._runtime.state()

    def advance_to(self, target_time_s: float) -> dict:
        if self._runtime is None:
            raise RuntimeError("Start the simulation before advancing it")
        if self._status_value(self._runtime) != "RUNNING":
            raise RuntimeError("Simulation must be RUNNING before advancing")
        return self._runtime.step_until(float(target_time_s))

    def evaluate_decision(
        self,
        *,
        decision_type: str,
        from_edge: str,
        blocked_edge: str,
        duration_s: float,
        guided_share: float,
        members: int,
        horizon_s: float,
    ) -> dict:
        if self._runtime is None:
            raise RuntimeError("Start the simulation before evaluating a decision")
        if decision_type != DecisionType.BLOCK_TURN.value:
            raise NotImplementedError(
                "M5 currently supports only BLOCK_TURN decisions"
            )
        if from_edge != "WJ" or blocked_edge != "JN":
            raise ValueError(
                "The current Decision Lab slice supports movement WJ -> JN"
            )

        status = self._status_value(self._runtime)
        if status == "RUNNING":
            self._runtime.pause()
        elif status != "PAUSED":
            raise RuntimeError(
                "Simulation must be RUNNING or PAUSED before evaluation"
            )

        now = float(self._runtime.state()["simulation_time_s"])
        self._evaluation_index += 1
        snapshot_id = f"decision_point_{self._evaluation_index:03d}"
        snapshot = self._runtime.capture_snapshot(snapshot_id)

        def decision(policy: ResponsePolicy, metadata=None) -> DecisionEvent:
            return DecisionEvent(
                DecisionType.BLOCK_TURN,
                at_s=now,
                duration_s=float(duration_s),
                from_edge=from_edge,
                blocked_edge=blocked_edge,
                response_policy=policy,
                metadata=metadata or {},
            )

        branches = [
            BranchSpec("BASELINE", None, "No intervention."),
            BranchSpec(
                "NATURAL_RESPONSE",
                decision(ResponsePolicy.NATURAL_REROUTE),
                "Block turn with local/late response.",
            ),
            BranchSpec(
                "GUIDED_DIVERSION",
                decision(ResponsePolicy.GUIDED_DETOUR),
                "Block turn with upstream guided diversion.",
            ),
            BranchSpec(
                "MIXED_RESPONSE",
                decision(
                    ResponsePolicy.HETEROGENEOUS_RESPONSE,
                    metadata={
                        "response_assumptions": {
                            "guided_share": float(guided_share),
                            "seed": "replaced-by-ensemble",
                            "local_trigger_position_m": 150.0,
                            "provenance": "ASSUMED",
                        }
                    },
                ),
                (
                    "Mixed guided/local response under explicit scenario "
                    "assumptions."
                ),
            ),
        ]

        ensemble = self._ensemble_factory(self._runtime).run(
            snapshot_id=snapshot.snapshot_id,
            branches=branches,
            horizon_s=float(horizon_s),
            members=int(members),
            base_seed=f"m5-evaluation-{self._evaluation_index:03d}",
        )

        return {
            "decision": {
                "type": DecisionType.BLOCK_TURN.value,
                "from_edge": from_edge,
                "blocked_edge": blocked_edge,
                "at_s": now,
                "duration_s": float(duration_s),
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
                "live_runtime_mutated_by_evaluated_branch": False,
                "claim": (
                    "StreetLab evaluates plausible counterfactual outcomes "
                    "under explicit assumptions; the authority makes the "
                    "final decision."
                ),
            },
        }

    def close(self) -> dict[str, str]:
        if self._runtime is not None:
            self._runtime.close()
        return {"status": "CLOSED"}
