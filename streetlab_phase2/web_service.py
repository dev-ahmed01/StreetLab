from __future__ import annotations

from pathlib import Path
from typing import Callable

from .decision_lab import _write_routes
from .demo_network import write_demo_network, write_vtypes
from .ensemble import EnsembleRunner
from .models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from .personas import load_phase1_personas, profiles_by_class
from .runtime import RuntimeSimulation
from .study import (
    EvidenceItem,
    EvidenceProvenance,
    ObservationPackage,
    StudyContract,
    demo_observation_package,
    evaluate_study,
)


DEFAULT_METRICS = (
    "max_approach_queue_vehicles",
    "mean_network_speed_mps",
    "rerouted_vehicles",
)


class DecisionLabService:
    """Application service connecting the web surface to M2-M6.

    Decision evaluation is evidence-gated. Unsupported studies return
    NEEDS_DATA before the live simulation is paused, snapshotted or forked.
    Supported studies pause one live state, capture one immutable snapshot and
    run counterfactual branches from that snapshot. No evaluated branch is
    applied back to the live runtime.
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
            simulation_id="m6-decision-lab",
        )

    @staticmethod
    def _status_value(runtime) -> str:
        status = getattr(runtime, "status", None)
        return str(getattr(status, "value", status or "UNKNOWN"))

    @staticmethod
    def _observation_package(
        evidence: list[dict] | tuple[dict, ...] | None,
    ) -> ObservationPackage:
        if evidence is None:
            return demo_observation_package()

        items = []
        for row in evidence:
            try:
                provenance = EvidenceProvenance(str(row["provenance"]))
                name = str(row["name"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Each evidence item requires a valid name and provenance"
                ) from exc
            items.append(
                EvidenceItem(
                    name=name,
                    provenance=provenance,
                    note=str(row.get("note", "")),
                )
            )
        return ObservationPackage(items=tuple(items))

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
                "simulation_id": "m6-decision-lab",
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

    def check_study(
        self,
        *,
        decision_type: str,
        requested_metrics: list[str] | tuple[str, ...],
        evidence: list[dict] | tuple[dict, ...] | None = None,
        area: str = "demo_junction",
        baseline: str = "current_state",
        scenario_family: str = "decision_lab",
        decision_question: str = "",
        unsupported_claims: list[str] | tuple[str, ...] = (),
    ) -> dict:
        try:
            kind = DecisionType(decision_type)
        except ValueError as exc:
            return {
                "status": "NEEDS_DATA",
                "can_simulate": False,
                "decision_type": decision_type,
                "area": area,
                "baseline": baseline,
                "scenario_family": scenario_family,
                "requested_metrics": list(requested_metrics),
                "missing_evidence": [],
                "unsupported_metrics": [],
                "unsupported_claims": list(unsupported_claims),
                "evidence_provenance": {},
                "message": f"Unsupported decision type: {decision_type}",
                "interpretation": (
                    "StreetLab refuses unsupported simulation instead of "
                    "fabricating confidence or filling missing evidence silently."
                ),
            }

        contract = StudyContract(
            decision_type=kind,
            area=area,
            baseline=baseline,
            scenario_family=scenario_family,
            requested_metrics=tuple(requested_metrics),
            decision_question=decision_question,
            unsupported_claims=tuple(unsupported_claims),
        )
        return evaluate_study(
            contract,
            self._observation_package(evidence),
        )

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
        requested_metrics: list[str] | tuple[str, ...] | None = None,
        evidence: list[dict] | tuple[dict, ...] | None = None,
    ) -> dict:
        if self._runtime is None:
            raise RuntimeError("Start the simulation before evaluating a decision")

        try:
            kind = DecisionType(decision_type)
        except ValueError as exc:
            raise NotImplementedError(
                "M6 currently supports BLOCK_TURN and APPLY_DETOUR"
            ) from exc

        if kind not in {DecisionType.BLOCK_TURN, DecisionType.APPLY_DETOUR}:
            raise NotImplementedError(
                "M6 currently supports BLOCK_TURN and APPLY_DETOUR"
            )
        if from_edge != "WJ" or blocked_edge != "JN":
            raise ValueError(
                "The current Decision Lab slice supports movement WJ -> JN"
            )

        metrics = tuple(requested_metrics or DEFAULT_METRICS)
        study = self.check_study(
            decision_type=kind.value,
            requested_metrics=metrics,
            evidence=evidence,
            area="demo_junction",
            baseline="current_state",
            scenario_family=(
                "movement_restriction"
                if kind == DecisionType.BLOCK_TURN
                else "temporary_route_management"
            ),
        )

        if not study["can_simulate"]:
            return {
                "study": study,
                "decision_evaluated": False,
                "decision": {
                    "type": kind.value,
                    "from_edge": from_edge,
                    "blocked_edge": blocked_edge,
                    "duration_s": float(duration_s),
                },
                "snapshot": None,
                "ensemble": None,
                "guardrails": {
                    "phase1_retuned": False,
                    "confidence_fabricated": False,
                    "authority_remains_final_decision_maker": True,
                    "claim": (
                        "StreetLab refuses unsupported simulation instead of "
                        "fabricating confidence."
                    ),
                },
            }

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
                kind,
                at_s=now,
                duration_s=float(duration_s),
                from_edge=from_edge,
                blocked_edge=blocked_edge,
                response_policy=policy,
                metadata=metadata or {},
            )

        mixed_metadata = {
            "response_assumptions": {
                "guided_share": float(guided_share),
                "seed": "replaced-by-ensemble",
                "local_trigger_position_m": 150.0,
                "provenance": "ASSUMED",
            }
        }

        if kind == DecisionType.BLOCK_TURN:
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
                        metadata=mixed_metadata,
                    ),
                    (
                        "Mixed guided/local response under explicit scenario "
                        "assumptions."
                    ),
                ),
            ]
        else:
            branches = [
                BranchSpec("BASELINE", None, "No intervention."),
                BranchSpec(
                    "DETOUR_GUIDED",
                    decision(ResponsePolicy.GUIDED_DETOUR),
                    "Apply temporary guided diversion without claiming a closure.",
                ),
                BranchSpec(
                    "DETOUR_MIXED",
                    decision(
                        ResponsePolicy.HETEROGENEOUS_RESPONSE,
                        metadata=mixed_metadata,
                    ),
                    (
                        "Apply temporary diversion with mixed uptake under "
                        "explicit assumptions."
                    ),
                ),
            ]

        ensemble = self._ensemble_factory(self._runtime).run(
            snapshot_id=snapshot.snapshot_id,
            branches=branches,
            horizon_s=float(horizon_s),
            members=int(members),
            base_seed=f"m6-evaluation-{self._evaluation_index:03d}",
        )

        return {
            "study": study,
            "decision_evaluated": True,
            "decision": {
                "type": kind.value,
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
                "confidence_fabricated": False,
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
