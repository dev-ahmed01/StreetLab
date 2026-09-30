from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Callable

from .decision_lab import (
    _add_deltas,
    _branch_run,
    _guide_northbound,
    _natural_reroute,
    _set_turn_block,
    _start_sumo,
)
from .models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy


class SimulationStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    simulation_id: str
    simulation_time_s: float
    path: str
    sha256: str
    injected_decision_count: int


ConnectionFactory = Callable[[Path, Path, Path, str], object]
BranchRunner = Callable[[Path, Path, Path, Path, BranchSpec, float], dict]


class RuntimeSimulation:
    """Reusable M2 runtime around one live SUMO/TraCI simulation."""

    def __init__(
        self,
        *,
        net: str | Path,
        routes: str | Path,
        types: str | Path,
        workdir: str | Path,
        simulation_id: str = "streetlab_runtime",
        connection_factory: ConnectionFactory | None = None,
        branch_runner: BranchRunner | None = None,
    ) -> None:
        self.net = Path(net)
        self.routes = Path(routes)
        self.types = Path(types)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.simulation_id = simulation_id
        self.status = SimulationStatus.CREATED
        self._conn = None
        self._connection_factory = connection_factory or _start_sumo
        self._branch_runner = branch_runner or _branch_run
        self._decisions: list[DecisionEvent] = []
        self._snapshots: dict[str, SnapshotRecord] = {}

    @property
    def connection(self):
        if self._conn is None:
            raise RuntimeError("Simulation has not been started")
        return self._conn

    def start(self) -> dict:
        if self.status != SimulationStatus.CREATED:
            raise RuntimeError(f"Cannot start simulation from {self.status.value}")
        label = f"runtime_{self.simulation_id}"
        self._conn = self._connection_factory(
            self.net, self.routes, self.types, label
        )
        self.status = SimulationStatus.RUNNING
        return self.state()

    def pause(self) -> dict:
        if self.status != SimulationStatus.RUNNING:
            raise RuntimeError("Only a running simulation can be paused")
        self.status = SimulationStatus.PAUSED
        return self.state()

    def resume(self) -> dict:
        if self.status != SimulationStatus.PAUSED:
            raise RuntimeError("Only a paused simulation can be resumed")
        self.status = SimulationStatus.RUNNING
        return self.state()

    def close(self) -> None:
        if self.status == SimulationStatus.CLOSED:
            return
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self.status = SimulationStatus.CLOSED

    def step(self, steps: int = 1) -> dict:
        if self.status != SimulationStatus.RUNNING:
            raise RuntimeError("Simulation must be RUNNING before stepping")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        for _ in range(steps):
            now = float(self.connection.simulation.getTime())
            self._apply_decisions(now)
            self.connection.simulationStep()
        return self.state()

    def step_until(self, target_time_s: float) -> dict:
        if target_time_s < self._time_s():
            raise ValueError("target_time_s cannot be earlier than current time")
        while self._time_s() < target_time_s:
            self.step()
        return self.state()

    def inject_decision(self, decision: DecisionEvent) -> dict:
        if self.status not in {SimulationStatus.RUNNING, SimulationStatus.PAUSED}:
            raise RuntimeError("Start the simulation before injecting a decision")
        if decision.decision_type != DecisionType.BLOCK_TURN:
            raise NotImplementedError(
                f"M2 currently supports only {DecisionType.BLOCK_TURN.value}"
            )
        now = self._time_s()
        if decision.at_s + 1e-9 < now:
            raise ValueError(
                f"Decision time {decision.at_s:.3f}s is before current "
                f"simulation time {now:.3f}s"
            )
        self._decisions.append(decision)
        return {
            "simulation_id": self.simulation_id,
            "accepted_at_simulation_time_s": now,
            "decision": asdict(decision),
            "provenance": {
                "response_behavior": "SCENARIO_ASSUMPTION",
                "phase1_longitudinal_tau_s": 0.50,
            },
        }

    def state(self) -> dict:
        if self.status == SimulationStatus.CREATED:
            return {
                "simulation_id": self.simulation_id,
                "status": self.status.value,
                "simulation_time_s": 0.0,
                "active_vehicles": 0,
                "mean_speed_mps": 0.0,
                "approach_queue_vehicles": 0,
                "injected_decisions": 0,
                "active_decisions": [],
            }
        if self.status == SimulationStatus.CLOSED:
            return {
                "simulation_id": self.simulation_id,
                "status": self.status.value,
                "injected_decisions": len(self._decisions),
                "snapshots": len(self._snapshots),
            }

        active = list(self.connection.vehicle.getIDList())
        speeds = [float(self.connection.vehicle.getSpeed(v)) for v in active]
        queue = sum(
            1
            for vid in active
            if self.connection.vehicle.getRoadID(vid) == "WJ"
            and self.connection.vehicle.getSpeed(vid) < 0.5
        )
        now = self._time_s()
        return {
            "simulation_id": self.simulation_id,
            "status": self.status.value,
            "simulation_time_s": now,
            "active_vehicles": len(active),
            "mean_speed_mps": mean(speeds) if speeds else 0.0,
            "approach_queue_vehicles": queue,
            "injected_decisions": len(self._decisions),
            "active_decisions": [
                asdict(d) for d in self._decisions if d.active_at(now)
            ],
        }

    def capture_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        if self.status not in {SimulationStatus.RUNNING, SimulationStatus.PAUSED}:
            raise RuntimeError("Start the simulation before capturing a snapshot")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", snapshot_id).strip("._")
        if not safe_id:
            raise ValueError("snapshot_id must contain at least one safe character")
        if safe_id in self._snapshots:
            raise ValueError(f"Snapshot {safe_id!r} already exists")

        path = self.workdir / f"{safe_id}.state.xml"
        self.connection.simulation.saveState(str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = SnapshotRecord(
            snapshot_id=safe_id,
            simulation_id=self.simulation_id,
            simulation_time_s=self._time_s(),
            path=str(path),
            sha256=digest,
            injected_decision_count=len(self._decisions),
        )
        self._snapshots[safe_id] = record
        return record

    def run_branches(
        self,
        *,
        snapshot_id: str,
        branches: list[BranchSpec],
        horizon_s: float,
    ) -> dict:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Unknown snapshot_id {snapshot_id!r}")
        if not branches:
            raise ValueError("At least one branch is required")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")

        snapshot = self._snapshots[snapshot_id]
        snapshot_path = Path(snapshot.path)
        results: list[dict] = []
        for spec in branches:
            row = self._branch_runner(
                self.net,
                self.routes,
                self.types,
                snapshot_path,
                spec,
                horizon_s,
            )
            row["provenance"] = {
                "simulation_id": self.simulation_id,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_time_s": snapshot.simulation_time_s,
                "snapshot_sha256": snapshot.sha256,
                "decision": asdict(spec.decision) if spec.decision else None,
            }
            results.append(row)

        if any(x["name"] == "BASELINE" for x in results):
            _add_deltas(results)

        return {
            "simulation_id": self.simulation_id,
            "snapshot": asdict(snapshot),
            "same_snapshot_for_all_branches": all(
                x["provenance"]["snapshot_sha256"] == snapshot.sha256
                for x in results
            ),
            "branches": results,
        }

    def _time_s(self) -> float:
        if self.status == SimulationStatus.CREATED:
            return 0.0
        if self.status == SimulationStatus.CLOSED:
            raise RuntimeError("Simulation is closed")
        return float(self.connection.simulation.getTime())

    def _apply_decisions(self, now: float) -> None:
        for decision in self._decisions:
            if not decision.active_at(now):
                continue
            _set_turn_block(self.connection, True)
            if decision.response_policy == ResponsePolicy.GUIDED_DETOUR:
                _guide_northbound(self.connection)
            elif decision.response_policy == ResponsePolicy.NATURAL_REROUTE:
                _natural_reroute(self.connection)

    def __enter__(self) -> "RuntimeSimulation":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
