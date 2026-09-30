from __future__ import annotations

from pathlib import Path

import pytest

from streetlab_phase2.models import BranchSpec, DecisionEvent, DecisionType, ResponsePolicy
from streetlab_phase2.runtime import RuntimeSimulation, SimulationStatus


class FakeSimulation:
    def __init__(self) -> None:
        self.time = 0.0

    def getTime(self):
        return self.time

    def saveState(self, path: str) -> None:
        Path(path).write_text(f"<state time=\"{self.time}\"/>", encoding="utf-8")


class FakeVehicle:
    def __init__(self) -> None:
        self.ids = ["n_0001", "e_0002"]
        self.routes = {
            "n_0001": ["WJ", "JN", "NS"],
            "e_0002": ["WJ", "JE", "EE"],
        }

    def getIDList(self):
        return list(self.ids)

    def getSpeed(self, vid):
        return 5.0 if vid == "n_0001" else 8.0

    def getRoadID(self, vid):
        return "WJ"

    def getLanePosition(self, vid):
        return 180.0

    def getRoute(self, vid):
        return tuple(self.routes[vid])

    def setRoute(self, vid, route):
        self.routes[vid] = list(route)


class FakeConnection:
    def __init__(self) -> None:
        self.simulation = FakeSimulation()
        self.vehicle = FakeVehicle()
        self.closed = False

    def simulationStep(self):
        self.simulation.time += 0.2

    def close(self):
        self.closed = True


def fake_factory(net, routes, types, label):
    return FakeConnection()


def fake_branch_runner(net, routes, types, snapshot, spec, horizon):
    return {
        "name": spec.name,
        "description": spec.description,
        "decision": None if spec.decision is None else {
            "decision_type": spec.decision.decision_type,
            "at_s": spec.decision.at_s,
        },
        "metrics": {
            "arrivals_after_decision": 10 if spec.name == "BASELINE" else 8,
            "affected_upstream_northbound": 0 if spec.name == "BASELINE" else 2,
            "rerouted_vehicles": 0 if spec.name == "BASELINE" else 1,
            "mean_active_vehicles": 2.0,
            "max_approach_queue_vehicles": 1 if spec.name == "BASELINE" else 3,
            "mean_network_speed_mps": 7.0 if spec.name == "BASELINE" else 5.0,
            "mean_instant_waiting_time_s": 1.0 if spec.name == "BASELINE" else 4.0,
        },
    }


def make_runtime(tmp_path: Path) -> RuntimeSimulation:
    for name in ("net.xml", "routes.xml", "types.xml"):
        (tmp_path / name).write_text("<x/>", encoding="utf-8")
    return RuntimeSimulation(
        net=tmp_path / "net.xml",
        routes=tmp_path / "routes.xml",
        types=tmp_path / "types.xml",
        workdir=tmp_path / "runtime",
        simulation_id="test-sim",
        connection_factory=fake_factory,
        branch_runner=fake_branch_runner,
    )


def test_runtime_lifecycle_and_state_query(tmp_path):
    runtime = make_runtime(tmp_path)

    assert runtime.state()["status"] == "CREATED"
    state = runtime.start()
    assert state["status"] == "RUNNING"
    assert state["active_vehicles"] == 2

    state = runtime.step(5)
    assert state["simulation_time_s"] == pytest.approx(1.0)

    paused = runtime.pause()
    assert paused["status"] == "PAUSED"
    with pytest.raises(RuntimeError):
        runtime.step()

    resumed = runtime.resume()
    assert resumed["status"] == "RUNNING"

    conn = runtime.connection
    runtime.close()
    assert runtime.status == SimulationStatus.CLOSED
    assert conn.closed is True


def test_decision_injection_is_structured_and_applied_while_running(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.start()
    event = DecisionEvent(
        DecisionType.BLOCK_TURN,
        at_s=0.0,
        duration_s=30.0,
        from_edge="WJ",
        blocked_edge="JN",
        response_policy=ResponsePolicy.GUIDED_DETOUR,
        metadata={"source": "test"},
    )

    receipt = runtime.inject_decision(event)
    assert receipt["decision"]["decision_type"] == DecisionType.BLOCK_TURN
    assert receipt["provenance"]["response_behavior"] == "SCENARIO_ASSUMPTION"

    runtime.step()
    assert runtime.connection.vehicle.routes["n_0001"] == [
        "WJ", "JE", "EN", "N2", "NS"
    ]
    assert runtime.state()["active_decisions"][0]["decision_type"] == DecisionType.BLOCK_TURN
    runtime.close()


def test_snapshot_is_tied_to_time_and_hash(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.start()
    runtime.step(10)

    snapshot = runtime.capture_snapshot("decision point")
    assert snapshot.snapshot_id == "decision_point"
    assert snapshot.simulation_time_s == pytest.approx(2.0)
    assert Path(snapshot.path).exists()
    assert len(snapshot.sha256) == 64
    runtime.close()


def test_branches_share_snapshot_and_preserve_decision_provenance(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.start()
    runtime.step(5)
    snapshot = runtime.capture_snapshot("fork")

    decision = DecisionEvent(
        DecisionType.BLOCK_TURN,
        at_s=snapshot.simulation_time_s,
        duration_s=60.0,
        from_edge="WJ",
        blocked_edge="JN",
        response_policy=ResponsePolicy.NATURAL_REROUTE,
    )
    report = runtime.run_branches(
        snapshot_id="fork",
        horizon_s=60.0,
        branches=[
            BranchSpec("BASELINE", None, "No change"),
            BranchSpec("NATURAL", decision, "Natural response"),
        ],
    )

    assert report["same_snapshot_for_all_branches"] is True
    hashes = {
        row["provenance"]["snapshot_sha256"] for row in report["branches"]
    }
    assert hashes == {snapshot.sha256}
    natural = next(x for x in report["branches"] if x["name"] == "NATURAL")
    assert natural["provenance"]["decision"]["decision_type"] == DecisionType.BLOCK_TURN
    assert natural["vs_baseline"]["max_queue_delta"] == 2
    runtime.close()


def test_rejects_past_decision(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.start()
    runtime.step(5)

    past = DecisionEvent(
        DecisionType.BLOCK_TURN,
        at_s=0.0,
        duration_s=10.0,
        from_edge="WJ",
        blocked_edge="JN",
        response_policy=ResponsePolicy.NONE,
    )
    with pytest.raises(ValueError, match="before current"):
        runtime.inject_decision(past)
    runtime.close()
