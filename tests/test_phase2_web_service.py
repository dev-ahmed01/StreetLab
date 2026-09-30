from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeSnapshot:
    snapshot_id: str
    simulation_time_s: float
    sha256: str = "snapshot-hash"


class FakeRuntime:
    def __init__(self):
        self.status = type("Status", (), {"value": "CREATED"})()
        self.time = 0.0
        self.pause_calls = 0
        self.close_calls = 0
        self.snapshot_ids = []

    def start(self):
        self.status = type("Status", (), {"value": "RUNNING"})()
        return self.state()

    def state(self):
        return {
            "simulation_id": "fake-runtime",
            "status": self.status.value,
            "simulation_time_s": self.time,
            "active_vehicles": 20,
            "mean_speed_mps": 9.0,
            "approach_queue_vehicles": 3,
        }

    def step_until(self, target_time_s):
        self.time = target_time_s
        return self.state()

    def pause(self):
        self.pause_calls += 1
        self.status = type("Status", (), {"value": "PAUSED"})()
        return self.state()

    def capture_snapshot(self, snapshot_id):
        self.snapshot_ids.append(snapshot_id)
        return FakeSnapshot(snapshot_id, self.time)

    def close(self):
        self.close_calls += 1
        self.status = type("Status", (), {"value": "CLOSED"})()


class FakeEnsembleRunner:
    def __init__(self, runtime, capture):
        self.runtime = runtime
        self.capture = capture

    def run(self, **kwargs):
        self.capture.update(kwargs)
        self.capture["branches_copy"] = kwargs["branches"]
        return {
            "ensemble": {
                "members": kwargs["members"],
                "base_seed": kwargs["base_seed"],
                "uncertainty_source": "ASSUMPTION_ASSIGNMENT",
                "provenance": "ASSUMED",
                "same_snapshot_for_all_members": True,
                "interpretation": "Scenario sensitivity, not probability.",
            },
            "branch_summaries": {},
            "member_runs": [],
        }


def _service(runtime, capture):
    from streetlab_phase2.web_service import DecisionLabService

    return DecisionLabService(
        runtime_factory=lambda: runtime,
        ensemble_factory=lambda rt: FakeEnsembleRunner(rt, capture),
    )


def test_service_reports_created_state_before_start_and_controls_lifecycle():
    runtime = FakeRuntime()
    capture = {}
    service = _service(runtime, capture)

    created = service.state()
    assert created["status"] == "CREATED"
    assert created["simulation_time_s"] == 0.0

    started = service.start()
    assert started["status"] == "RUNNING"

    advanced = service.advance_to(60.0)
    assert advanced["simulation_time_s"] == 60.0

    closed = service.close()
    assert closed == {"status": "CLOSED"}
    assert runtime.close_calls == 1


def test_service_rejects_advance_before_start():
    import pytest

    runtime = FakeRuntime()
    service = _service(runtime, {})

    with pytest.raises(RuntimeError, match="Start"):
        service.advance_to(10.0)


def test_service_evaluates_same_snapshot_without_mutating_live_route_choice():
    from streetlab_phase2.models import ResponsePolicy

    runtime = FakeRuntime()
    capture = {}
    service = _service(runtime, capture)
    service.start()
    service.advance_to(60.0)

    result = service.evaluate_decision(
        decision_type="BLOCK_TURN",
        from_edge="WJ",
        blocked_edge="JN",
        duration_s=120.0,
        guided_share=0.50,
        members=4,
        horizon_s=120.0,
    )

    assert runtime.pause_calls == 1
    assert runtime.status.value == "PAUSED"
    assert len(runtime.snapshot_ids) == 1
    assert capture["snapshot_id"] == runtime.snapshot_ids[0]
    assert capture["members"] == 4
    assert capture["horizon_s"] == 120.0

    branches = capture["branches_copy"]
    assert [b.name for b in branches] == [
        "BASELINE",
        "NATURAL_RESPONSE",
        "GUIDED_DIVERSION",
        "MIXED_RESPONSE",
    ]
    assert branches[0].decision is None
    assert branches[1].decision.response_policy == ResponsePolicy.NATURAL_REROUTE
    assert branches[2].decision.response_policy == ResponsePolicy.GUIDED_DETOUR
    assert branches[3].decision.response_policy == ResponsePolicy.HETEROGENEOUS_RESPONSE
    assert branches[3].decision.metadata["response_assumptions"] == {
        "guided_share": 0.50,
        "seed": "replaced-by-ensemble",
        "local_trigger_position_m": 150.0,
        "provenance": "ASSUMED",
    }

    assert result["decision"]["at_s"] == 60.0
    assert result["snapshot"]["sha256"] == "snapshot-hash"
    assert result["guardrails"]["authority_remains_final_decision_maker"] is True
    assert result["guardrails"]["uncertainty_is_probability_forecast"] is False
    assert "recommendation" not in result
    assert "best_branch" not in result


def test_service_rejects_unsupported_decision_or_movement():
    import pytest

    runtime = FakeRuntime()
    service = _service(runtime, {})
    service.start()

    with pytest.raises(NotImplementedError, match="BLOCK_TURN"):
        service.evaluate_decision(
            decision_type="CHANGE_SIGNAL_PLAN",
            from_edge="WJ",
            blocked_edge="JN",
            duration_s=60.0,
            guided_share=0.5,
            members=4,
            horizon_s=60.0,
        )

    with pytest.raises(ValueError, match="WJ.*JN"):
        service.evaluate_decision(
            decision_type="BLOCK_TURN",
            from_edge="JE",
            blocked_edge="EE",
            duration_s=60.0,
            guided_share=0.5,
            members=4,
            horizon_s=60.0,
        )
