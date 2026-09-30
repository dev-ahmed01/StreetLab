from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeSnapshot:
    snapshot_id: str
    simulation_time_s: float
    sha256: str = "study-snapshot"


class FakeRuntime:
    def __init__(self):
        self.status = type("Status", (), {"value": "CREATED"})()
        self.time = 0.0
        self.pause_calls = 0
        self.snapshot_calls = 0

    def start(self):
        self.status = type("Status", (), {"value": "RUNNING"})()
        return self.state()

    def state(self):
        return {
            "simulation_id": "m6-fake",
            "status": self.status.value,
            "simulation_time_s": self.time,
            "active_vehicles": 10,
            "mean_speed_mps": 8.0,
            "approach_queue_vehicles": 2,
        }

    def step_until(self, target):
        self.time = target
        return self.state()

    def pause(self):
        self.pause_calls += 1
        self.status = type("Status", (), {"value": "PAUSED"})()
        return self.state()

    def capture_snapshot(self, snapshot_id):
        self.snapshot_calls += 1
        return FakeSnapshot(snapshot_id, self.time)

    def close(self):
        self.status = type("Status", (), {"value": "CLOSED"})()


class CaptureEnsemble:
    def __init__(self, capture):
        self.capture = capture

    def run(self, **kwargs):
        self.capture.update(kwargs)
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
        ensemble_factory=lambda _: CaptureEnsemble(capture),
    )


def _supported_evidence():
    return [
        {"name": "geometry", "provenance": "ASSUMED", "note": "demo"},
        {"name": "demand", "provenance": "ASSUMED", "note": "demo"},
        {"name": "turn_movements", "provenance": "ASSUMED", "note": "demo"},
        {"name": "alternate_route", "provenance": "ASSUMED", "note": "demo"},
        {"name": "speeds", "provenance": "CALIBRATED", "note": "phase1"},
    ]


def test_m6_needs_data_stops_before_snapshot_or_ensemble():
    runtime = FakeRuntime()
    capture = {}
    service = _service(runtime, capture)
    service.start()
    service.advance_to(60.0)

    result = service.evaluate_decision(
        decision_type="APPLY_DETOUR",
        from_edge="WJ",
        blocked_edge="JN",
        duration_s=120.0,
        guided_share=0.5,
        members=3,
        horizon_s=120.0,
        requested_metrics=[
            "max_approach_queue_vehicles",
            "mean_network_speed_mps",
            "rerouted_vehicles",
        ],
        evidence=[
            {"name": "geometry", "provenance": "ASSUMED"},
            {"name": "demand", "provenance": "ASSUMED"},
            {"name": "turn_movements", "provenance": "ASSUMED"},
            {"name": "speeds", "provenance": "CALIBRATED"},
        ],
    )

    assert result["study"]["status"] == "NEEDS_DATA"
    assert result["study"]["missing_evidence"] == ["alternate_route"]
    assert result["ensemble"] is None
    assert result["decision_evaluated"] is False
    assert runtime.pause_calls == 0
    assert runtime.snapshot_calls == 0
    assert capture == {}
    assert "confidence" not in result


def test_m6_apply_detour_runs_only_after_supported_study_gate():
    from streetlab_phase2.models import DecisionType, ResponsePolicy

    runtime = FakeRuntime()
    capture = {}
    service = _service(runtime, capture)
    service.start()
    service.advance_to(60.0)

    result = service.evaluate_decision(
        decision_type="APPLY_DETOUR",
        from_edge="WJ",
        blocked_edge="JN",
        duration_s=120.0,
        guided_share=0.5,
        members=3,
        horizon_s=120.0,
        requested_metrics=[
            "max_approach_queue_vehicles",
            "mean_network_speed_mps",
            "rerouted_vehicles",
        ],
        evidence=_supported_evidence(),
    )

    assert result["study"]["status"] == "SUPPORTED"
    assert result["decision_evaluated"] is True
    assert runtime.pause_calls == 1
    assert runtime.snapshot_calls == 1

    branches = capture["branches"]
    assert [b.name for b in branches] == [
        "BASELINE",
        "DETOUR_GUIDED",
        "DETOUR_MIXED",
    ]
    assert branches[0].decision is None
    assert branches[1].decision.decision_type == DecisionType.APPLY_DETOUR
    assert branches[1].decision.response_policy == ResponsePolicy.GUIDED_DETOUR
    assert branches[2].decision.decision_type == DecisionType.APPLY_DETOUR
    assert branches[2].decision.response_policy == ResponsePolicy.HETEROGENEOUS_RESPONSE
    assert result["decision"]["type"] == "APPLY_DETOUR"


def test_m6_block_turn_remains_supported_with_demo_evidence_defaults():
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
        guided_share=0.5,
        members=3,
        horizon_s=120.0,
    )

    assert result["study"]["status"] == "SUPPORTED"
    assert result["decision_evaluated"] is True
    assert [b.name for b in capture["branches"]] == [
        "BASELINE",
        "NATURAL_RESPONSE",
        "GUIDED_DIVERSION",
        "MIXED_RESPONSE",
    ]
