from __future__ import annotations

from streetlab_phase2.models import (
    BranchSpec,
    DecisionEvent,
    DecisionType,
    ResponsePolicy,
)


def _mixed_branch(seed: str = "m3-base") -> BranchSpec:
    return BranchSpec(
        "MIXED_RESPONSE",
        DecisionEvent(
            DecisionType.BLOCK_TURN,
            at_s=60.0,
            duration_s=120.0,
            from_edge="WJ",
            blocked_edge="JN",
            response_policy=ResponsePolicy.HETEROGENEOUS_RESPONSE,
            metadata={
                "response_assumptions": {
                    "guided_share": 0.50,
                    "seed": seed,
                    "local_trigger_position_m": 150.0,
                    "provenance": "ASSUMED",
                }
            },
        ),
        "Mixed response",
    )


def test_summary_statistics_report_distribution_not_single_prediction():
    from streetlab_phase2.ensemble import summarize_values

    summary = summarize_values([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary == {
        "min": 1.0,
        "mean": 3.0,
        "median": 3.0,
        "p10": 1.4,
        "p90": 4.6,
        "max": 5.0,
        "spread": 4.0,
    }


def test_ensemble_runner_uses_distinct_assumption_seeds_without_mutating_branch():
    from streetlab_phase2.ensemble import EnsembleRunner

    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def run_branches(self, *, snapshot_id, branches, horizon_s):
            self.calls.append((snapshot_id, branches, horizon_s))
            mixed = next(b for b in branches if b.name == "MIXED_RESPONSE")
            seed = mixed.decision.metadata["response_assumptions"]["seed"]
            member = int(seed.rsplit("-", 1)[-1])
            return {
                "snapshot": {
                    "snapshot_id": snapshot_id,
                    "sha256": "abc123",
                    "simulation_time_s": 60.0,
                },
                "same_snapshot_for_all_branches": True,
                "branches": [
                    {
                        "name": "BASELINE",
                        "metrics": {
                            "arrivals_after_decision": 100,
                            "max_approach_queue_vehicles": 10,
                            "mean_network_speed_mps": 8.0,
                            "mean_instant_waiting_time_s": 2.0,
                            "rerouted_vehicles": 0,
                        },
                        "vs_baseline": {
                            "arrivals_delta": 0,
                            "max_queue_delta": 0,
                            "mean_speed_delta_mps": 0.0,
                            "mean_waiting_delta_s": 0.0,
                        },
                    },
                    {
                        "name": "MIXED_RESPONSE",
                        "metrics": {
                            "arrivals_after_decision": 95 + member,
                            "max_approach_queue_vehicles": 6 + member,
                            "mean_network_speed_mps": 9.0 - member * 0.1,
                            "mean_instant_waiting_time_s": 1.0 + member * 0.1,
                            "rerouted_vehicles": 70 + member,
                        },
                        "vs_baseline": {
                            "arrivals_delta": -5 + member,
                            "max_queue_delta": -4 + member,
                            "mean_speed_delta_mps": 1.0 - member * 0.1,
                            "mean_waiting_delta_s": -1.0 + member * 0.1,
                        },
                    },
                ],
            }

    runtime = FakeRuntime()
    original = _mixed_branch("original-seed")
    runner = EnsembleRunner(runtime)

    report = runner.run(
        snapshot_id="decision_point",
        branches=[BranchSpec("BASELINE", None, "No intervention"), original],
        horizon_s=120.0,
        members=3,
        base_seed="ensemble",
    )

    observed_seeds = [
        next(b for b in branches if b.name == "MIXED_RESPONSE")
        .decision.metadata["response_assumptions"]["seed"]
        for _, branches, _ in runtime.calls
    ]
    assert observed_seeds == [
        "ensemble-000",
        "ensemble-001",
        "ensemble-002",
    ]
    assert (
        original.decision.metadata["response_assumptions"]["seed"]
        == "original-seed"
    )
    assert report["ensemble"]["members"] == 3
    assert report["ensemble"]["uncertainty_source"] == "ASSUMPTION_ASSIGNMENT"
    assert report["ensemble"]["same_snapshot_for_all_members"] is True
    mixed = report["branch_summaries"]["MIXED_RESPONSE"]
    assert mixed["metrics"]["max_approach_queue_vehicles"]["min"] == 6.0
    assert mixed["metrics"]["max_approach_queue_vehicles"]["max"] == 8.0
    assert mixed["metrics"]["max_approach_queue_vehicles"]["spread"] == 2.0


def test_ensemble_runner_preserves_same_decision_except_member_seed():
    from streetlab_phase2.ensemble import EnsembleRunner

    class FakeRuntime:
        def __init__(self):
            self.branches_seen = []

        def run_branches(self, *, snapshot_id, branches, horizon_s):
            self.branches_seen.append(branches)
            return {
                "snapshot": {
                    "snapshot_id": snapshot_id,
                    "sha256": "same-snapshot",
                    "simulation_time_s": 60.0,
                },
                "same_snapshot_for_all_branches": True,
                "branches": [
                    {
                        "name": b.name,
                        "metrics": {
                            "arrivals_after_decision": 1,
                            "max_approach_queue_vehicles": 1,
                            "mean_network_speed_mps": 1.0,
                            "mean_instant_waiting_time_s": 1.0,
                            "rerouted_vehicles": 1,
                        },
                        "vs_baseline": {
                            "arrivals_delta": 0,
                            "max_queue_delta": 0,
                            "mean_speed_delta_mps": 0.0,
                            "mean_waiting_delta_s": 0.0,
                        },
                    }
                    for b in branches
                ],
            }

    original = _mixed_branch()
    runtime = FakeRuntime()
    EnsembleRunner(runtime).run(
        snapshot_id="decision_point",
        branches=[BranchSpec("BASELINE", None, "No intervention"), original],
        horizon_s=90.0,
        members=2,
        base_seed="ens",
    )

    for member_branches in runtime.branches_seen:
        mixed = next(b for b in member_branches if b.name == "MIXED_RESPONSE")
        assert mixed.decision.at_s == original.decision.at_s
        assert mixed.decision.duration_s == original.decision.duration_s
        assert mixed.decision.from_edge == original.decision.from_edge
        assert mixed.decision.blocked_edge == original.decision.blocked_edge
        assert mixed.decision.response_policy == original.decision.response_policy
        assert (
            mixed.decision.metadata["response_assumptions"]["guided_share"]
            == 0.50
        )
        assert (
            mixed.decision.metadata["response_assumptions"]["local_trigger_position_m"]
            == 150.0
        )


def test_ensemble_rejects_ambiguous_or_invalid_runs():
    import pytest

    from streetlab_phase2.ensemble import EnsembleRunner

    class NeverCalledRuntime:
        def run_branches(self, **kwargs):
            raise AssertionError("runtime should not be called")

    runner = EnsembleRunner(NeverCalledRuntime())

    with pytest.raises(ValueError, match="members"):
        runner.run(
            snapshot_id="s",
            branches=[BranchSpec("BASELINE", None, "No intervention")],
            horizon_s=10.0,
            members=1,
        )

    with pytest.raises(ValueError, match="unique"):
        runner.run(
            snapshot_id="s",
            branches=[
                BranchSpec("DUP", None, "a"),
                BranchSpec("DUP", None, "b"),
            ],
            horizon_s=10.0,
            members=2,
        )
