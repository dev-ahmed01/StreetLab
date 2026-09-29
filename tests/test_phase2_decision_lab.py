from pathlib import Path

import pytest

from streetlab_phase2.decision_lab import default_branches
from streetlab_phase2.models import DecisionType, ResponsePolicy
from streetlab_phase2.personas import load_phase1_personas, profiles_by_class


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_personas_are_reused_but_tau_is_frozen():
    profiles = load_phase1_personas(
        ROOT / "data" / "models" / "sumo_persona_priors.csv"
    )
    assert len(profiles) == 7
    assert {p.vehicle_class for p in profiles} == {1, 2, 6}
    assert all(p.tau_s == pytest.approx(0.50) for p in profiles)
    grouped = profiles_by_class(profiles)
    assert len(grouped[1]) == 3
    assert len(grouped[2]) == 2
    assert len(grouped[6]) == 2


def test_default_branches_fork_one_decision_point():
    branches = default_branches(60.0)
    assert [x.name for x in branches] == [
        "BASELINE",
        "BLOCK_NATURAL_120",
        "BLOCK_GUIDED_120",
        "BLOCK_GUIDED_60",
    ]
    for branch in branches[1:]:
        assert branch.decision is not None
        assert branch.decision.decision_type == DecisionType.BLOCK_TURN
        assert branch.decision.at_s == pytest.approx(60.0)
        assert branch.decision.blocked_edge == "JN"


def test_decision_window_is_half_open():
    event = default_branches(10.0)[1].decision
    assert event is not None
    assert event.active_at(10.0)
    assert event.active_at(129.999)
    assert not event.active_at(130.0)
    assert not event.active_at(9.999)


def test_guided_and_natural_policies_both_exist():
    policies = {
        x.decision.response_policy
        for x in default_branches()
        if x.decision is not None
    }
    assert ResponsePolicy.NATURAL_REROUTE in policies
    assert ResponsePolicy.GUIDED_DETOUR in policies
