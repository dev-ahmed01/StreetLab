from __future__ import annotations


def test_m6_study_gate_reports_missing_evidence_instead_of_fabricating_confidence():
    from streetlab_phase2.models import DecisionType
    from streetlab_phase2.study import (
        EvidenceItem,
        EvidenceProvenance,
        ObservationPackage,
        StudyContract,
        StudyStatus,
        evaluate_study,
    )

    contract = StudyContract(
        decision_type=DecisionType.APPLY_DETOUR,
        area="demo_junction",
        baseline="current_state",
        scenario_family="temporary_route_management",
        requested_metrics=(
            "max_approach_queue_vehicles",
            "mean_network_speed_mps",
            "rerouted_vehicles",
        ),
    )
    observations = ObservationPackage(
        items=(
            EvidenceItem("geometry", EvidenceProvenance.ASSUMED),
            EvidenceItem("demand", EvidenceProvenance.ASSUMED),
            EvidenceItem("turn_movements", EvidenceProvenance.ASSUMED),
            EvidenceItem("speeds", EvidenceProvenance.CALIBRATED),
        )
    )

    result = evaluate_study(contract, observations)

    assert result["status"] == StudyStatus.NEEDS_DATA.value
    assert result["can_simulate"] is False
    assert result["missing_evidence"] == ["alternate_route"]
    assert "confidence" not in result
    assert result["message"].startswith("StreetLab needs additional evidence")


def test_m6_study_gate_accepts_explicit_assumptions_but_preserves_provenance():
    from streetlab_phase2.models import DecisionType
    from streetlab_phase2.study import (
        EvidenceItem,
        EvidenceProvenance,
        ObservationPackage,
        StudyContract,
        StudyStatus,
        evaluate_study,
    )

    contract = StudyContract(
        decision_type=DecisionType.APPLY_DETOUR,
        area="demo_junction",
        baseline="current_state",
        scenario_family="temporary_route_management",
        requested_metrics=(
            "max_approach_queue_vehicles",
            "mean_network_speed_mps",
            "rerouted_vehicles",
        ),
    )
    observations = ObservationPackage(
        items=(
            EvidenceItem("geometry", EvidenceProvenance.ASSUMED, "toy network"),
            EvidenceItem("demand", EvidenceProvenance.ASSUMED, "synthetic demand"),
            EvidenceItem("turn_movements", EvidenceProvenance.ASSUMED),
            EvidenceItem("alternate_route", EvidenceProvenance.ASSUMED),
            EvidenceItem("trajectories", EvidenceProvenance.CALIBRATED),
            EvidenceItem("speeds", EvidenceProvenance.CALIBRATED),
        )
    )

    result = evaluate_study(contract, observations)

    assert result["status"] == StudyStatus.SUPPORTED.value
    assert result["can_simulate"] is True
    assert result["missing_evidence"] == []
    assert result["evidence_provenance"]["alternate_route"] == "ASSUMED"
    assert result["evidence_provenance"]["speeds"] == "CALIBRATED"
    assert result["interpretation"] == (
        "Evidence sufficiency permits scenario simulation; it does not validate "
        "the assumptions as observed facts."
    )


def test_m6_study_gate_rejects_unknown_metrics_with_needs_data():
    from streetlab_phase2.models import DecisionType
    from streetlab_phase2.study import ObservationPackage, StudyContract, StudyStatus, evaluate_study

    contract = StudyContract(
        decision_type=DecisionType.BLOCK_TURN,
        area="demo_junction",
        baseline="current_state",
        scenario_family="movement_restriction",
        requested_metrics=("exact_crash_reduction",),
    )

    result = evaluate_study(contract, ObservationPackage(items=()))

    assert result["status"] == StudyStatus.NEEDS_DATA.value
    assert result["can_simulate"] is False
    assert result["unsupported_metrics"] == ["exact_crash_reduction"]
    assert "exact_crash_reduction" in result["message"]


def test_m6_demo_observation_package_is_explicit_about_assumptions():
    from streetlab_phase2.study import demo_observation_package

    package = demo_observation_package()
    provenance = package.provenance_map()

    assert provenance["geometry"] == "ASSUMED"
    assert provenance["demand"] == "ASSUMED"
    assert provenance["turn_movements"] == "ASSUMED"
    assert provenance["alternate_route"] == "ASSUMED"
    assert provenance["trajectories"] == "CALIBRATED"
    assert provenance["speeds"] == "CALIBRATED"
