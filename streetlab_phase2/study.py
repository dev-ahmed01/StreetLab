from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .models import DecisionType


class EvidenceProvenance(str, Enum):
    OBSERVED_MANUAL = "OBSERVED_MANUAL"
    OBSERVED_AUTO = "OBSERVED_AUTO"
    SENSOR = "SENSOR"
    CALIBRATED = "CALIBRATED"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"


class StudyStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    NEEDS_DATA = "NEEDS_DATA"


@dataclass(frozen=True)
class EvidenceItem:
    name: str
    provenance: EvidenceProvenance
    note: str = ""


@dataclass(frozen=True)
class ObservationPackage:
    items: tuple[EvidenceItem, ...]

    def by_name(self) -> dict[str, EvidenceItem]:
        return {item.name: item for item in self.items}

    def provenance_map(self) -> dict[str, str]:
        return {item.name: item.provenance.value for item in self.items}


@dataclass(frozen=True)
class StudyContract:
    decision_type: DecisionType
    area: str
    baseline: str
    scenario_family: str
    requested_metrics: tuple[str, ...]
    decision_question: str = ""
    unsupported_claims: tuple[str, ...] = ()


METRIC_EVIDENCE: dict[str, set[str]] = {
    "arrivals_after_decision": {"geometry", "demand", "turn_movements"},
    "max_approach_queue_vehicles": {"geometry", "demand", "turn_movements"},
    "mean_network_speed_mps": {"geometry", "demand", "speeds"},
    "mean_instant_waiting_time_s": {"geometry", "demand", "turn_movements"},
    "rerouted_vehicles": {
        "geometry",
        "demand",
        "turn_movements",
        "alternate_route",
    },
}

DECISION_EVIDENCE: dict[DecisionType, set[str]] = {
    DecisionType.BLOCK_TURN: {"geometry", "demand", "turn_movements"},
    DecisionType.APPLY_DETOUR: {
        "geometry",
        "demand",
        "turn_movements",
        "alternate_route",
    },
}


def required_evidence(contract: StudyContract) -> tuple[set[str], list[str]]:
    unsupported_metrics = sorted(
        metric
        for metric in contract.requested_metrics
        if metric not in METRIC_EVIDENCE
    )

    required = set(DECISION_EVIDENCE.get(contract.decision_type, set()))
    for metric in contract.requested_metrics:
        required.update(METRIC_EVIDENCE.get(metric, set()))
    return required, unsupported_metrics


def evaluate_study(
    contract: StudyContract,
    observations: ObservationPackage,
) -> dict:
    required, unsupported_metrics = required_evidence(contract)
    available = observations.by_name()
    missing = sorted(name for name in required if name not in available)

    can_simulate = not missing and not unsupported_metrics
    status = (
        StudyStatus.SUPPORTED
        if can_simulate
        else StudyStatus.NEEDS_DATA
    )

    if unsupported_metrics:
        message = (
            "StreetLab needs additional evidence or a supported metric definition "
            "before simulation. Unsupported metrics: "
            + ", ".join(unsupported_metrics)
        )
    elif missing:
        message = (
            "StreetLab needs additional evidence before simulation. Missing: "
            + ", ".join(missing)
        )
    else:
        message = (
            "Study evidence is sufficient for scenario simulation under the "
            "declared provenance and assumptions."
        )

    return {
        "status": status.value,
        "can_simulate": can_simulate,
        "decision_type": contract.decision_type.value,
        "area": contract.area,
        "baseline": contract.baseline,
        "scenario_family": contract.scenario_family,
        "requested_metrics": list(contract.requested_metrics),
        "missing_evidence": missing,
        "unsupported_metrics": unsupported_metrics,
        "unsupported_claims": list(contract.unsupported_claims),
        "evidence_provenance": observations.provenance_map(),
        "message": message,
        "interpretation": (
            "Evidence sufficiency permits scenario simulation; it does not validate "
            "the assumptions as observed facts."
            if can_simulate
            else (
                "StreetLab refuses unsupported simulation instead of fabricating "
                "confidence or filling missing evidence silently."
            )
        ),
    }


def demo_observation_package() -> ObservationPackage:
    """Evidence declaration for the synthetic Decision Lab demo.

    Geometry, demand, turning movements and alternate-route availability are
    constructed scenario inputs in the toy network. Phase-1 trajectory/speed
    behavior is reused as calibrated evidence, not reclassified as observed
    evidence for this demo junction.
    """
    return ObservationPackage(
        items=(
            EvidenceItem(
                "geometry",
                EvidenceProvenance.ASSUMED,
                "Synthetic M1-M6 Decision Lab network.",
            ),
            EvidenceItem(
                "demand",
                EvidenceProvenance.ASSUMED,
                "Synthetic route demand used by the Decision Lab demo.",
            ),
            EvidenceItem(
                "turn_movements",
                EvidenceProvenance.ASSUMED,
                "Synthetic north/east movement split in the demo.",
            ),
            EvidenceItem(
                "alternate_route",
                EvidenceProvenance.ASSUMED,
                "Synthetic detour WJ -> JE -> EN -> N2 -> NS.",
            ),
            EvidenceItem(
                "trajectories",
                EvidenceProvenance.CALIBRATED,
                "Phase-1 Chennai trajectory evidence reused for calibration.",
            ),
            EvidenceItem(
                "speeds",
                EvidenceProvenance.CALIBRATED,
                "Phase-1 speed behavior reused from the frozen prototype.",
            ),
        )
    )
