from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionType(str, Enum):
    BLOCK_TURN = "BLOCK_TURN"
    APPLY_DETOUR = "APPLY_DETOUR"


class ResponsePolicy(str, Enum):
    NONE = "NONE"
    NATURAL_REROUTE = "NATURAL_REROUTE"
    GUIDED_DETOUR = "GUIDED_DETOUR"
    HETEROGENEOUS_RESPONSE = "HETEROGENEOUS_RESPONSE"


@dataclass(frozen=True)
class DecisionEvent:
    decision_type: DecisionType
    at_s: float
    duration_s: float
    from_edge: str
    blocked_edge: str
    response_policy: ResponsePolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_s(self) -> float:
        return self.at_s + self.duration_s

    def active_at(self, t: float) -> bool:
        return self.at_s <= t < self.end_s


@dataclass(frozen=True)
class BranchSpec:
    name: str
    decision: DecisionEvent | None
    description: str


@dataclass(frozen=True)
class PersonaProfile:
    vehicle_class: int
    vehicle_class_name: str
    persona: str
    calibration_share: float
    speed_factor: float
    min_gap_lat_m: float
    tau_s: float = 0.50

    @property
    def type_id(self) -> str:
        return (
            f"sl_{self.vehicle_class_name.lower()}_"
            f"{self.persona.lower()}"
        )
