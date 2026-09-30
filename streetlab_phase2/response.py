from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResponseMode(str, Enum):
    GUIDED = "GUIDED"
    LOCAL = "LOCAL"


@dataclass(frozen=True)
class ResponseAssumptions:
    """Explicit M3 scenario assumptions for route-response heterogeneity.

    These values are not learned from Phase 1. They are scenario inputs used
    to test sensitivity of counterfactual outcomes to mixed driver response.
    """

    guided_share: float = 0.50
    seed: str = "streetlab-m3"
    local_trigger_position_m: float = 150.0
    provenance: str = "ASSUMED"

    def __post_init__(self) -> None:
        if not 0.0 <= self.guided_share <= 1.0:
            raise ValueError("guided_share must be between 0 and 1")
        if self.local_trigger_position_m < 0.0:
            raise ValueError("local_trigger_position_m must be >= 0")
        if self.provenance != "ASSUMED":
            raise ValueError("M3 response assumptions must use provenance='ASSUMED'")

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> "ResponseAssumptions":
        metadata = metadata or {}
        raw = metadata.get("response_assumptions", {})
        if not isinstance(raw, dict):
            raise ValueError("response_assumptions metadata must be an object")
        return cls(
            guided_share=float(raw.get("guided_share", 0.50)),
            seed=str(raw.get("seed", "streetlab-m3")),
            local_trigger_position_m=float(
                raw.get("local_trigger_position_m", 150.0)
            ),
            provenance="ASSUMED",
        )

    def as_provenance(self) -> dict[str, Any]:
        return {
            "guided_share": self.guided_share,
            "seed": self.seed,
            "local_trigger_position_m": self.local_trigger_position_m,
            "provenance": self.provenance,
            "claim": (
                "Scenario assumption for counterfactual sensitivity; "
                "not learned from Phase-1 observations."
            ),
        }


def deterministic_response_mode(
    vehicle_id: str,
    vehicle_type_id: str,
    assumptions: ResponseAssumptions,
) -> ResponseMode:
    """Assign a stable assumed response mode across counterfactual branches."""
    key = f"{assumptions.seed}|{vehicle_id}|{vehicle_type_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    u = int(digest[:12], 16) / float(16**12 - 1)
    return (
        ResponseMode.GUIDED
        if u < assumptions.guided_share
        else ResponseMode.LOCAL
    )


def apply_heterogeneous_response(
    conn,
    assumptions: ResponseAssumptions,
    assignments: dict[str, ResponseMode],
) -> set[str]:
    """Apply mixed guided/local rerouting to northbound vehicles on WJ.

    Guided responders divert as soon as they are observed upstream. Local
    responders use the same late-discovery trigger as M1's natural response.
    """

    target = ["WJ", "JE", "EN", "N2", "NS"]
    changed: set[str] = set()

    for raw_vid in list(conn.vehicle.getIDList()):
        vid = str(raw_vid)
        if not vid.startswith("n_") or conn.vehicle.getRoadID(vid) != "WJ":
            continue

        if vid not in assignments:
            type_id = str(conn.vehicle.getTypeID(vid))
            assignments[vid] = deterministic_response_mode(
                vid, type_id, assumptions
            )

        mode = assignments[vid]
        if mode == ResponseMode.LOCAL:
            if float(conn.vehicle.getLanePosition(vid)) < assumptions.local_trigger_position_m:
                continue

        try:
            current = list(conn.vehicle.getRoute(vid))
            if current != target:
                conn.vehicle.setRoute(vid, target)
                changed.add(vid)
        except Exception:
            continue

    return changed
