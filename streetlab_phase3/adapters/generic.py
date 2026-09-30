from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import TrackPoint
from .base import Row, TrajectoryAdapter, require_rows


@dataclass(frozen=True)
class GenericTrajectoryMapping:
    track_id: str
    vehicle_class: str
    time_s: str
    x_m: str
    y_m: str
    speed_mps: str | None = None
    acceleration_mps2: str | None = None
    movement: str | None = None
    frame_number: str | None = None
    confidence: str | None = None
    lane_id: str | None = None
    road_section: str | None = None


class GenericTrajectoryAdapter(TrajectoryAdapter):
    name = "generic"

    def __init__(self, mapping: GenericTrajectoryMapping | None = None) -> None:
        self.mapping = mapping

    def can_handle(self, rows: Sequence[Row]) -> bool:
        if not rows or self.mapping is None:
            return False
        required = {
            self.mapping.track_id,
            self.mapping.vehicle_class,
            self.mapping.time_s,
            self.mapping.x_m,
            self.mapping.y_m,
        }
        return required.issubset(rows[0].keys())

    def normalize_tracks(self, rows: Sequence[Row]) -> list[TrackPoint]:
        require_rows(rows)
        if self.mapping is None:
            raise ValueError(
                "Generic trajectory input requires an explicit column mapping"
            )
        if not self.can_handle(rows):
            raise ValueError("Generic trajectory rows do not satisfy the mapping")

        m = self.mapping
        result: list[TrackPoint] = []
        for row in rows:
            mapping = map_vehicle_class(row[m.vehicle_class])
            result.append(
                TrackPoint(
                    source_provider=self.name,
                    source_track_id=str(row[m.track_id]),
                    source_class=str(row[m.vehicle_class]),
                    vehicle_class=mapping.canonical,
                    behavior_support=mapping.behavior_support,
                    provenance=EvidenceProvenance.OBSERVED_AUTO,
                    time_s=float(row[m.time_s]),
                    x_m=float(row[m.x_m]),
                    y_m=float(row[m.y_m]),
                    speed_mps=(
                        float(row[m.speed_mps])
                        if m.speed_mps and row.get(m.speed_mps) is not None
                        else None
                    ),
                    acceleration_mps2=(
                        float(row[m.acceleration_mps2])
                        if m.acceleration_mps2
                        and row.get(m.acceleration_mps2) is not None
                        else None
                    ),
                    movement=(
                        str(row[m.movement])
                        if m.movement and row.get(m.movement) not in (None, "")
                        else None
                    ),
                    frame_number=(
                        int(row[m.frame_number])
                        if m.frame_number
                        and row.get(m.frame_number) is not None
                        else None
                    ),
                    confidence=(
                        float(row[m.confidence])
                        if m.confidence and row.get(m.confidence) is not None
                        else None
                    ),
                    lane_id=(
                        str(row[m.lane_id])
                        if m.lane_id and row.get(m.lane_id) is not None
                        else None
                    ),
                    road_section=(
                        str(row[m.road_section])
                        if m.road_section and row.get(m.road_section) is not None
                        else None
                    ),
                )
            )
        return result
