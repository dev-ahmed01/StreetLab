from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import Iterable

from streetlab_phase2.study import EvidenceItem, EvidenceProvenance


class VehicleClass(str, Enum):
    MOTORCYCLE = "MOTORCYCLE"
    CAR = "CAR"
    AUTO_RICKSHAW = "AUTO_RICKSHAW"
    LIGHT_COMMERCIAL = "LIGHT_COMMERCIAL"
    HEAVY_VEHICLE = "HEAVY_VEHICLE"
    BUS = "BUS"
    BICYCLE = "BICYCLE"
    PEDESTRIAN = "PEDESTRIAN"
    OTHER = "OTHER"


class BehaviorSupport(str, Enum):
    CALIBRATED = "CALIBRATED"
    ASSUMED = "ASSUMED"


@dataclass(frozen=True)
class ClassMapping:
    source_label: str
    canonical: VehicleClass
    behavior_support: BehaviorSupport


@dataclass(frozen=True)
class TrackPoint:
    source_provider: str
    source_track_id: str
    source_class: str
    vehicle_class: VehicleClass
    behavior_support: BehaviorSupport
    provenance: EvidenceProvenance
    time_s: float | None
    x_m: float
    y_m: float
    speed_mps: float | None = None
    acceleration_mps2: float | None = None
    heading_rad: float | None = None
    movement: str | None = None
    entry_direction: str | None = None
    exit_direction: str | None = None
    frame_number: int | None = None
    confidence: float | None = None
    lane_id: str | None = None
    road_section: str | None = None
    source_timestamp: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationSummary:
    unique_tracks: int
    class_counts: dict[str, int]
    movement_counts: dict[str, int]
    mean_speed_mps: float | None
    min_time_s: float | None
    max_time_s: float | None
    duration_s: float | None


@dataclass(frozen=True)
class ObservationPackage:
    source_provider: str
    tracks: tuple[TrackPoint, ...]
    summary: ObservationSummary
    behavior_support: dict[str, str]
    persona_calibration_modified: bool = False
    site_calibration_only: bool = True

    def to_phase2_evidence(self) -> tuple[EvidenceItem, ...]:
        evidence: list[EvidenceItem] = []
        if self.tracks:
            evidence.append(
                EvidenceItem(
                    "trajectories",
                    EvidenceProvenance.OBSERVED_AUTO,
                    f"Normalized from provider: {self.source_provider}",
                )
            )

        if any(p.speed_mps is not None for p in self.tracks):
            evidence.append(
                EvidenceItem(
                    "speeds",
                    EvidenceProvenance.OBSERVED_AUTO,
                    f"Observed/extracted by provider: {self.source_provider}",
                )
            )

        if any(p.movement for p in self.tracks):
            evidence.append(
                EvidenceItem(
                    "turn_movements",
                    EvidenceProvenance.OBSERVED_AUTO,
                    f"Observed/extracted by provider: {self.source_provider}",
                )
            )

        return tuple(evidence)


def build_summary(points: Iterable[TrackPoint]) -> ObservationSummary:
    pts = list(points)
    track_class: dict[str, VehicleClass] = {}
    track_movement: dict[str, str] = {}

    for p in pts:
        track_class.setdefault(p.source_track_id, p.vehicle_class)
        if p.movement:
            track_movement.setdefault(p.source_track_id, p.movement)

    class_counts: dict[str, int] = {}
    for cls in track_class.values():
        class_counts[cls.value] = class_counts.get(cls.value, 0) + 1

    movement_counts: dict[str, int] = {}
    for movement in track_movement.values():
        movement_counts[movement] = movement_counts.get(movement, 0) + 1

    speeds = [float(p.speed_mps) for p in pts if p.speed_mps is not None]
    times = [float(p.time_s) for p in pts if p.time_s is not None]
    min_time = min(times) if times else None
    max_time = max(times) if times else None

    return ObservationSummary(
        unique_tracks=len(track_class),
        class_counts=dict(sorted(class_counts.items())),
        movement_counts=dict(sorted(movement_counts.items())),
        mean_speed_mps=(sum(speeds) / len(speeds)) if speeds else None,
        min_time_s=min_time,
        max_time_s=max_time,
        duration_s=(max_time - min_time) if min_time is not None and max_time is not None else None,
    )
