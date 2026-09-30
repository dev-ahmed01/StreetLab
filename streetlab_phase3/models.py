from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
class SignalRecord:
    provider: str
    intersection_name: str | None
    direction: str
    turn: str
    state: str
    begin_time_s: float
    end_time_s: float
    duration_s: float
    cycle_id: str | None
    provenance: EvidenceProvenance = EvidenceProvenance.OBSERVED_AUTO


@dataclass(frozen=True)
class RouteRecord:
    provider: str
    source_track_id: str
    source_class: str
    vehicle_class: VehicleClass
    behavior_support: BehaviorSupport
    in_time_s: float
    out_time_s: float
    travel_time_s: float
    entry_direction: str | None
    exit_direction: str | None
    movement: str | None
    turn: str | None
    in_state: str | None
    out_state: str | None
    provenance: EvidenceProvenance = EvidenceProvenance.OBSERVED_AUTO
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
class RouteSummary:
    unique_routes: int
    movement_counts: dict[str, int]
    mean_travel_time_s: float | None


@dataclass(frozen=True)
class QualityReport:
    total_points: int
    missing_speed_points: int
    negative_speed_points: int
    extreme_speed_points: int
    unknown_class_tracks: int
    route_rows_total: int = 0
    route_rows_complete: int = 0
    route_rows_incomplete: int = 0
    persona_calibration_modified: bool = False


@dataclass(frozen=True)
class ObservationPackage:
    source_provider: str
    tracks: tuple[TrackPoint, ...]
    signals: tuple[SignalRecord, ...]
    routes: tuple[RouteRecord, ...]
    summary: ObservationSummary
    route_summary: RouteSummary
    quality: QualityReport
    behavior_support: dict[str, str]
    persona_calibration_modified: bool = False
    site_calibration_only: bool = True

    def to_phase2_evidence(self) -> tuple[EvidenceItem, ...]:
        evidence: list[EvidenceItem] = []
        if self.tracks:
            evidence.append(
                EvidenceItem(
                    "trajectories",
                    self.tracks[0].provenance,
                    f"Normalized from provider: {self.source_provider}",
                )
            )
        if any(p.speed_mps is not None for p in self.tracks):
            speed_provenances = {
                p.provenance
                for p in self.tracks
                if p.speed_mps is not None
            }
            speed_provenance = (
                next(iter(speed_provenances))
                if len(speed_provenances) == 1
                else EvidenceProvenance.INFERRED
            )
            evidence.append(
                EvidenceItem(
                    "speeds",
                    speed_provenance,
                    f"Observed/extracted by provider: {self.source_provider}",
                )
            )
        if any(p.movement for p in self.tracks):
            movement_provenances = {
                p.provenance
                for p in self.tracks
                if p.movement
            }
            movement_provenance = (
                next(iter(movement_provenances))
                if len(movement_provenances) == 1
                else EvidenceProvenance.INFERRED
            )
            evidence.append(
                EvidenceItem(
                    "turn_movements",
                    movement_provenance,
                    f"Observed/extracted by provider: {self.source_provider}",
                )
            )
        if self.signals:
            signal_provenances = {record.provenance for record in self.signals}
            evidence.append(
                EvidenceItem(
                    "signals",
                    (
                        next(iter(signal_provenances))
                        if len(signal_provenances) == 1
                        else EvidenceProvenance.INFERRED
                    ),
                    "Normalized traffic-signal timeline.",
                )
            )
        if self.routes:
            route_provenances = {record.provenance for record in self.routes}
            evidence.append(
                EvidenceItem(
                    "routes",
                    (
                        next(iter(route_provenances))
                        if len(route_provenances) == 1
                        else EvidenceProvenance.INFERRED
                    ),
                    "Normalized entry/exit route observations.",
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

    point_speeds = [
        float(p.speed_mps)
        for p in pts
        if p.speed_mps is not None
        and p.metadata.get("speed_kind") != "track_average"
    ]
    track_average_speeds: dict[str, float] = {}
    for p in pts:
        if (
            p.speed_mps is not None
            and p.metadata.get("speed_kind") == "track_average"
        ):
            track_average_speeds.setdefault(
                p.source_track_id,
                float(p.speed_mps),
            )
    speeds = point_speeds + list(track_average_speeds.values())

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
        duration_s=(max_time - min_time)
        if min_time is not None and max_time is not None
        else None,
    )


def build_route_summary(routes: Iterable[RouteRecord]) -> RouteSummary:
    rows = list(routes)
    movement_counts: dict[str, int] = {}
    for route in rows:
        if route.movement:
            movement_counts[route.movement] = movement_counts.get(route.movement, 0) + 1
    travel_times = [route.travel_time_s for route in rows]
    return RouteSummary(
        unique_routes=len(rows),
        movement_counts=dict(sorted(movement_counts.items())),
        mean_travel_time_s=(
            sum(travel_times) / len(travel_times) if travel_times else None
        ),
    )


def build_quality(
    points: Iterable[TrackPoint],
    *,
    route_rows_total: int = 0,
    route_rows_complete: int = 0,
    route_rows_incomplete: int = 0,
) -> QualityReport:
    pts = list(points)
    unknown_tracks = {
        p.source_track_id for p in pts if p.vehicle_class == VehicleClass.OTHER
    }
    speeds = [p.speed_mps for p in pts]
    return QualityReport(
        total_points=len(pts),
        missing_speed_points=sum(speed is None for speed in speeds),
        negative_speed_points=sum(
            speed is not None and speed < 0 for speed in speeds
        ),
        extreme_speed_points=sum(
            speed is not None and speed > 70.0 for speed in speeds
        ),
        unknown_class_tracks=len(unknown_tracks),
        route_rows_total=route_rows_total,
        route_rows_complete=route_rows_complete,
        route_rows_incomplete=route_rows_incomplete,
        persona_calibration_modified=False,
    )
