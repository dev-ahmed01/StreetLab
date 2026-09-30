from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import RouteRecord, SignalRecord, TrackPoint
from .base import Row, TrajectoryAdapter, require_rows


def _normalize_turn(value: object) -> str:
    raw = str(value).strip().lower()
    return {
        "s": "STRAIGHT",
        "straight": "STRAIGHT",
        "l": "LEFT",
        "left": "LEFT",
        "left/u-turn": "LEFT_OR_UTURN",
        "u-turn": "UTURN",
        "r": "RIGHT",
        "right": "RIGHT",
        "p": "PEDESTRIAN",
    }.get(raw, raw.upper())


def _normalize_state(value: object) -> str:
    raw = str(value).strip().lower()
    return {
        "g": "GREEN",
        "green": "GREEN",
        "r": "RED",
        "red": "RED",
        "y": "YELLOW",
        "yellow": "YELLOW",
        "amber": "YELLOW",
    }.get(raw, raw.upper())


def _provenance(value: str) -> EvidenceProvenance:
    try:
        return EvidenceProvenance(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported evidence provenance: {value}") from exc


def _speed_to_mps(value: object, unit: str) -> float:
    speed = float(value)
    normalized = unit.strip().lower()
    if normalized in {"m/s", "mps", "meter/second", "meters/second"}:
        return speed
    if normalized in {"km/h", "kmh", "kph"}:
        return speed / 3.6
    raise ValueError(f"Unsupported speed unit: {unit}")


@dataclass(frozen=True)
class GenericTrajectoryMapping:
    track_id: str
    vehicle_class: str
    time_s: str
    x_m: str
    y_m: str
    speed_mps: str | None = None
    speed_unit: str = "m/s"
    acceleration_mps2: str | None = None
    movement: str | None = None
    frame_number: str | None = None
    confidence: str | None = None
    lane_id: str | None = None
    road_section: str | None = None
    provenance: str = "OBSERVED_AUTO"


@dataclass(frozen=True)
class GenericSignalMapping:
    direction: str
    turn: str
    state: str
    begin_time_s: str
    end_time_s: str
    cycle_id: str | None = None
    intersection_name: str | None = None
    provenance: str = "OBSERVED_AUTO"


@dataclass(frozen=True)
class GenericRouteMapping:
    track_id: str
    vehicle_class: str
    in_time_s: str
    out_time_s: str
    entry_direction: str | None = None
    exit_direction: str | None = None
    movement: str | None = None
    turn: str | None = None
    in_state: str | None = None
    out_state: str | None = None
    provenance: str = "OBSERVED_AUTO"


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
        provenance = _provenance(m.provenance)
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
                    provenance=provenance,
                    time_s=float(row[m.time_s]),
                    x_m=float(row[m.x_m]),
                    y_m=float(row[m.y_m]),
                    speed_mps=(
                        _speed_to_mps(row[m.speed_mps], m.speed_unit)
                        if m.speed_mps and row.get(m.speed_mps) not in (None, "")
                        else None
                    ),
                    acceleration_mps2=(
                        float(row[m.acceleration_mps2])
                        if m.acceleration_mps2
                        and row.get(m.acceleration_mps2) not in (None, "")
                        else None
                    ),
                    movement=(
                        str(row[m.movement])
                        if m.movement and row.get(m.movement) not in (None, "")
                        else None
                    ),
                    frame_number=(
                        int(float(row[m.frame_number]))
                        if m.frame_number
                        and row.get(m.frame_number) not in (None, "")
                        else None
                    ),
                    confidence=(
                        float(row[m.confidence])
                        if m.confidence and row.get(m.confidence) not in (None, "")
                        else None
                    ),
                    lane_id=(
                        str(row[m.lane_id])
                        if m.lane_id and row.get(m.lane_id) not in (None, "")
                        else None
                    ),
                    road_section=(
                        str(row[m.road_section])
                        if m.road_section and row.get(m.road_section) not in (None, "")
                        else None
                    ),
                )
            )
        return result


class GenericSignalAdapter:
    name = "generic"

    def __init__(self, mapping: GenericSignalMapping) -> None:
        self.mapping = mapping

    def normalize_signals(self, rows: Sequence[Row]) -> list[SignalRecord]:
        require_rows(rows)
        m = self.mapping
        required = {m.direction, m.turn, m.state, m.begin_time_s, m.end_time_s}
        if not required.issubset(rows[0].keys()):
            raise ValueError("Generic signal rows do not satisfy the mapping")

        provenance = _provenance(m.provenance)
        result: list[SignalRecord] = []
        for row in rows:
            begin = float(row[m.begin_time_s])
            end = float(row[m.end_time_s])
            result.append(
                SignalRecord(
                    provider=self.name,
                    intersection_name=(
                        str(row[m.intersection_name])
                        if m.intersection_name
                        and row.get(m.intersection_name) not in (None, "")
                        else None
                    ),
                    direction=str(row[m.direction]),
                    turn=_normalize_turn(row[m.turn]),
                    state=_normalize_state(row[m.state]),
                    begin_time_s=begin,
                    end_time_s=end,
                    duration_s=end - begin,
                    cycle_id=(
                        str(row[m.cycle_id])
                        if m.cycle_id and row.get(m.cycle_id) not in (None, "")
                        else None
                    ),
                    provenance=provenance,
                )
            )
        return result


class GenericRouteAdapter:
    name = "generic"

    def __init__(self, mapping: GenericRouteMapping) -> None:
        self.mapping = mapping

    def normalize_routes(self, rows: Sequence[Row]) -> list[RouteRecord]:
        require_rows(rows)
        m = self.mapping
        required = {m.track_id, m.vehicle_class, m.in_time_s, m.out_time_s}
        if not required.issubset(rows[0].keys()):
            raise ValueError("Generic route rows do not satisfy the mapping")

        provenance = _provenance(m.provenance)
        result: list[RouteRecord] = []
        for row in rows:
            mapping = map_vehicle_class(row[m.vehicle_class])
            tin = float(row[m.in_time_s])
            tout = float(row[m.out_time_s])
            result.append(
                RouteRecord(
                    provider=self.name,
                    source_track_id=str(row[m.track_id]),
                    source_class=str(row[m.vehicle_class]),
                    vehicle_class=mapping.canonical,
                    behavior_support=mapping.behavior_support,
                    in_time_s=tin,
                    out_time_s=tout,
                    travel_time_s=tout - tin,
                    entry_direction=(
                        str(row[m.entry_direction])
                        if m.entry_direction
                        and row.get(m.entry_direction) not in (None, "")
                        else None
                    ),
                    exit_direction=(
                        str(row[m.exit_direction])
                        if m.exit_direction
                        and row.get(m.exit_direction) not in (None, "")
                        else None
                    ),
                    movement=(
                        str(row[m.movement])
                        if m.movement and row.get(m.movement) not in (None, "")
                        else None
                    ),
                    turn=(
                        _normalize_turn(row[m.turn])
                        if m.turn and row.get(m.turn) not in (None, "")
                        else None
                    ),
                    in_state=(
                        _normalize_state(row[m.in_state])
                        if m.in_state and row.get(m.in_state) not in (None, "")
                        else None
                    ),
                    out_state=(
                        _normalize_state(row[m.out_state])
                        if m.out_state and row.get(m.out_state) not in (None, "")
                        else None
                    ),
                    provenance=provenance,
                )
            )
        return result
