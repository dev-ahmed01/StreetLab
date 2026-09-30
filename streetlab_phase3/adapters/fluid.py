from __future__ import annotations

from math import hypot
from typing import Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import RouteRecord, SignalRecord, TrackPoint
from .base import Row, TrajectoryAdapter, require_rows


def _present(value: object) -> bool:
    return value not in (None, "")


def _float_or_none(value: object) -> float | None:
    if not _present(value):
        return None
    return float(value)


def _direction(value: object) -> str | None:
    if not _present(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "unknown":
        return None
    return raw


def _movement(value: object) -> str | None:
    if not _present(value):
        return None
    raw = str(value).strip()
    if not raw or "unknown" in raw.lower():
        return None
    return raw


def _turn(value: object) -> str | None:
    if not _present(value):
        return None
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


def _state(value: object) -> str | None:
    if not _present(value):
        return None
    raw = str(value).strip()
    return {
        "G": "GREEN",
        "g": "GREEN",
        "r": "RED",
        "R": "RED",
        "y": "YELLOW",
        "Y": "YELLOW",
    }.get(raw, raw.upper())


class FluidAdapter(TrajectoryAdapter):
    name = "fluid"
    REQUIRED = {"id", "type", "cx_m", "cy_m", "time"}

    def can_handle(self, rows: Sequence[Row]) -> bool:
        return bool(rows) and self.REQUIRED.issubset(rows[0].keys())

    def normalize_tracks(self, rows: Sequence[Row]) -> list[TrackPoint]:
        require_rows(rows)
        if not self.can_handle(rows):
            raise ValueError("Rows do not match the FLUID trajectory schema")

        result: list[TrackPoint] = []
        for row in rows:
            mapping = map_vehicle_class(row.get("type", "unknown"))
            speed = row.get("speed_smooth")
            if not _present(speed):
                speed = row.get("speed")

            ax = _float_or_none(row.get("ax"))
            ay = _float_or_none(row.get("ay"))
            acceleration = hypot(ax, ay) if ax is not None and ay is not None else None

            raw_movement = (
                str(row["overall_direction"])
                if _present(row.get("overall_direction"))
                else None
            )

            result.append(
                TrackPoint(
                    source_provider=self.name,
                    source_track_id=str(row["id"]),
                    source_class=str(row.get("type", "unknown")),
                    vehicle_class=mapping.canonical,
                    behavior_support=mapping.behavior_support,
                    provenance=EvidenceProvenance.OBSERVED_AUTO,
                    time_s=float(row["time"]),
                    x_m=float(row["cx_m"]),
                    y_m=float(row["cy_m"]),
                    speed_mps=_float_or_none(speed),
                    acceleration_mps2=acceleration,
                    heading_rad=_float_or_none(row.get("course")),
                    movement=_movement(row.get("overall_direction")),
                    entry_direction=_direction(row.get("entry_direction")),
                    exit_direction=_direction(row.get("exit_direction")),
                    frame_number=(
                        int(float(row["frame"]))
                        if _present(row.get("frame"))
                        else None
                    ),
                    confidence=_float_or_none(row.get("confidence")),
                    metadata={
                        "is_real_detection": row.get("isReal"),
                        "raw_movement": raw_movement,
                    },
                )
            )
        return result


class FluidSignalAdapter:
    name = "fluid"

    def normalize_signals(self, rows: Sequence[Row]) -> list[SignalRecord]:
        require_rows(rows)
        required = {"direction", "turn", "state", "begin_time", "end_time"}
        if not required.issubset(rows[0].keys()):
            raise ValueError("Rows do not match the FLUID signal schema")

        result: list[SignalRecord] = []
        for row in rows:
            begin = float(row["begin_time"])
            end = float(row["end_time"])
            duration = (
                float(row["duration"])
                if _present(row.get("duration"))
                else end - begin
            )
            result.append(
                SignalRecord(
                    provider=self.name,
                    intersection_name=(
                        str(row["name"]) if _present(row.get("name")) else None
                    ),
                    direction=str(row["direction"]),
                    turn=_turn(row["turn"]) or "UNKNOWN",
                    state=_state(row["state"]) or "UNKNOWN",
                    begin_time_s=begin,
                    end_time_s=end,
                    duration_s=duration,
                    cycle_id=(
                        str(row["cycle"]) if _present(row.get("cycle")) else None
                    ),
                )
            )
        return result


class FluidRouteAdapter:
    name = "fluid"

    def __init__(self) -> None:
        self.last_total_rows = 0
        self.last_complete_rows = 0
        self.last_incomplete_rows = 0

    def normalize_routes(self, rows: Sequence[Row]) -> list[RouteRecord]:
        require_rows(rows)
        required = {"id", "in_time", "out_time", "type"}
        if not required.issubset(rows[0].keys()):
            raise ValueError("Rows do not match the FLUID route schema")

        self.last_total_rows = len(rows)
        self.last_complete_rows = 0
        self.last_incomplete_rows = 0

        result: list[RouteRecord] = []
        for row in rows:
            in_time = _float_or_none(row.get("in_time"))
            out_time = _float_or_none(row.get("out_time"))
            movement = _movement(row.get("overall_direction"))
            entry = _direction(row.get("entry_direction"))
            exit_direction = _direction(row.get("exit_direction"))

            if (
                in_time is None
                or out_time is None
                or movement is None
                or entry is None
                or exit_direction is None
            ):
                self.last_incomplete_rows += 1
                continue

            self.last_complete_rows += 1
            mapping = map_vehicle_class(row["type"])
            result.append(
                RouteRecord(
                    provider=self.name,
                    source_track_id=str(row["id"]),
                    source_class=str(row["type"]),
                    vehicle_class=mapping.canonical,
                    behavior_support=mapping.behavior_support,
                    in_time_s=in_time,
                    out_time_s=out_time,
                    travel_time_s=out_time - in_time,
                    entry_direction=entry,
                    exit_direction=exit_direction,
                    movement=movement,
                    turn=_turn(row.get("turn")),
                    in_state=_state(row.get("in_state")),
                    out_state=_state(row.get("out_state")),
                    metadata={"geometry": row.get("geometry")},
                )
            )

        return result
