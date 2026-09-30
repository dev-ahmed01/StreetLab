from __future__ import annotations

from math import hypot
from typing import Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import RouteRecord, SignalRecord, TrackPoint
from .base import Row, TrajectoryAdapter, require_rows


def _turn(value: object) -> str:
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


def _state(value: object) -> str:
    raw = str(value).strip()
    return {"G": "GREEN", "g": "GREEN", "r": "RED", "R": "RED", "y": "YELLOW", "Y": "YELLOW"}.get(raw, raw.upper())


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
            if speed is None:
                speed = row.get("speed")
            ax, ay = row.get("ax"), row.get("ay")
            acceleration = (
                hypot(float(ax), float(ay))
                if ax is not None and ay is not None
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
                    speed_mps=float(speed) if speed is not None else None,
                    acceleration_mps2=acceleration,
                    heading_rad=float(row["course"]) if row.get("course") is not None else None,
                    movement=str(row["overall_direction"]) if row.get("overall_direction") not in (None, "", "Unknown") else None,
                    entry_direction=str(row["entry_direction"]) if row.get("entry_direction") not in (None, "", "Unknown") else None,
                    exit_direction=str(row["exit_direction"]) if row.get("exit_direction") not in (None, "", "Unknown") else None,
                    frame_number=int(row["frame"]) if row.get("frame") is not None else None,
                    confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
                    metadata={"is_real_detection": row.get("isReal")},
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
        return [
            SignalRecord(
                provider=self.name,
                intersection_name=str(row["name"]) if row.get("name") is not None else None,
                direction=str(row["direction"]),
                turn=_turn(row["turn"]),
                state=_state(row["state"]),
                begin_time_s=float(row["begin_time"]),
                end_time_s=float(row["end_time"]),
                duration_s=float(row.get("duration", float(row["end_time"]) - float(row["begin_time"]))),
                cycle_id=str(row["cycle"]) if row.get("cycle") is not None else None,
            )
            for row in rows
        ]


class FluidRouteAdapter:
    name = "fluid"

    def normalize_routes(self, rows: Sequence[Row]) -> list[RouteRecord]:
        require_rows(rows)
        required = {"id", "in_time", "out_time", "type"}
        if not required.issubset(rows[0].keys()):
            raise ValueError("Rows do not match the FLUID route schema")
        result: list[RouteRecord] = []
        for row in rows:
            mapping = map_vehicle_class(row["type"])
            in_time = float(row["in_time"])
            out_time = float(row["out_time"])
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
                    entry_direction=str(row["entry_direction"]) if row.get("entry_direction") not in (None, "") else None,
                    exit_direction=str(row["exit_direction"]) if row.get("exit_direction") not in (None, "") else None,
                    movement=str(row["overall_direction"]) if row.get("overall_direction") not in (None, "") else None,
                    turn=_turn(row["turn"]) if row.get("turn") is not None else None,
                    in_state=_state(row["in_state"]) if row.get("in_state") is not None else None,
                    out_state=_state(row["out_state"]) if row.get("out_state") is not None else None,
                    metadata={"geometry": row.get("geometry")},
                )
            )
        return result
