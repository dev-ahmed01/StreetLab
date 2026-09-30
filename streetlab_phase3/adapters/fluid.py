from __future__ import annotations

from math import hypot
from typing import Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import TrackPoint
from .base import Row, TrajectoryAdapter, require_rows


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

            ax = row.get("ax")
            ay = row.get("ay")
            acceleration = None
            if ax is not None and ay is not None:
                acceleration = hypot(float(ax), float(ay))

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
                    heading_rad=(
                        float(row["course"]) if row.get("course") is not None else None
                    ),
                    movement=(
                        str(row["overall_direction"])
                        if row.get("overall_direction") not in (None, "", "Unknown")
                        else None
                    ),
                    entry_direction=(
                        str(row["entry_direction"])
                        if row.get("entry_direction") not in (None, "", "Unknown")
                        else None
                    ),
                    exit_direction=(
                        str(row["exit_direction"])
                        if row.get("exit_direction") not in (None, "", "Unknown")
                        else None
                    ),
                    frame_number=(
                        int(row["frame"]) if row.get("frame") is not None else None
                    ),
                    confidence=(
                        float(row["confidence"])
                        if row.get("confidence") is not None
                        else None
                    ),
                    metadata={
                        "is_real_detection": row.get("isReal"),
                    },
                )
            )
        return result
