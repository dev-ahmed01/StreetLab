from __future__ import annotations

from typing import Sequence

from streetlab_phase2.study import EvidenceProvenance

from ..class_mapping import map_vehicle_class
from ..models import TrackPoint
from .base import Row, TrajectoryAdapter, require_rows


_GEOTRAX_CLASS = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "motorcycle",
    "0": "car",
    "1": "bus",
    "2": "truck",
    "3": "motorcycle",
}


class GeoTraxAdapter(TrajectoryAdapter):
    name = "geotrax"

    REQUIRED = {
        "Vehicle_ID",
        "Vehicle_Class",
        "Local_X",
        "Local_Y",
        "Vehicle_Speed",
    }

    def can_handle(self, rows: Sequence[Row]) -> bool:
        return bool(rows) and self.REQUIRED.issubset(rows[0].keys())

    def normalize_tracks(self, rows: Sequence[Row]) -> list[TrackPoint]:
        require_rows(rows)
        if not self.can_handle(rows):
            raise ValueError("Rows do not match the Geo-trax trajectory schema")

        result: list[TrackPoint] = []
        for row in rows:
            raw_class = row.get("Vehicle_Class", "unknown")
            label = _GEOTRAX_CLASS.get(raw_class, str(raw_class))
            mapping = map_vehicle_class(label)

            result.append(
                TrackPoint(
                    source_provider=self.name,
                    source_track_id=str(row["Vehicle_ID"]),
                    source_class=str(raw_class),
                    vehicle_class=mapping.canonical,
                    behavior_support=mapping.behavior_support,
                    provenance=EvidenceProvenance.OBSERVED_AUTO,
                    time_s=None,
                    x_m=float(row["Local_X"]),
                    y_m=float(row["Local_Y"]),
                    # Geo-trax publishes Vehicle_Speed in km/h.
                    speed_mps=float(row["Vehicle_Speed"]) / 3.6,
                    acceleration_mps2=(
                        float(row["Vehicle_Acceleration"])
                        if row.get("Vehicle_Acceleration") is not None
                        else None
                    ),
                    frame_number=(
                        int(row["Frame_Number"])
                        if row.get("Frame_Number") is not None
                        else None
                    ),
                    lane_id=(
                        str(row["Lane_Number"])
                        if row.get("Lane_Number") is not None
                        else None
                    ),
                    road_section=(
                        str(row["Road_Section"])
                        if row.get("Road_Section") is not None
                        else None
                    ),
                    source_timestamp=(
                        str(row["Timestamp"])
                        if row.get("Timestamp") not in (None, "")
                        else None
                    ),
                    metadata={
                        "visibility": row.get("Visibility"),
                        "latitude": row.get("Latitude"),
                        "longitude": row.get("Longitude"),
                    },
                )
            )
        return result
