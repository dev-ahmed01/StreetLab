from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .models import VehicleClass


_GEOTRAX_CLASSES = {
    0: VehicleClass.CAR.value,
    1: VehicleClass.BUS.value,
    2: VehicleClass.HEAVY_VEHICLE.value,
    3: VehicleClass.MOTORCYCLE.value,
}


@dataclass(frozen=True, slots=True)
class GeoTraxPixelPoint:
    frame: int
    track_id: str
    x_px: float
    y_px: float
    x_stabilized_px: float
    y_stabilized_px: float
    vehicle_class: str
    confidence: float | None


def load_geotrax_pixel_tracks(path: str | Path) -> list[GeoTraxPixelPoint]:
    source = Path(path)
    result: list[GeoTraxPixelPoint] = []

    with source.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 14:
                raise ValueError(
                    f"Geo-trax tracking row {line_no} has {len(row)} columns; "
                    "expected at least 14 columns"
                )
            class_id = int(float(row[10]))
            vehicle_class = _GEOTRAX_CLASSES.get(class_id, VehicleClass.OTHER.value)
            confidence = float(row[11]) if row[11] not in ("", None) else None
            result.append(
                GeoTraxPixelPoint(
                    frame=int(float(row[0])),
                    track_id=str(int(float(row[1]))),
                    x_px=float(row[2]),
                    y_px=float(row[3]),
                    x_stabilized_px=float(row[6]),
                    y_stabilized_px=float(row[7]),
                    vehicle_class=vehicle_class,
                    confidence=confidence,
                )
            )
    return result
