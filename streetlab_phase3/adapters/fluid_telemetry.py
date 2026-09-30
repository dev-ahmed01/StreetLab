from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class TelemetrySample:
    time_ms: int
    datetime_local: str | None
    height_m: float | None
    speed_mps: float | None
    compass_heading_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    recording: bool
    gimbal_heading_deg: float | None
    gimbal_pitch_deg: float | None
    gimbal_roll_deg: float | None
    battery_percent: float | None
    flight_state: str | None


@dataclass(frozen=True)
class TelemetrySummary:
    total_samples: int
    recording_samples: int
    recording_fraction: float | None
    mean_height_m: float | None
    min_height_m: float | None
    max_height_m: float | None
    mean_gimbal_pitch_deg: float | None


def _float(value):
    if value in (None, ""):
        return None
    return float(value)


class FluidTelemetryAdapter:
    def normalize(self, rows: Sequence[dict[str, object]]) -> list[TelemetrySample]:
        result: list[TelemetrySample] = []
        for row in rows:
            height_ft = _float(row.get("height_above_takeoff(feet)"))
            speed_mph = _float(row.get("speed(mph)"))
            result.append(
                TelemetrySample(
                    time_ms=int(float(row["time(millisecond)"])),
                    datetime_local=(
                        str(row["datetime"]) if row.get("datetime") not in (None, "") else None
                    ),
                    height_m=height_ft * 0.3048 if height_ft is not None else None,
                    speed_mps=speed_mph * 0.44704 if speed_mph is not None else None,
                    compass_heading_deg=_float(row.get("compass_heading(degrees)")),
                    pitch_deg=_float(row.get("pitch(degrees)")),
                    roll_deg=_float(row.get("roll(degrees)")),
                    recording=str(row.get("isVideo", "0")).strip() in {"1", "true", "True"},
                    gimbal_heading_deg=_float(row.get("gimbal_heading(degrees)")),
                    gimbal_pitch_deg=_float(row.get("gimbal_pitch(degrees)")),
                    gimbal_roll_deg=_float(row.get("gimbal_roll(degrees)")),
                    battery_percent=_float(row.get("battery_percent")),
                    flight_state=(
                        str(row["flycState"]) if row.get("flycState") not in (None, "") else None
                    ),
                )
            )
        return result


def summarize_telemetry(samples: Sequence[TelemetrySample]) -> TelemetrySummary:
    heights = [x.height_m for x in samples if x.height_m is not None]
    gimbal = [x.gimbal_pitch_deg for x in samples if x.gimbal_pitch_deg is not None]
    recording = sum(1 for x in samples if x.recording)
    return TelemetrySummary(
        total_samples=len(samples),
        recording_samples=recording,
        recording_fraction=(recording / len(samples)) if samples else None,
        mean_height_m=mean(heights) if heights else None,
        min_height_m=min(heights) if heights else None,
        max_height_m=max(heights) if heights else None,
        mean_gimbal_pitch_deg=mean(gimbal) if gimbal else None,
    )
