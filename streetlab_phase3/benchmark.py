from __future__ import annotations

from dataclasses import dataclass

from .adapters.fluid_telemetry import FluidTelemetryAdapter, TelemetrySummary, summarize_telemetry
from .calibration import SiteCalibrator


@dataclass(frozen=True)
class FluidBenchmarkReport:
    study_id: str
    unique_tracks: int
    trajectory_points: int
    trajectory_start_s: float | None
    trajectory_end_s: float | None
    trajectory_duration_s: float | None
    class_counts: dict[str, int]
    known_movement_tracks: int
    movement_counts: dict[str, int]
    complete_routes: int
    incomplete_routes: int
    mean_route_travel_time_s: float | None
    signal_records: int
    signal_coverage_end_s: float | None
    telemetry: TelemetrySummary
    persona_calibration_modified: bool


class FluidBenchmark:
    def build(
        self,
        *,
        study_id: str,
        track_rows,
        signal_rows=None,
        route_rows=None,
        telemetry_rows=None,
    ) -> FluidBenchmarkReport:
        package = SiteCalibrator().calibrate(
            track_rows=track_rows,
            signal_rows=signal_rows,
            route_rows=route_rows,
        )
        telemetry_samples = (
            FluidTelemetryAdapter().normalize(telemetry_rows)
            if telemetry_rows
            else []
        )
        telemetry = summarize_telemetry(telemetry_samples)

        signal_end = (
            max(record.end_time_s for record in package.signals)
            if package.signals
            else None
        )

        return FluidBenchmarkReport(
            study_id=study_id,
            unique_tracks=package.summary.unique_tracks,
            trajectory_points=package.quality.total_points,
            trajectory_start_s=package.summary.min_time_s,
            trajectory_end_s=package.summary.max_time_s,
            trajectory_duration_s=package.summary.duration_s,
            class_counts=package.summary.class_counts,
            known_movement_tracks=sum(package.summary.movement_counts.values()),
            movement_counts=package.summary.movement_counts,
            complete_routes=package.quality.route_rows_complete,
            incomplete_routes=package.quality.route_rows_incomplete,
            mean_route_travel_time_s=package.route_summary.mean_travel_time_s,
            signal_records=len(package.signals),
            signal_coverage_end_s=signal_end,
            telemetry=telemetry,
            persona_calibration_modified=package.persona_calibration_modified,
        )
