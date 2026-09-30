from __future__ import annotations

import csv
import json
from pathlib import Path

from .adapters.generic import (
    GenericRouteAdapter,
    GenericRouteMapping,
    GenericSignalAdapter,
    GenericSignalMapping,
    GenericTrajectoryAdapter,
    GenericTrajectoryMapping,
)
from .adapters.simjam import SimJamBundleAdapter
from .calibration import SiteCalibrator
from .models import ObservationPackage, build_quality, build_route_summary, build_summary
from .providers import ProviderRegistry


def load_rows(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("JSON trajectory input must be a list of objects")
        return payload
    raise ValueError(f"Unsupported input format: {source.suffix}")


class CalibrationPipeline:
    def calibrate_files(
        self,
        *,
        track_file: str | Path,
        signal_file: str | Path | None = None,
        route_file: str | Path | None = None,
        generic_mapping: GenericTrajectoryMapping | None = None,
        generic_signal_mapping: GenericSignalMapping | None = None,
        generic_route_mapping: GenericRouteMapping | None = None,
    ) -> ObservationPackage:
        rows = load_rows(track_file)
        registry = ProviderRegistry.default()
        if generic_mapping is not None:
            registry = registry.with_provider(
                GenericTrajectoryAdapter(mapping=generic_mapping)
            )

        signal_rows = load_rows(signal_file) if signal_file else None
        route_rows = load_rows(route_file) if route_file else None

        signal_adapter = (
            GenericSignalAdapter(generic_signal_mapping)
            if generic_signal_mapping is not None
            else None
        )
        route_adapter = (
            GenericRouteAdapter(generic_route_mapping)
            if generic_route_mapping is not None
            else None
        )

        return SiteCalibrator(registry=registry).calibrate(
            track_rows=rows,
            signal_rows=signal_rows,
            route_rows=route_rows,
            signal_adapter=signal_adapter,
            route_adapter=route_adapter,
        )

    def calibrate_simjam_files(
        self,
        *,
        track_file: str | Path,
        summary_file: str | Path,
    ) -> ObservationPackage:
        track_rows = load_rows(track_file)
        summary_rows = load_rows(summary_file)
        points = SimJamBundleAdapter().normalize_bundle(track_rows, summary_rows)

        behavior_support: dict[str, str] = {}
        for p in points:
            behavior_support[p.vehicle_class.value] = p.behavior_support.value

        return ObservationPackage(
            source_provider="simjam",
            tracks=tuple(points),
            signals=tuple(),
            routes=tuple(),
            summary=build_summary(points),
            route_summary=build_route_summary(()),
            quality=build_quality(points),
            behavior_support=dict(sorted(behavior_support.items())),
            persona_calibration_modified=False,
            site_calibration_only=True,
        )
