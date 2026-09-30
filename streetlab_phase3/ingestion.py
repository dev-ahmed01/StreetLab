from __future__ import annotations

import csv
import json
from pathlib import Path

from .adapters.generic import GenericTrajectoryAdapter, GenericTrajectoryMapping
from .calibration import SiteCalibrator
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
    ):
        rows = load_rows(track_file)
        registry = ProviderRegistry.default()
        if generic_mapping is not None:
            registry = registry.with_provider(
                GenericTrajectoryAdapter(mapping=generic_mapping)
            )

        signal_rows = load_rows(signal_file) if signal_file else None
        route_rows = load_rows(route_file) if route_file else None

        return SiteCalibrator(registry=registry).calibrate(
            track_rows=rows,
            signal_rows=signal_rows,
            route_rows=route_rows,
        )
