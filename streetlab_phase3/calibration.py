from __future__ import annotations

from collections.abc import Mapping, Sequence

from .adapters.fluid import FluidRouteAdapter, FluidSignalAdapter
from .models import (
    ObservationPackage,
    build_quality,
    build_route_summary,
    build_summary,
)
from .providers import ProviderRegistry


class SiteCalibrator:
    """Build site-specific observations without retraining behavior personas."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self.registry = registry or ProviderRegistry.default()

    def calibrate(
        self,
        *,
        track_rows: Sequence[Mapping[str, object]],
        signal_rows: Sequence[Mapping[str, object]] | None = None,
        route_rows: Sequence[Mapping[str, object]] | None = None,
        signal_adapter=None,
        route_adapter=None,
    ) -> ObservationPackage:
        provider = self.registry.detect_track_provider(track_rows)
        points = provider.normalize_tracks(track_rows)

        if signal_rows:
            adapter = signal_adapter or FluidSignalAdapter()
            signals = adapter.normalize_signals(signal_rows)
        else:
            signals = []

        if route_rows:
            adapter = route_adapter or FluidRouteAdapter()
            routes = adapter.normalize_routes(route_rows)
        else:
            routes = []

        behavior_support: dict[str, str] = {}
        for p in points:
            behavior_support[p.vehicle_class.value] = p.behavior_support.value

        return ObservationPackage(
            source_provider=provider.name,
            tracks=tuple(points),
            signals=tuple(signals),
            routes=tuple(routes),
            summary=build_summary(points),
            route_summary=build_route_summary(routes),
            quality=build_quality(points),
            behavior_support=dict(sorted(behavior_support.items())),
            persona_calibration_modified=False,
            site_calibration_only=True,
        )
