from __future__ import annotations

from collections.abc import Sequence

from .adapters.base import Row, TrajectoryAdapter
from .adapters.fluid import FluidAdapter
from .adapters.geotrax import GeoTraxAdapter


class ProviderRegistry:
    def __init__(self, providers: Sequence[TrajectoryAdapter]) -> None:
        self.providers = tuple(providers)

    @classmethod
    def default(cls) -> "ProviderRegistry":
        return cls((FluidAdapter(), GeoTraxAdapter()))

    def with_provider(self, provider: TrajectoryAdapter) -> "ProviderRegistry":
        return ProviderRegistry((*self.providers, provider))

    def detect_track_provider(self, rows: Sequence[Row]) -> TrajectoryAdapter:
        matches = [provider for provider in self.providers if provider.can_handle(rows)]
        if not matches:
            raise ValueError(
                "Unsupported trajectory schema. Use a supported provider export "
                "or supply an explicit GenericTrajectoryMapping."
            )
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise ValueError(f"Ambiguous trajectory schema matched providers: {names}")
        return matches[0]
