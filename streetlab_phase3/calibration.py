from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import ObservationPackage, build_summary
from .providers import ProviderRegistry


class SiteCalibrator:
    """Build site-specific observations without retraining behavior personas."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self.registry = registry or ProviderRegistry.default()

    def calibrate(
        self,
        *,
        track_rows: Sequence[Mapping[str, object]],
    ) -> ObservationPackage:
        provider = self.registry.detect_track_provider(track_rows)
        points = provider.normalize_tracks(track_rows)
        summary = build_summary(points)

        behavior_support: dict[str, str] = {}
        for p in points:
            behavior_support[p.vehicle_class.value] = p.behavior_support.value

        return ObservationPackage(
            source_provider=provider.name,
            tracks=tuple(points),
            summary=summary,
            behavior_support=dict(sorted(behavior_support.items())),
            persona_calibration_modified=False,
            site_calibration_only=True,
        )
