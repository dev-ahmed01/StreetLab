from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from streetlab_phase3.models import TrackPoint


Row = Mapping[str, object]


class TrajectoryAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, rows: Sequence[Row]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def normalize_tracks(self, rows: Sequence[Row]) -> list[TrackPoint]:
        raise NotImplementedError


def require_rows(rows: Sequence[Row]) -> None:
    if not rows:
        raise ValueError("Trajectory input is empty")
