from __future__ import annotations

import csv
import json
from pathlib import Path

from .geotrax_pixel import load_geotrax_pixel_tracks
from .pixel_benchmark import PixelBenchmark, normalize_fluid_pixel_truth
from .serialization import package_to_dict


def _read_csv(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def run_pixel_benchmark(
    *,
    geotrax_tracks: str | Path,
    fluid_tracks: str | Path,
    output: str | Path,
    max_distance_px: float = 50.0,
):
    predicted = load_geotrax_pixel_tracks(geotrax_tracks)
    truth = normalize_fluid_pixel_truth(_read_csv(fluid_tracks))
    report = PixelBenchmark(max_distance_px=max_distance_px).evaluate(
        predicted,
        truth,
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(package_to_dict(report), indent=2),
        encoding="utf-8",
    )
    return report
