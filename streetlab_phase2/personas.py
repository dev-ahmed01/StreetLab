from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .models import PersonaProfile


PHYSICAL = {
    1: {"length_m": 1.80, "width_m": 0.60, "vclass": "motorcycle"},
    2: {"length_m": 4.53, "width_m": 1.52, "vclass": "passenger"},
    6: {"length_m": 2.60, "width_m": 1.40, "vclass": "passenger"},
}


def load_phase1_personas(path: str | Path) -> list[PersonaProfile]:
    """Load Phase-1 priors while freezing longitudinal tau to H0=0.50 s."""
    path = Path(path)
    rows: list[PersonaProfile] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cls = int(r["vehicle_class"])
            if cls not in PHYSICAL:
                continue
            rows.append(
                PersonaProfile(
                    vehicle_class=cls,
                    vehicle_class_name=r["vehicle_class_name"],
                    persona=r["persona"],
                    calibration_share=float(r["calibration_share"]),
                    speed_factor=float(r["speedFactor_seed"]),
                    min_gap_lat_m=float(r["minGapLat_seed_m"]),
                    tau_s=0.50,
                )
            )
    if not rows:
        raise ValueError(f"No supported personas found in {path}")
    return rows


def profiles_by_class(
    profiles: list[PersonaProfile],
) -> dict[int, list[PersonaProfile]]:
    out: dict[int, list[PersonaProfile]] = {}
    for p in profiles:
        out.setdefault(p.vehicle_class, []).append(p)
    for cls, vals in out.items():
        total = sum(max(v.calibration_share, 0.0) for v in vals)
        if total <= 0:
            raise ValueError(f"Class {cls} has no positive persona share")
    return out


def deterministic_persona(
    vehicle_id: str,
    vehicle_class: int,
    grouped: dict[int, list[PersonaProfile]],
) -> PersonaProfile:
    """Stable weighted assignment so branch comparisons use identical drivers."""
    vals = grouped[vehicle_class]
    h = hashlib.sha256(vehicle_id.encode("utf-8")).hexdigest()
    u = int(h[:12], 16) / float(16**12 - 1)

    total = sum(v.calibration_share for v in vals)
    acc = 0.0
    for p in vals:
        acc += p.calibration_share / total
        if u <= acc:
            return p
    return vals[-1]
