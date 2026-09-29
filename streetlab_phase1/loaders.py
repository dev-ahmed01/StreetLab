from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

STANDARD = {
    "timestamp": ["timestamp", "time", "time_sec", "new_timestamp", "newtimestamp", "global_time"],
    "vehicle_id": ["vehicle_id", "vehicleid", "veh_id", "vehid", "id", "vehicle_number", "vehiclenumber"],
    "vehicle_class": ["vehicle_class", "vehicleclass", "veh_class", "vehclass", "type", "vehicle_type", "vehicletype"],
    "length": ["length", "length_m", "vehicle_length", "vehiclelength"],
    "width": ["width", "width_m", "vehicle_width", "vehiclewidth"],
    "long_pos": ["long_smooth", "longitudinal_position", "longitudinalposition", "long_pos", "longpos", "long_distance", "long_distance_m", "local_y", "localy", "x_position", "xposition"],
    "long_speed": ["v_smooth", "longitudinal_speed", "longitudinalspeed", "long_speed", "long_speed_m_sec", "speed", "velocity", "v_vel", "vvel"],
    "long_accel": ["a_smooth", "longitudinal_acceleration", "longitudinalacceleration", "long_acc", "long_acc_m_sec2", "acceleration", "v_acc", "vacc"],
    "lat_pos": ["latright_smooth", "lateral_position", "lateralposition", "lat_pos", "latpos", "lat_distance", "lat_distance_m", "local_x", "localx", "y_position", "yposition"],
    "lat_speed": ["vy_smooth", "lateral_speed", "lateralspeed", "lat_speed", "latspeed", "lat_speed_m_sec"],
    "lat_accel": ["ay_smooth", "lateral_acceleration", "lateralacceleration", "lat_accel", "lataccel", "lat_acc", "lat_acc_m_sec2"],
    "flag": ["flag", "manual_flag", "correction_flag", "corrected_flag"]
}

REQUIRED = [
    "timestamp", "vehicle_id", "vehicle_class", "length", "width",
    "long_pos", "long_speed", "long_accel", "lat_pos", "lat_speed", "lat_accel"
]


def _key(name: object) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _choose_sheet(path: Path) -> pd.DataFrame:
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if len(df.columns) >= 5 and len(df) > 0:
            return df
    raise ValueError(f"No non-empty tabular sheet found in {path}")


def load_trajectory_xlsx(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _choose_sheet(path)
    normalized = {_key(c): c for c in raw.columns}
    rename: dict[object, str] = {}
    mapped_from: dict[str, str] = {}

    for standard_name, aliases in STANDARD.items():
        matches = []
        for alias in aliases:
            k = _key(alias)
            if k in normalized:
                matches.append(normalized[k])
        if matches:
            source = matches[0]
            rename[source] = standard_name
            mapped_from[standard_name] = str(source)

    df = raw.rename(columns=rename).copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Could not map required columns: " + ", ".join(missing) +
            "\nFound columns: " + ", ".join(map(str, raw.columns))
        )

    keep = REQUIRED + (["flag"] if "flag" in df.columns else [])
    df = df[keep].copy()
    for c in ["timestamp", "vehicle_id", "vehicle_class", "length", "width", "long_pos", "long_speed", "long_accel", "lat_pos", "lat_speed", "lat_accel"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "flag" in df.columns:
        df["flag"] = pd.to_numeric(df["flag"], errors="coerce").fillna(0)
    return df, mapped_from
