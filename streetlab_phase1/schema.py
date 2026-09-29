from enum import Enum
from pydantic import BaseModel, Field


class Provenance(str, Enum):
    OBSERVED_MANUAL = "OBSERVED_MANUAL"
    OBSERVED_AUTO = "OBSERVED_AUTO"
    SENSOR = "SENSOR"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"


class DatasetRole(str, Enum):
    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"


class DatasetSummary(BaseModel):
    path: str
    role: DatasetRole
    rows_raw: int
    rows_clean: int
    unique_vehicles: int
    flagged_rows_removed: int = 0
    duplicate_vehicle_time_rows: int = 0
    median_sample_seconds: float | None = None
    vehicle_class_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
