from __future__ import annotations

from .models import BehaviorSupport, ClassMapping, VehicleClass


def _normalize(label: object) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )


_CALIBRATED = {
    VehicleClass.MOTORCYCLE,
    VehicleClass.CAR,
    VehicleClass.AUTO_RICKSHAW,
}

_ALIAS_MAP: dict[str, VehicleClass] = {
    # Two-wheelers
    "motorcycle": VehicleClass.MOTORCYCLE,
    "motorbike": VehicleClass.MOTORCYCLE,
    "bike": VehicleClass.MOTORCYCLE,
    "moped": VehicleClass.MOTORCYCLE,
    "scooter": VehicleClass.MOTORCYCLE,
    "2w": VehicleClass.MOTORCYCLE,
    "two wheeler": VehicleClass.MOTORCYCLE,

    # Cars
    "car": VehicleClass.CAR,
    "taxi": VehicleClass.CAR,
    "sedan": VehicleClass.CAR,

    # Indian auto-rickshaw / three-wheeler aliases
    "auto": VehicleClass.AUTO_RICKSHAW,
    "autorickshaw": VehicleClass.AUTO_RICKSHAW,
    "auto rickshaw": VehicleClass.AUTO_RICKSHAW,
    "rickshaw": VehicleClass.AUTO_RICKSHAW,
    "3w": VehicleClass.AUTO_RICKSHAW,
    "three wheeler": VehicleClass.AUTO_RICKSHAW,

    # Other road users / vehicle families
    "van": VehicleClass.LIGHT_COMMERCIAL,
    "lcv": VehicleClass.LIGHT_COMMERCIAL,
    "ltv": VehicleClass.LIGHT_COMMERCIAL,
    "light commercial": VehicleClass.LIGHT_COMMERCIAL,
    "light commercial vehicle": VehicleClass.LIGHT_COMMERCIAL,
    "truck": VehicleClass.HEAVY_VEHICLE,
    "heavy truck": VehicleClass.HEAVY_VEHICLE,
    "hgv": VehicleClass.HEAVY_VEHICLE,
    "bus": VehicleClass.BUS,
    "bicycle": VehicleClass.BICYCLE,
    "cycle": VehicleClass.BICYCLE,
    "pedestrian": VehicleClass.PEDESTRIAN,
    "person": VehicleClass.PEDESTRIAN,
}


def map_vehicle_class(source_label: object) -> ClassMapping:
    raw = str(source_label)
    normalized = _normalize(source_label)
    canonical = _ALIAS_MAP.get(normalized, VehicleClass.OTHER)
    support = (
        BehaviorSupport.CALIBRATED
        if canonical in _CALIBRATED
        else BehaviorSupport.ASSUMED
    )
    return ClassMapping(
        source_label=raw,
        canonical=canonical,
        behavior_support=support,
    )
