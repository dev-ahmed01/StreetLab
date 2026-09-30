from __future__ import annotations

import pytest


def test_class_mapping_preserves_source_label_and_behavior_boundary():
    from streetlab_phase3.class_mapping import map_vehicle_class
    from streetlab_phase3.models import BehaviorSupport, VehicleClass

    moped = map_vehicle_class("moped")
    assert moped.canonical == VehicleClass.MOTORCYCLE
    assert moped.behavior_support == BehaviorSupport.CALIBRATED
    assert moped.source_label == "moped"

    auto = map_vehicle_class("auto-rickshaw")
    assert auto.canonical == VehicleClass.AUTO_RICKSHAW
    assert auto.behavior_support == BehaviorSupport.CALIBRATED

    lcv = map_vehicle_class("lcv")
    assert lcv.canonical == VehicleClass.LIGHT_COMMERCIAL
    assert lcv.behavior_support == BehaviorSupport.ASSUMED

    unknown = map_vehicle_class("mystery_vehicle")
    assert unknown.canonical == VehicleClass.OTHER
    assert unknown.behavior_support == BehaviorSupport.ASSUMED
    assert unknown.source_label == "mystery_vehicle"


def test_fluid_tracks_normalize_to_streetlab_schema():
    from streetlab_phase3.adapters.fluid import FluidAdapter
    from streetlab_phase3.models import EvidenceProvenance, VehicleClass

    rows = [
        {
            "frame": 10,
            "id": 7,
            "type": "moped",
            "confidence": 0.94,
            "cx_m": 12.5,
            "cy_m": 8.25,
            "time": 1.0,
            "speed": 6.0,
            "speed_smooth": 5.5,
            "ax": 0.2,
            "ay": 0.1,
            "course": 0.4,
            "entry_direction": "N",
            "exit_direction": "E",
            "overall_direction": "N-E",
        }
    ]

    points = FluidAdapter().normalize_tracks(rows)

    assert len(points) == 1
    p = points[0]
    assert p.source_provider == "fluid"
    assert p.source_track_id == "7"
    assert p.vehicle_class == VehicleClass.MOTORCYCLE
    assert p.source_class == "moped"
    assert p.time_s == 1.0
    assert p.x_m == 12.5
    assert p.y_m == 8.25
    assert p.speed_mps == 5.5
    assert p.acceleration_mps2 == pytest.approx((0.2**2 + 0.1**2) ** 0.5)
    assert p.movement == "N-E"
    assert p.provenance == EvidenceProvenance.OBSERVED_AUTO


def test_geotrax_tracks_normalize_units_and_classes():
    from streetlab_phase3.adapters.geotrax import GeoTraxAdapter
    from streetlab_phase3.models import VehicleClass

    rows = [
        {
            "Vehicle_ID": 22,
            "Timestamp": "2026-01-01 10:00:01.500",
            "Frame_Number": 45,
            "Local_X": 21.0,
            "Local_Y": 33.0,
            "Vehicle_Class": 3,
            "Vehicle_Speed": 36.0,
            "Vehicle_Acceleration": 1.2,
            "Road_Section": 5,
            "Lane_Number": 2,
            "Visibility": 1,
        }
    ]

    points = GeoTraxAdapter().normalize_tracks(rows)

    assert len(points) == 1
    p = points[0]
    assert p.source_provider == "geotrax"
    assert p.vehicle_class == VehicleClass.MOTORCYCLE
    assert p.source_class == "3"
    assert p.speed_mps == pytest.approx(10.0)
    assert p.acceleration_mps2 == 1.2
    assert p.road_section == "5"
    assert p.lane_id == "2"
    assert p.frame_number == 45


def test_provider_registry_autodetects_supported_track_formats():
    from streetlab_phase3.providers import ProviderRegistry

    registry = ProviderRegistry.default()

    fluid = registry.detect_track_provider(
        [{"id": 1, "type": "car", "cx_m": 1, "cy_m": 2, "time": 0, "speed": 3}]
    )
    assert fluid.name == "fluid"

    geotrax = registry.detect_track_provider(
        [{
            "Vehicle_ID": 1,
            "Vehicle_Class": 0,
            "Local_X": 1,
            "Local_Y": 2,
            "Vehicle_Speed": 20,
            "Frame_Number": 3,
        }]
    )
    assert geotrax.name == "geotrax"

    with pytest.raises(ValueError, match="Unsupported trajectory schema"):
        registry.detect_track_provider([{"foo": 1, "bar": 2}])


def test_observation_calibration_builds_site_targets_without_retraining_personas():
    from streetlab_phase3.calibration import SiteCalibrator
    from streetlab_phase3.models import BehaviorSupport

    rows = [
        {"id": 1, "type": "moped", "cx_m": 0, "cy_m": 0, "time": 0, "speed": 5, "overall_direction": "N-E"},
        {"id": 1, "type": "moped", "cx_m": 1, "cy_m": 0, "time": 1, "speed": 6, "overall_direction": "N-E"},
        {"id": 2, "type": "car", "cx_m": 0, "cy_m": 1, "time": 0, "speed": 8, "overall_direction": "W-E"},
        {"id": 2, "type": "car", "cx_m": 1, "cy_m": 1, "time": 1, "speed": 10, "overall_direction": "W-E"},
        {"id": 3, "type": "truck", "cx_m": 2, "cy_m": 1, "time": 0, "speed": 4, "overall_direction": "S-N"},
    ]

    package = SiteCalibrator().calibrate(track_rows=rows)

    assert package.source_provider == "fluid"
    assert package.summary.unique_tracks == 3
    assert package.summary.class_counts["MOTORCYCLE"] == 1
    assert package.summary.class_counts["CAR"] == 1
    assert package.summary.class_counts["HEAVY_VEHICLE"] == 1
    assert package.summary.movement_counts == {"N-E": 1, "S-N": 1, "W-E": 1}
    assert package.summary.mean_speed_mps == pytest.approx((5 + 6 + 8 + 10 + 4) / 5)

    assert package.behavior_support["MOTORCYCLE"] == BehaviorSupport.CALIBRATED.value
    assert package.behavior_support["CAR"] == BehaviorSupport.CALIBRATED.value
    assert package.behavior_support["HEAVY_VEHICLE"] == BehaviorSupport.ASSUMED.value

    assert package.persona_calibration_modified is False
    assert package.site_calibration_only is True


def test_phase3_exports_phase2_evidence_without_promoting_assumptions():
    from streetlab_phase3.calibration import SiteCalibrator

    rows = [
        {"id": 1, "type": "auto", "cx_m": 0, "cy_m": 0, "time": 0, "speed": 4, "overall_direction": "N-S"},
        {"id": 1, "type": "auto", "cx_m": 2, "cy_m": 0, "time": 1, "speed": 5, "overall_direction": "N-S"},
    ]

    package = SiteCalibrator().calibrate(track_rows=rows)
    evidence = {item.name: item.provenance.value for item in package.to_phase2_evidence()}

    assert evidence["trajectories"] == "OBSERVED_AUTO"
    assert evidence["speeds"] == "OBSERVED_AUTO"
    assert evidence["turn_movements"] == "OBSERVED_AUTO"
    assert "geometry" not in evidence
    assert "alternate_route" not in evidence


def test_generic_adapter_requires_explicit_mapping_instead_of_guessing_columns():
    from streetlab_phase3.adapters.generic import GenericTrajectoryAdapter, GenericTrajectoryMapping

    rows = [
        {"vehicle": "a", "category": "bike", "t": 0.0, "east": 10.0, "north": 20.0, "velocity": 7.5}
    ]

    with pytest.raises(ValueError, match="mapping"):
        GenericTrajectoryAdapter().normalize_tracks(rows)

    mapping = GenericTrajectoryMapping(
        track_id="vehicle",
        vehicle_class="category",
        time_s="t",
        x_m="east",
        y_m="north",
        speed_mps="velocity",
    )
    points = GenericTrajectoryAdapter(mapping=mapping).normalize_tracks(rows)

    assert points[0].source_track_id == "a"
    assert points[0].vehicle_class.value == "MOTORCYCLE"
    assert points[0].speed_mps == 7.5
