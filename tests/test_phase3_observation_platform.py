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


def test_fluid_signal_and_route_records_normalize():
    from streetlab_phase3.adapters.fluid import FluidSignalAdapter, FluidRouteAdapter

    signals = FluidSignalAdapter().normalize_signals([
        {
            "name": "FI",
            "direction": "N",
            "turn": "s",
            "state": "G",
            "begin_time": 10.0,
            "end_time": 25.0,
            "duration": 15.0,
            "cycle": 2,
        }
    ])
    assert signals[0].provider == "fluid"
    assert signals[0].direction == "N"
    assert signals[0].turn == "STRAIGHT"
    assert signals[0].state == "GREEN"
    assert signals[0].begin_time_s == 10.0
    assert signals[0].end_time_s == 25.0
    assert signals[0].cycle_id == "2"

    routes = FluidRouteAdapter().normalize_routes([
        {
            "id": 101,
            "in_time": 3.0,
            "out_time": 12.0,
            "type": "car",
            "entry_direction": "N",
            "exit_direction": "E",
            "overall_direction": "N-E",
            "turn": "Right",
            "in_state": "G",
            "out_state": "G",
        }
    ])
    assert routes[0].provider == "fluid"
    assert routes[0].source_track_id == "101"
    assert routes[0].movement == "N-E"
    assert routes[0].turn == "RIGHT"
    assert routes[0].travel_time_s == 9.0


def test_site_calibration_can_include_signal_and_route_targets():
    from streetlab_phase3.calibration import SiteCalibrator

    tracks = [
        {"id": 1, "type": "car", "cx_m": 0, "cy_m": 0, "time": 0, "speed": 5, "overall_direction": "N-E"},
        {"id": 1, "type": "car", "cx_m": 1, "cy_m": 1, "time": 1, "speed": 6, "overall_direction": "N-E"},
    ]
    signals = [
        {"name": "FI", "direction": "N", "turn": "s", "state": "G", "begin_time": 0, "end_time": 20, "duration": 20, "cycle": 1}
    ]
    routes = [
        {"id": 1, "in_time": 0, "out_time": 8, "type": "car", "entry_direction": "N", "exit_direction": "E", "overall_direction": "N-E", "turn": "Right", "in_state": "G", "out_state": "G"}
    ]

    package = SiteCalibrator().calibrate(
        track_rows=tracks,
        signal_rows=signals,
        route_rows=routes,
    )

    assert len(package.signals) == 1
    assert len(package.routes) == 1
    assert package.route_summary.mean_travel_time_s == 8.0
    evidence = {item.name: item.provenance.value for item in package.to_phase2_evidence()}
    assert evidence["signals"] == "OBSERVED_AUTO"
    assert evidence["routes"] == "OBSERVED_AUTO"


def test_quality_report_flags_impossible_values_without_retraining_personas():
    from streetlab_phase3.calibration import SiteCalibrator

    rows = [
        {"id": 1, "type": "car", "cx_m": 0, "cy_m": 0, "time": 0, "speed": -4, "overall_direction": "N-E"},
        {"id": 2, "type": "moped", "cx_m": 1, "cy_m": 1, "time": 0, "speed": 90, "overall_direction": "S-N"},
        {"id": 3, "type": "mystery", "cx_m": 1, "cy_m": 1, "time": 0, "speed": 5, "overall_direction": "W-E"},
    ]

    package = SiteCalibrator().calibrate(track_rows=rows)

    assert package.quality.total_points == 3
    assert package.quality.negative_speed_points == 1
    assert package.quality.extreme_speed_points == 1
    assert package.quality.unknown_class_tracks == 1
    assert package.quality.persona_calibration_modified is False


def test_geotrax_video_provider_builds_external_extraction_plan_without_vendoring():
    from streetlab_phase3.video.geotrax_provider import GeoTraxVideoProvider

    provider = GeoTraxVideoProvider(executable="geotrax")

    pixel = provider.plan(video="junction.mp4")
    assert pixel.command == (
        "geotrax", "batch", "junction.mp4", "--no-geo"
    )
    assert pixel.georeferenced is False
    assert pixel.calibration_ready is False

    full = provider.plan(
        video="junction.mp4",
        orthophotos="orthos",
        segmentations="segments",
        master_frames="masters",
    )
    assert full.command == (
        "geotrax", "batch", "junction.mp4",
        "-orf", "orthos",
        "-osf", "segments",
        "-mf", "masters",
    )
    assert full.georeferenced is True
    assert full.calibration_ready is True
    assert full.license_boundary == "EXTERNAL_PROVIDER"


def test_generic_provider_is_the_fallback_for_new_raw_trajectory_formats():
    from streetlab_phase3.adapters.generic import GenericTrajectoryAdapter, GenericTrajectoryMapping
    from streetlab_phase3.calibration import SiteCalibrator
    from streetlab_phase3.providers import ProviderRegistry

    mapping = GenericTrajectoryMapping(
        track_id="vid",
        vehicle_class="kind",
        time_s="seconds",
        x_m="x_world",
        y_m="y_world",
        speed_mps="v",
        movement="movement",
    )
    registry = ProviderRegistry.default().with_provider(
        GenericTrajectoryAdapter(mapping=mapping)
    )
    calibrator = SiteCalibrator(registry=registry)

    package = calibrator.calibrate(track_rows=[
        {"vid": "a1", "kind": "auto-rickshaw", "seconds": 0, "x_world": 1, "y_world": 2, "v": 4, "movement": "E-N"}
    ])

    assert package.source_provider == "generic"
    assert package.summary.class_counts["AUTO_RICKSHAW"] == 1
    assert package.behavior_support["AUTO_RICKSHAW"] == "CALIBRATED"


def test_file_pipeline_auto_detects_fluid_csv(tmp_path):
    from streetlab_phase3.ingestion import CalibrationPipeline

    path = tmp_path / "tracks.csv"
    path.write_text(
        "id,type,cx_m,cy_m,time,speed,overall_direction\n"
        "1,moped,1.0,2.0,0.0,5.0,N-E\n"
        "1,moped,2.0,3.0,1.0,6.0,N-E\n",
        encoding="utf-8",
    )

    package = CalibrationPipeline().calibrate_files(track_file=path)

    assert package.source_provider == "fluid"
    assert package.summary.unique_tracks == 1
    assert package.summary.class_counts == {"MOTORCYCLE": 1}


def test_file_pipeline_accepts_unknown_csv_with_explicit_mapping(tmp_path):
    from streetlab_phase3.adapters.generic import GenericTrajectoryMapping
    from streetlab_phase3.ingestion import CalibrationPipeline

    path = tmp_path / "india_tracks.csv"
    path.write_text(
        "veh,klass,t,x,y,speed,move\n"
        "a1,auto-rickshaw,0,1,2,4,E-N\n",
        encoding="utf-8",
    )
    mapping = GenericTrajectoryMapping(
        track_id="veh",
        vehicle_class="klass",
        time_s="t",
        x_m="x",
        y_m="y",
        speed_mps="speed",
        movement="move",
    )

    package = CalibrationPipeline().calibrate_files(
        track_file=path,
        generic_mapping=mapping,
    )

    assert package.source_provider == "generic"
    assert package.summary.class_counts == {"AUTO_RICKSHAW": 1}


def test_file_pipeline_rejects_unknown_csv_without_mapping(tmp_path):
    from streetlab_phase3.ingestion import CalibrationPipeline

    path = tmp_path / "unknown.csv"
    path.write_text("veh,x,y\na,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported trajectory schema"):
        CalibrationPipeline().calibrate_files(track_file=path)


def test_package_serialization_is_json_ready():
    import json

    from streetlab_phase3.calibration import SiteCalibrator
    from streetlab_phase3.serialization import package_to_dict

    package = SiteCalibrator().calibrate(track_rows=[
        {"id": 1, "type": "car", "cx_m": 0, "cy_m": 0, "time": 0, "speed": 5, "overall_direction": "N-E"}
    ])

    payload = package_to_dict(package)
    encoded = json.dumps(payload)

    assert '"source_provider": "fluid"' in encoded
    assert payload["tracks"][0]["vehicle_class"] == "CAR"
    assert payload["tracks"][0]["provenance"] == "OBSERVED_AUTO"
    assert payload["quality"]["persona_calibration_modified"] is False


def test_simjam_bundle_normalizes_fixed_camera_tracks():
    from streetlab_phase3.adapters.simjam import SimJamBundleAdapter

    track_rows = [
        {"frame": 0, "time_s": 0.0, "vehicle_id": 11, "x_m": 1.0, "y_m": 2.0, "img_x": 100, "img_y": 200},
        {"frame": 1, "time_s": 0.1, "vehicle_id": 11, "x_m": 1.5, "y_m": 2.0, "img_x": 102, "img_y": 200},
    ]
    summary_rows = [
        {"vehicle_id": 11, "label": "car", "avg_speed_kmh": 18.0, "start_frame": 0, "end_frame": 1, "start_time_s": 0.0, "end_time_s": 0.1}
    ]

    points = SimJamBundleAdapter().normalize_bundle(track_rows, summary_rows)

    assert len(points) == 2
    assert points[0].source_provider == "simjam"
    assert points[0].vehicle_class.value == "CAR"
    assert points[0].speed_mps == pytest.approx(5.0)
    assert points[0].metadata["speed_kind"] == "track_average"
    assert points[0].metadata["img_x"] == 100.0


def test_file_pipeline_calibrates_simjam_bundle(tmp_path):
    from streetlab_phase3.ingestion import CalibrationPipeline

    tracks = tmp_path / "vehicle_tracks_xy.csv"
    tracks.write_text(
        "frame,time_s,vehicle_id,x_m,y_m,img_x,img_y\n"
        "0,0.0,11,1.0,2.0,100,200\n"
        "1,0.1,11,1.5,2.0,102,200\n",
        encoding="utf-8",
    )
    summary = tmp_path / "vehicle_summary.csv"
    summary.write_text(
        "vehicle_id,label,avg_speed_kmh,start_frame,end_frame,start_time_s,end_time_s\n"
        "11,motorcycle,18.0,0,1,0.0,0.1\n",
        encoding="utf-8",
    )

    package = CalibrationPipeline().calibrate_simjam_files(
        track_file=tracks,
        summary_file=summary,
    )

    assert package.source_provider == "simjam"
    assert package.summary.unique_tracks == 1
    assert package.summary.class_counts == {"MOTORCYCLE": 1}
    assert package.behavior_support["MOTORCYCLE"] == "CALIBRATED"


def test_generic_signal_and_route_mappings_support_new_study_schemas():
    from streetlab_phase3.adapters.generic import (
        GenericRouteAdapter,
        GenericRouteMapping,
        GenericSignalAdapter,
        GenericSignalMapping,
    )

    signal = GenericSignalAdapter(
        GenericSignalMapping(
            direction="approach",
            turn="movement",
            state="colour",
            begin_time_s="from_s",
            end_time_s="to_s",
            cycle_id="cycle_no",
        )
    ).normalize_signals([
        {"approach": "N", "movement": "straight", "colour": "green", "from_s": 0, "to_s": 20, "cycle_no": 1}
    ])[0]

    assert signal.direction == "N"
    assert signal.turn == "STRAIGHT"
    assert signal.state == "GREEN"
    assert signal.duration_s == 20.0

    route = GenericRouteAdapter(
        GenericRouteMapping(
            track_id="veh",
            vehicle_class="klass",
            in_time_s="tin",
            out_time_s="tout",
            movement="od",
            turn="turn",
        )
    ).normalize_routes([
        {"veh": "r1", "klass": "auto", "tin": 2, "tout": 9, "od": "W-N", "turn": "left"}
    ])[0]

    assert route.source_track_id == "r1"
    assert route.vehicle_class.value == "AUTO_RICKSHAW"
    assert route.travel_time_s == 7.0
    assert route.movement == "W-N"
    assert route.turn == "LEFT"


def test_file_pipeline_uses_generic_mappings_for_tracks_signals_and_routes(tmp_path):
    from streetlab_phase3.adapters.generic import (
        GenericRouteMapping,
        GenericSignalMapping,
        GenericTrajectoryMapping,
    )
    from streetlab_phase3.ingestion import CalibrationPipeline

    tracks = tmp_path / "tracks.csv"
    tracks.write_text(
        "veh,klass,t,x,y,v,move\n"
        "1,car,0,0,0,5,N-E\n",
        encoding="utf-8",
    )
    signals = tmp_path / "signals.csv"
    signals.write_text(
        "approach,movement,colour,from_s,to_s,cycle\n"
        "N,straight,G,0,20,1\n",
        encoding="utf-8",
    )
    routes = tmp_path / "routes.csv"
    routes.write_text(
        "veh,klass,tin,tout,od,turn\n"
        "1,car,0,8,N-E,right\n",
        encoding="utf-8",
    )

    package = CalibrationPipeline().calibrate_files(
        track_file=tracks,
        signal_file=signals,
        route_file=routes,
        generic_mapping=GenericTrajectoryMapping(
            track_id="veh",
            vehicle_class="klass",
            time_s="t",
            x_m="x",
            y_m="y",
            speed_mps="v",
            movement="move",
        ),
        generic_signal_mapping=GenericSignalMapping(
            direction="approach",
            turn="movement",
            state="colour",
            begin_time_s="from_s",
            end_time_s="to_s",
            cycle_id="cycle",
        ),
        generic_route_mapping=GenericRouteMapping(
            track_id="veh",
            vehicle_class="klass",
            in_time_s="tin",
            out_time_s="tout",
            movement="od",
            turn="turn",
        ),
    )

    assert len(package.signals) == 1
    assert len(package.routes) == 1
    assert package.route_summary.mean_travel_time_s == 8.0


def test_generic_mapping_converts_speed_units_and_preserves_declared_provenance():
    from streetlab_phase3.adapters.generic import (
        GenericTrajectoryAdapter,
        GenericTrajectoryMapping,
    )

    adapter = GenericTrajectoryAdapter(
        GenericTrajectoryMapping(
            track_id="id",
            vehicle_class="class",
            time_s="t",
            x_m="x",
            y_m="y",
            speed_mps="speed",
            speed_unit="km/h",
            provenance="OBSERVED_MANUAL",
        )
    )
    point = adapter.normalize_tracks([
        {"id": "v1", "class": "car", "t": 0, "x": 1, "y": 2, "speed": 36}
    ])[0]

    assert point.speed_mps == pytest.approx(10.0)
    assert point.provenance.value == "OBSERVED_MANUAL"


def test_simjam_track_average_speed_is_not_weighted_by_track_length(tmp_path):
    from streetlab_phase3.ingestion import CalibrationPipeline

    tracks = tmp_path / "tracks.csv"
    tracks.write_text(
        "frame,time_s,vehicle_id,x_m,y_m,img_x,img_y\n"
        "0,0.0,1,0,0,0,0\n"
        "1,0.1,1,1,0,1,0\n"
        "2,0.2,1,2,0,2,0\n"
        "0,0.0,2,0,1,0,1\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "vehicle_id,label,avg_speed_kmh,start_frame,end_frame,start_time_s,end_time_s\n"
        "1,car,18,0,2,0,0.2\n"
        "2,car,36,0,0,0,0\n",
        encoding="utf-8",
    )

    package = CalibrationPipeline().calibrate_simjam_files(
        track_file=tracks,
        summary_file=summary,
    )

    assert package.summary.mean_speed_mps == pytest.approx(7.5)


def test_fluid_partial_direction_is_not_promoted_to_turn_movement():
    from streetlab_phase3.adapters.fluid import FluidAdapter

    points = FluidAdapter().normalize_tracks([
        {
            "frame": "4074",
            "id": "938",
            "type": "car",
            "confidence": "0.89",
            "cx_m": "8.8",
            "cy_m": "-13.35",
            "time": "407.4",
            "speed": "14.89",
            "speed_smooth": "14.83",
            "ax": "-0.05",
            "ay": "-1.10",
            "entry_direction": "S",
            "exit_direction": "Unknown",
            "overall_direction": "S-Unknown",
        }
    ])

    assert points[0].entry_direction == "S"
    assert points[0].exit_direction is None
    assert points[0].movement is None
    assert points[0].metadata["raw_movement"] == "S-Unknown"


def test_fluid_blank_numeric_fields_do_not_crash_real_schema():
    from streetlab_phase3.adapters.fluid import FluidAdapter

    points = FluidAdapter().normalize_tracks([
        {
            "frame": "1",
            "id": "31",
            "type": "moped",
            "confidence": "",
            "cx_m": "-1.54",
            "cy_m": "9.58",
            "time": "0.1",
            "speed": "",
            "speed_smooth": "",
            "ax": "",
            "ay": "",
            "course": "",
            "entry_direction": "N",
            "exit_direction": "E",
            "overall_direction": "N-E",
        }
    ])

    p = points[0]
    assert p.speed_mps is None
    assert p.acceleration_mps2 is None
    assert p.heading_rad is None
    assert p.confidence is None


def test_fluid_incomplete_route_rows_are_skipped_and_reported():
    from streetlab_phase3.calibration import SiteCalibrator

    tracks = [
        {
            "id": "1",
            "type": "car",
            "cx_m": "0",
            "cy_m": "0",
            "time": "0",
            "speed": "5",
            "entry_direction": "N",
            "exit_direction": "S",
            "overall_direction": "N-S",
        }
    ]
    routes = [
        {
            "id": "1",
            "in_time": "1.0",
            "out_time": "7.0",
            "type": "car",
            "entry_direction": "N",
            "exit_direction": "S",
            "overall_direction": "N-S",
            "turn": "s",
            "in_state": "G",
            "out_state": "G",
        },
        {
            "id": "923",
            "in_time": "",
            "out_time": "",
            "type": "car",
            "entry_direction": "N",
            "exit_direction": "Unknown",
            "overall_direction": "N-Unknown",
            "turn": "",
            "in_state": "",
            "out_state": "",
        },
    ]

    package = SiteCalibrator().calibrate(track_rows=tracks, route_rows=routes)

    assert len(package.routes) == 1
    assert package.routes[0].source_track_id == "1"
    assert package.quality.route_rows_total == 2
    assert package.quality.route_rows_complete == 1
    assert package.quality.route_rows_incomplete == 1


def test_fluid_telemetry_normalizes_camera_state():
    from streetlab_phase3.adapters.fluid_telemetry import FluidTelemetryAdapter

    rows = [
        {
            "time(millisecond)": "265200",
            "datetime": "2025-01-20 16:32:34",
            "height_above_takeoff(feet)": "426.837284",
            "speed(mph)": "0.0",
            "compass_heading(degrees)": "0.1",
            "pitch(degrees)": "-4.9",
            "roll(degrees)": "3.3",
            "isVideo": "1",
            "gimbal_heading(degrees)": "3.5",
            "gimbal_pitch(degrees)": "-89.9",
            "gimbal_roll(degrees)": "0",
            "battery_percent": "85",
            "flycState": "Tripod",
        }
    ]

    sample = FluidTelemetryAdapter().normalize(rows)[0]

    assert sample.time_ms == 265200
    assert sample.recording is True
    assert sample.height_m == pytest.approx(130.10, abs=0.02)
    assert sample.gimbal_pitch_deg == -89.9
    assert sample.flight_state == "Tripod"


def test_fluid_benchmark_report_combines_observation_and_telemetry():
    from streetlab_phase3.benchmark import FluidBenchmark

    tracks = [
        {"id": "1", "type": "moped", "cx_m": "0", "cy_m": "0", "time": "0.1", "speed": "5", "entry_direction": "N", "exit_direction": "E", "overall_direction": "N-E"},
        {"id": "1", "type": "moped", "cx_m": "1", "cy_m": "1", "time": "0.2", "speed": "6", "entry_direction": "N", "exit_direction": "E", "overall_direction": "N-E"},
        {"id": "2", "type": "car", "cx_m": "0", "cy_m": "1", "time": "0.2", "speed": "7", "entry_direction": "W", "exit_direction": "Unknown", "overall_direction": "W-Unknown"},
    ]
    signals = [
        {"name": "FIDRT", "direction": "N", "turn": "s", "begin_time": "0", "end_time": "20", "duration": "20", "state": "G", "cycle": "1"}
    ]
    routes = [
        {"id": "1", "in_time": "0.1", "out_time": "3.0", "type": "moped", "entry_direction": "N", "exit_direction": "E", "overall_direction": "N-E", "turn": "r", "in_state": "G", "out_state": "G"},
        {"id": "2", "in_time": "", "out_time": "", "type": "car", "entry_direction": "W", "exit_direction": "Unknown", "overall_direction": "W-Unknown"},
    ]
    telemetry = [
        {"time(millisecond)": "1000", "datetime": "2025-01-01 10:00:00", "height_above_takeoff(feet)": "400", "speed(mph)": "0", "compass_heading(degrees)": "0", "pitch(degrees)": "0", "roll(degrees)": "0", "isVideo": "1", "gimbal_heading(degrees)": "0", "gimbal_pitch(degrees)": "-90", "gimbal_roll(degrees)": "0", "battery_percent": "90", "flycState": "Tripod"},
        {"time(millisecond)": "1100", "datetime": "2025-01-01 10:00:00", "height_above_takeoff(feet)": "401", "speed(mph)": "0", "compass_heading(degrees)": "0", "pitch(degrees)": "0", "roll(degrees)": "0", "isVideo": "1", "gimbal_heading(degrees)": "0", "gimbal_pitch(degrees)": "-90", "gimbal_roll(degrees)": "0", "battery_percent": "89", "flycState": "Tripod"},
    ]

    report = FluidBenchmark().build(
        study_id="demo",
        track_rows=tracks,
        signal_rows=signals,
        route_rows=routes,
        telemetry_rows=telemetry,
    )

    assert report.study_id == "demo"
    assert report.unique_tracks == 2
    assert report.complete_routes == 1
    assert report.incomplete_routes == 1
    assert report.known_movement_tracks == 1
    assert report.signal_coverage_end_s == 20.0
    assert report.telemetry.recording_fraction == 1.0
    assert report.telemetry.mean_height_m == pytest.approx(122.07, abs=0.05)
    assert report.persona_calibration_modified is False


def test_fluid_benchmark_serializes_json_ready():
    import json
    from streetlab_phase3.benchmark import FluidBenchmark
    from streetlab_phase3.serialization import package_to_dict

    report = FluidBenchmark().build(
        study_id="demo",
        track_rows=[
            {"id": "1", "type": "car", "cx_m": "0", "cy_m": "0", "time": "0.1", "speed": "5", "entry_direction": "N", "exit_direction": "S", "overall_direction": "N-S"}
        ],
    )

    payload = package_to_dict(report)
    assert json.loads(json.dumps(payload))["study_id"] == "demo"
