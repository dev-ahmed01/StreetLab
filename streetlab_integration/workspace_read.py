"""M6 single-project operational read model.

The overview never reads the global M1 latest pointer, never promotes
tracker IDs to physical counts, and never treats simulated differences as
physical observations. Every published downstream state is verified through
its own immutable source chain before it reaches the browser.
"""
from __future__ import annotations

from streetlab_integration.video_jobs import VideoStore
from streetlab_integration.worker import job_report
from streetlab_integration.primary_runner import verified_run
from streetlab_integration.reconstruction import latest_reconstruction
from streetlab_integration.site_baseline import latest_baseline
from streetlab_integration.scenario_experiments import latest_scenario


def project_workspace(store: VideoStore, project_id: str) -> dict:
    project=store.project(project_id)
    with store.connection() as conn:
        row=conn.execute(
            "SELECT * FROM sources WHERE project_id=?", (project_id,)
        ).fetchone()
    source=store._dict(row)
    jobs=store.jobs(project_id)
    newest=jobs[0] if jobs else None
    successful=next((j for j in jobs if j["status"]=="SUCCEEDED"),None)

    observed=None
    overlay=None
    if successful:
        report=job_report(store,successful["id"])
        actual=store.source(successful["source_id"])
        manifest=verified_run(store.root/"runs"/successful["id"],actual["sha256"])
        observed={
            "job_id":successful["id"],
            "observed_points":report["observed_points"],
            "observed_frames":report["observed_frames"],
            "tracking_identity_count":report["tracking_identity_count"],
            "tracks_by_class":report["tracks_by_class"],
            "source_tracking_sha256":report["source_tracking_sha256"],
            "physical_vehicle_count_confirmed":False,
            "coordinate_system":"SOURCE_IMAGE_PIXELS",
        }
        names=manifest.get("overlay_files",[])
        if names:
            overlay="/api/jobs/"+successful["id"]+"/overlays/"+names[0]

    if not source:
        obs_status="NEEDS_SOURCE"
        obs_note="Upload an actual traffic video before analysis"
    elif not newest:
        obs_status="NEEDS_ANALYSIS"
        obs_note="Queue the frozen primary tracking run; a local worker is required"
    elif newest["status"] in ("QUEUED","RUNNING"):
        obs_status=newest["status"]
        obs_note="Local analysis progress is saved; the separate worker processes frames"
    elif observed:
        obs_status="OBSERVATION_VERIFIED"
        obs_note="Source-pixel tracking verified; tracker IDs are not distinct vehicle counts"
    else:
        obs_status=newest["status"]
        obs_note="Analysis did not produce a verified observation; inspect and retry"
    progress=(
        round(100*newest["completed_frames"]/newest["total_frames"],1)
        if newest and newest["total_frames"] else 0.0
    )
    observation={
        "status":obs_status,"note":obs_note,
        "source":({
            "id":source["id"],"original_filename":source["original_filename"],
            "sha256":source["sha256"],"size_bytes":source["size_bytes"],
            "metadata":source["metadata"]}
            if source else None),
        "latest_job":({
            "id":newest["id"],"status":newest["status"],
            "completed_frames":newest["completed_frames"],
            "total_frames":newest["total_frames"],
            "progress_percent":progress,
            "error":newest["error"]}
            if newest else None),
        "last_verified_job":observed,"overlay_url":overlay,
    }

    spatial=latest_reconstruction(store,project_id)
    if spatial.get("status")=="NOT_CONFIGURED":
        geo_status="NEEDS_GEOMETRY" if observed else "WAITING_FOR_OBSERVATION"
        geometry={"status":geo_status,
                  "revision":None,"surveyed_local_meters":False,
                  "movement_count":None,
                  "missing_evidence":spatial["quality"]["missing_evidence"],
                  "anchors":0,"checks":0,"site":None,
                  "note":"Add actual source-frame geometry; survey evidence is never auto-invented"}
    else:
        q=spatial["quality"]
        model=spatial["model"]
        current=bool(observed and observed["source_tracking_sha256"]==model["source_track_sha256"])
        ready=(q["status"]=="GUIDED_GEOMETRY_REVIEWED" and current)
        geometry={
            "status":("GEOMETRY_REVIEWED" if ready else
                      "STALE_OBSERVATION" if not current else "NEEDS_DATA"),
            "revision":spatial["revision"],"surveyed_local_meters":bool(q["metric_transform_checked"]),
            "movement_count":len(model["movements"]),
            "missing_evidence":q["missing_evidence"]+(
                ["Reconstruction belongs to a different verified observation job"]
                if not current else []),
            "anchors":len(model["anchors"]),"checks":len(model["checks"]),
            "site":{"coordinate_system":model["coordinate_system"],
                    "zones":[{"id":z["id"],"role":z["role"],
                              "polygon_pixels":z["polygon_pixels"],
                              "lane_count":z["lane_count"]}
                             for z in model["zones"]],
                    "width":model["width"],"height":model["height"]},
            "note":"This is a surveyed local plane, not GPS or verified physical traffic demand",
        }
    baseline=latest_baseline(store,project_id)
    if baseline.get("status")=="NOT_CONFIGURED":
        base_status="NEEDS_BASELINE" if geometry["status"]=="GEOMETRY_REVIEWED" else "WAITING_FOR_GEOMETRY"
        site_baseline={"status":base_status,"revision":None,
                       "execution_status":"NOT_EXECUTED","scenario_eligible":False,
                       "manual_distinct_vehicle_count":None,"movement_fidelity":[],
                       "missing_evidence":baseline["quality"]["missing_evidence"],
                       "road_graph":None,
                       "note":"Provide an independent census, surveyed lane links and holdout"}
    else:
        q=baseline["quality"]; run=baseline["runtime"]
        current=bool(geometry["revision"] and baseline["model"]["spatial_revision"]==geometry["revision"]
                     and geometry["status"]=="GEOMETRY_REVIEWED")
        base_status=("STALE_GEOMETRY" if not current else
                     "BASELINE_QA_PASSED" if run["status"]=="BASELINE_FIDELITY_CHECKED" else
                     "BASELINE_NEEDS_REVIEW" if run["status"]=="BASELINE_NEEDS_REVIEW" else
                     "BASELINE_NOT_EXECUTED")
        site_baseline={
            "status":base_status,"revision":baseline["revision"],
            "execution_status":run["status"],
            "scenario_eligible":bool(current and run.get("real_site_sumo_allowed") is True),
            "manual_distinct_vehicle_count":q["manual_distinct_vehicle_count"],
            "movement_fidelity":run.get("per_movement",[]),
            "missing_evidence":([] if current else ["Baseline references a previous M3 site revision"]),
            "road_graph":{
                "coordinate_system":baseline["model"]["coordinate_system"],
                "center_world_m":baseline["model"]["center_world_m"],
                "arms":[{"zone_id":a["zone_id"],"role":a["role"],
                         "outer_world_m":a["outer_world_m"],
                         "lane_count":a["lane_count"]}
                        for a in baseline["model"]["arms"]],
            },
            "note":"QA thresholds are software heuristics; field validity is not certified",
        }
    experiment=latest_scenario(store,project_id)
    if experiment.get("status")=="NOT_CONFIGURED":
        exp_status="NEEDS_SCENARIO" if site_baseline["scenario_eligible"] else "WAITING_FOR_BASELINE"
        scenarios={
            "status":exp_status,"revision":None,"execution_status":"NOT_EXECUTED",
            "intervention":None,"comparison":None,
            "missing_evidence":experiment["quality"]["missing_evidence"],
            "note":"A reviewed intervention requires an eligible, actually run M4 baseline"}
    else:
        run=experiment["runtime"]; model=experiment["proposal"]
        current=bool(site_baseline["scenario_eligible"] and model["baseline_revision"]==site_baseline["revision"])
        status=("STALE_BASELINE" if not current else
                "SIMULATED_COMPARISON_ELIGIBLE" if run["status"]=="SIMULATED_COMPARISON_ELIGIBLE" else
                "INDETERMINATE" if run["status"]=="INDETERMINATE_INCOMPLETE_TRIPS" else
                "PROPOSAL_NOT_EXECUTED")
        scenarios={
            "status":status,"revision":experiment["revision"],
            "execution_status":run["status"],
            "intervention":model["intervention"],
            "comparison":({
                "paired_runs":run["paired_runs"],
                "comparable_runs":run["comparable_runs"],
                "mean_paired_simulation_difference_s":run["mean_paired_simulation_difference_s"],
                "min_paired_simulation_difference_s":run["min_paired_simulation_difference_s"],
                "max_paired_simulation_difference_s":run["max_paired_simulation_difference_s"],
                "simulated_improvement_in_every_condition":run["simulated_improvement_in_every_condition"],
                "all_conditions_comparable":run["all_conditions_comparable"],
                "real_world_outcome_verified":False,
                "records":[{"seed":x["seed"],"demand_multiplier":x["demand_multiplier"],
                            "paired_difference_s":x["paired_difference_s"],
                            "comparison_eligible":x["comparison_eligible"]}
                           for x in run["records"]]}
                if run["status"]!="NOT_EXECUTED" and current else None),
            "missing_evidence":([] if current else ["Scenario references a different current baseline revision"]),
            "note":"Paired SUMO sensitivity is not an observed or causal real-road outcome",
        }

    if obs_status in ("NEEDS_SOURCE","NEEDS_ANALYSIS","FAILED","CANCELLED"):
        stage,why,url="observation",obs_note,"/?advanced=1#m2Panel"
    elif obs_status in ("QUEUED","RUNNING"):
        stage,why,url="observation",obs_note,"/?advanced=1#m2Panel"
    elif geometry["status"]!="GEOMETRY_REVIEWED":
        stage,why,url="geometry","Review measured source-frame geometry and independent metric check","/calibration?project_id="+project_id
    elif not site_baseline["scenario_eligible"]:
        stage,why,url="baseline","Supply field census/holdout or run and review baseline in SUMO","/baseline?project_id="+project_id
    elif scenarios["status"]!="SIMULATED_COMPARISON_ELIGIBLE":
        stage,why,url="scenarios","Create, run or review bounded, paired SUMO experiments","/scenarios?project_id="+project_id
    else:
        stage,why,url="review","Inspect the simulated comparison and its scientific limitations","/scenarios?project_id="+project_id

    return {
        "schema_version":1,
        "project":{"id":project["id"],"name":project["name"]},
        "stages":{"observation":observation,"geometry":geometry,
                  "baseline":site_baseline,"scenarios":scenarios},
        "next_action":{"stage":stage,"reason":why,"url":url},
        "research_model":"FROZEN_W04_PRIMARY_HARD_NMS_IOS_030",
        "synthetic_phase2_separate":True,
        "physical_vehicle_count_inferred_from_tracker_ids":False,
        "real_world_impact_verified":False,
    }
