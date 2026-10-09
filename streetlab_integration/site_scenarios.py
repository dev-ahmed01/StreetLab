"""M5 reviewed, bounded counterfactual experiment contracts and comparisons.

Experiments are simulation-only hypotheses, NEVER observed causal effects.
Paired runs use identical input demand and RNG seeds in each condition.
"""
from __future__ import annotations

from copy import deepcopy
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

from streetlab_integration.site_inputs import SiteInputError, number, text, ident
from streetlab_integration.site_sumo import render_sumo

SEEDS=(42,43,44)
DEFAULT_LOADS=(0.9,1.0,1.1)
MAX_TRIPS_PER_FLOW=20000
MAX_TOTAL_TRIPS=50000


class ScenarioError(ValueError):
    pass


def validate_scenario(raw: dict, baseline: dict) -> dict:
    if not isinstance(raw,dict):
        raise ScenarioError("A scenario object is required")
    runtime=baseline["runtime"]
    if runtime.get("status")!="BASELINE_FIDELITY_CHECKED" or runtime.get("real_site_sumo_allowed") is not True:
        raise ScenarioError("Cannot run a real-site scenario without a successful M4 independent baseline")
    intervention=raw.get("intervention")
    if not isinstance(intervention,dict):
        raise ScenarioError("Explicit intervention is required")
    kind=intervention.get("kind")
    model=baseline["model"]
    original_control=model["control"]
    if kind=="APPROACH_SPEED_LIMIT":
        zone=ident(intervention.get("zone_id"),"affected road arm")
        by_zone={a["zone_id"]:a for a in model["arms"]}
        if zone not in by_zone:
            raise ScenarioError("The changed arm must exist in the audited M4 baseline")
        if by_zone[zone]["role"]!="APPROACH":
            raise ScenarioError("Only a surveyed APPROACH arm may receive an approach speed-limit intervention")
        old=by_zone[zone]["speed_mps"]
        speed=number(intervention.get("speed_mps"),"proposed speed limit",1,45)
        if not (speed<old and speed>=old*.5):
            raise ScenarioError("Speed-limit study only allows reductions of up to 50% below the reviewed baseline")
        applied={"kind":kind,"zone_id":zone,"speed_mps":speed}
    elif kind=="FIXED_SIGNAL_PLAN":
        if original_control["kind"]!="FIXED_TIME_SIGNAL":
            raise ScenarioError("A fixed-signal intervention requires an M4 reviewed fixed-time signal")
        phases=original_control["phases"]
        values=intervention.get("phase_durations_s")
        if not isinstance(values,list) or len(values)!=len(phases):
            raise ScenarioError("Specify one reviewed duration for every existing signal phase")
        durations=[number(x,"signal phase seconds",1,300) for x in values]
        if (sum(durations)>1200 or
            all(abs(p["duration_s"]-d)<1e-8 for p,d in zip(phases,durations))):
            raise ScenarioError("Phase timings must change while maintaining the same lanes and signal state order")
        if any(d < p["duration_s"]*.5 or d > p["duration_s"]*1.5
               for p,d in zip(phases,durations)):
            raise ScenarioError("Sensitivity bounds limit each phase change to ±50%")
        applied={"kind":kind,"phase_durations_s":durations}
    else:
        raise ScenarioError("Only APPROACH_SPEED_LIMIT and FIXED_SIGNAL_PLAN are supported; closures need verified detours")
    assumptions=raw.get("assumptions")
    if not isinstance(assumptions,dict):
        raise ScenarioError("Document operational scenario assumptions and references")
    reason=text(assumptions.get("decision_question"),"decision question")
    feasibility=text(assumptions.get("feasibility_evidence_ref"),"feasibility/implementation assumption")
    reviewer=text(assumptions.get("reviewer"),"scenario reviewer")
    if assumptions.get("provenance")!="HYPOTHETICAL_REVIEWED":
        raise ScenarioError("Intervention must be explicitly labeled HYPOTHETICAL_REVIEWED")
    loads=raw.get("demand_multipliers",list(DEFAULT_LOADS))
    if (not isinstance(loads,list) or len(loads)!=3 or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in loads)):
        raise ScenarioError("Exactly three finite demand multipliers are required")
    loads=sorted(float(x) for x in loads)
    if loads[0]<.8 or loads[2]>1.2 or not loads[0]<1.0<loads[2] or abs(loads[1]-1)>1e-9:
        raise ScenarioError("Sensitivity grid needs a low [0.8,1), central 1.0 and high (1,1.2] demand")
    max_count=sum(int(math.floor(d["count"]*loads[2]+.5)) for d in model["demand"])
    if max_count>MAX_TOTAL_TRIPS:
        raise ScenarioError("Demand exceeds bounded simulation capacity of 50,000 trips per run")
    if any(int(math.floor(d["count"]*loads[2]+.5))>MAX_TRIPS_PER_FLOW for d in model["demand"]):
        raise ScenarioError("One scaled movement-class flow exceeds 20,000 vehicles")
    return {
        "schema_version":1,"baseline_revision":baseline["revision"],
        "source_video_sha256":model["source_video_sha256"],
        "source_track_sha256":model["source_track_sha256"],
        "spatial_revision":model["spatial_revision"],
        "intervention":applied,
        "assumptions":{"decision_question":reason,
                       "feasibility_evidence_ref":feasibility,
                       "reviewer":reviewer,"provenance":"HYPOTHETICAL_REVIEWED"},
        "sensitivity":{"demand_multipliers":loads,
                       "seeds":list(SEEDS),
                       "paired_baseline":True,
                       "arrival_model":"UNIFORM_WITHIN_FIELD_COUNT_WINDOW"},
        "status":"SIMULATION_PROPOSAL_NOT_REAL_WORLD_CAUSAL_EVIDENCE",
    }


def experiment_model(baseline_model: dict, scenario: dict, load: float, changed: bool) -> dict:
    model=deepcopy(baseline_model)
    for demand in model["demand"]:
        demand["count"]=int(math.floor(demand["count"]*load+.5))
    if changed:
        intervention=scenario["intervention"]
        if intervention["kind"]=="APPROACH_SPEED_LIMIT":
            for arm in model["arms"]:
                if arm["zone_id"]==intervention["zone_id"]:
                    arm["speed_mps"]=intervention["speed_mps"]
        else:
            for phase,t in zip(model["control"]["phases"],intervention["phase_durations_s"]):
                phase["duration_s"]=t
    return model


def completed_trip_times(model: dict, data: bytes) -> dict:
    """Reject unknown ID, duplicates, surplus trips and invalid durations.

    Separate from M4 holdout evaluator: M5 scaled demand is an explicit stress
    assumption, not new independently measured physical demand.
    """
    if len(data)>120_000_000:
        raise ScenarioError("Tripinfo exceeds experiment output size budget")
    try:
        root=ET.fromstring(data)
    except ET.ParseError as exc:
        raise ScenarioError("SUMO tripinfo XML invalid") from exc
    if root.tag!="tripinfos":
        raise ScenarioError("Unexpected SUMO tripinfo root")
    ordered=sorted(model["demand"],key=lambda d:
                   (d["from_zone"],d["to_zone"],d["type_id"]))
    lookup={f"flow_{i:03d}":d for i,d in enumerate(ordered) if d["count"]>0}
    flows=defaultdict(int)
    by_move=defaultdict(list)
    seen=set()
    for entry in root:
        if entry.tag!="tripinfo":
            continue
        ident_=entry.get("id","")
        stem,sep,tail=ident_.partition(".")
        if stem not in lookup or not sep or not tail.isdigit() or ident_ in seen:
            raise ScenarioError("Unknown or duplicate output trip ID")
        seen.add(ident_)
        d=lookup[stem]
        flows[stem]+=1
        if flows[stem]>d["count"]:
            raise ScenarioError("Simulated trips exceed configured movement-class demand")
        duration=float(entry.get("duration","nan"))
        if not math.isfinite(duration) or not 0<=duration<=14400:
            raise ScenarioError("Invalid output travel time")
        by_move[(d["from_zone"],d["to_zone"])].append(duration)
    expected=sum(d["count"] for d in model["demand"])
    if expected<1:
        raise ScenarioError("Sensitivity fixture created an empty population")
    groups=[]
    for d in sorted({(d["from_zone"],d["to_zone"]) for d in model["demand"]}):
        vals=by_move[d]
        groups.append({"from_zone":d[0],"to_zone":d[1],
                       "completed_trips":len(vals),
                       "mean_trip_duration_s":sum(vals)/len(vals) if vals else None})
    return {"expected_trips":expected,"completed_trips":len(seen),
            "completion_ratio":len(seen)/expected,
            "mean_trip_duration_s":(
                sum(sum(x) for x in by_move.values())/len(seen)
                if seen else None),
            "per_movement":groups}


def paired_scorecard(pairs: list[dict]) -> dict:
    """9 fixed-seed, low/base/high-demand pairs; no p-values or empirical confidence."""
    if len(pairs)!=9:
        raise ScenarioError("Scenario completion requires all nine paired model conditions")
    differences=[]
    records=[]
    safe=True
    for p in pairs:
        baseline=p["baseline"]
        altered=p["intervention"]
        ready=(baseline["completion_ratio"]>=.95
               and altered["completion_ratio"]>=.95
               and baseline["mean_trip_duration_s"] is not None
               and altered["mean_trip_duration_s"] is not None
               and all(m["completed_trips"]>0 for m in baseline["per_movement"])
               and all(m["completed_trips"]>0 for m in altered["per_movement"]))
        safe=safe and ready
        diff=(altered["mean_trip_duration_s"]-baseline["mean_trip_duration_s"]
              if ready else None)
        if diff is not None:
            differences.append(diff)
        records.append({
            "seed":p["seed"],"demand_multiplier":p["demand_multiplier"],
            "baseline":baseline,"intervention":altered,
            "paired_difference_s":round(diff,5) if diff is not None else None,
            "comparison_eligible":bool(ready)})
    status="SIMULATED_COMPARISON_ELIGIBLE" if safe else "INDETERMINATE_INCOMPLETE_TRIPS"
    results={
        "status":status,"real_world_outcome_verified":False,
        "records":records,
        "paired_runs":len(pairs),
        "comparable_runs":len(differences),
        "all_conditions_comparable":bool(safe),
        "mean_paired_simulation_difference_s":(
            round(sum(differences)/len(differences),5) if safe else None),
        "min_paired_simulation_difference_s":round(min(differences),5) if safe else None,
        "max_paired_simulation_difference_s":round(max(differences),5) if safe else None,
        "simulated_improvement_in_every_condition":bool(safe and all(x<0 for x in differences)),
        "difference_definition":"intervention mean trip duration minus paired baseline mean (seconds)",
        "interpretation":"SIMULATION_HYPOTHESIS_ONLY",
        "limitations":[
            "Comparisons use the same model, RNG seed, and demand assumption within each pair",
            "The range is across prescribed experimental settings, NOT a statistical confidence interval",
            "A baseline holdout gate cannot establish counterfactual causal or physical validity",
            "Unfinished trips censor travel-time means; incomplete conditions block the comparison",
            "A positive or negative difference is not evidence of actual future road performance",
            "One-center network, uniform arrivals, operator-supplied behavioral parameters remain limitations",
        ],
    }
    return results
