"""M4 strictly reviewed local-road network and independent field-count inputs.

No physical count is inferred from tracker IDs, and no coordinate is inferred
from a synthetic example. Survey evidence is attested, not independently proven.
"""
from __future__ import annotations

import math
import re
from streetlab_integration.reconstruction import _json_bytes

class SiteInputError(ValueError):
    pass

PATTERN=re.compile(r"^[a-z][a-z0-9_]{0,29}$")
CLASSES={"passenger","motorcycle","bus","truck","delivery","taxi"}

def number(v,label,lo,hi):
    if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:
        raise SiteInputError(f"{label}: expected a finite number in [{lo}, {hi}]")
    return float(v)

def text(v,label):
    if not isinstance(v,str) or not 12<=len(v.strip())<=500:
        raise SiteInputError(f"{label}: supply a specific 12–500 character evidence reference")
    return v.strip()

def ident(v,label):
    if not isinstance(v,str) or PATTERN.fullmatch(v) is None:
        raise SiteInputError(f"{label}: use a safe lowercase identifier")
    return v

def point(v,label):
    if not isinstance(v,list) or len(v)!=2:
        raise SiteInputError(f"{label}: expected [local_x_m, local_y_m]")
    return [number(x,label,-100000,100000) for x in v]

def entries(v,label,minimum,maximum):
    if not isinstance(v,list) or not minimum<=len(v)<=maximum:
        raise SiteInputError(f"{label}: expected {minimum}–{maximum} entries")
    if not all(isinstance(x,dict) for x in v):
        raise SiteInputError(f"{label}: entries must be objects")
    return v

def pair(v):
    return (ident(v.get("from_zone"),"approach"),
            ident(v.get("to_zone"),"exit"))

def validate_site(raw,spatial):
    if not isinstance(raw,dict) or len(_json_bytes(raw))>200000:
        raise SiteInputError("M4 site inputs must be a JSON object no larger than 200 kB")
    geo=spatial["model"]
    q=spatial["quality"]
    if q["status"]!="GUIDED_GEOMETRY_REVIEWED" or not q["metric_transform_checked"]:
        raise SiteInputError("M3 needs manually reviewed turns and independently checked local meters")
    zones={z["id"]:z for z in geo["zones"]}
    movements={(m["from_zone"],m["to_zone"]) for m in geo["movements"]}
    center=point(raw.get("center_world_m"),"junction center")
    arms={}
    for a in entries(raw.get("arms"),"surveyed arms",len(zones),len(zones)):
        zid=ident(a.get("zone_id"),"arm zone")
        if zid not in zones or zid in arms:
            raise SiteInputError("Each M3 zone must have exactly one independently measured road arm")
        outer=point(a.get("outer_world_m"),"arm endpoint")
        dist=math.dist(center,outer)
        if not 15<=dist<=2000:
            raise SiteInputError("Surveyed arm lengths must be 15–2000 meters")
        count=a.get("lane_count")
        if type(count) is not int or not 1<=count<=12:
            raise SiteInputError("Lane count must be an observed integer from 1 to 12")
        if zones[zid]["lane_count"] is not None and count!=zones[zid]["lane_count"]:
            raise SiteInputError("M4 lane count conflicts with M3 manual geometry")
        arms[zid]={"zone_id":zid,"role":zones[zid]["role"],"outer_world_m":outer,
                   "lane_count":count,"speed_mps":number(a.get("speed_mps"),"posted speed",1,45),
                   "measurement_ref":text(a.get("measurement_ref"),"arm survey")}
    links=[]
    seen=set()
    supported=set()
    for c in entries(raw.get("connections"),"lane links",len(movements),150):
        a,b=pair(c)
        if (a,b) not in movements:
            raise SiteInputError("Lane link does not belong to an M3 manually reviewed turn")
        i,j=c.get("from_lane"),c.get("to_lane")
        if (type(i) is not int or type(j) is not int or
            not 0<=i<arms[a]["lane_count"] or not 0<=j<arms[b]["lane_count"]):
            raise SiteInputError("Lane link is not supported by surveyed lane counts")
        key=(a,b,i,j)
        if key in seen:
            raise SiteInputError("Duplicate lane-level connection")
        seen.add(key); supported.add((a,b))
        links.append({"from_zone":a,"to_zone":b,"from_lane":i,"to_lane":j})
    if supported!=movements:
        raise SiteInputError("All M3 turns require at least one surveyed lane link")
    links.sort(key=lambda c:(c["from_zone"],c["to_zone"],c["from_lane"],c["to_lane"]))
    control=raw.get("control")
    if not isinstance(control,dict) or control.get("kind") not in (
        "UNSIGNALIZED_CONFIRMED","FIXED_TIME_SIGNAL"):
        raise SiteInputError("Control must be confirmed unsignalized or an explicitly documented fixed-time signal")
    control={"kind":control["kind"],
             "measurement_ref":text(control.get("measurement_ref"),"signal/control evidence"),
             **({"phases":control.get("phases"),"link_index_review_ref":control.get("link_index_review_ref")}
                if control["kind"]=="FIXED_TIME_SIGNAL" else {})}
    if control["kind"]=="FIXED_TIME_SIGNAL":
        phases=[]
        for p in entries(control["phases"],"fixed signal phases",2,20):
            duration=number(p.get("duration_s"),"phase duration",1,300)
            state=p.get("state")
            if not isinstance(state,str) or len(state)!=len(links) or set(state)-set("rRyYgG"):
                raise SiteInputError("Signal states must match the sorted lane-link index exactly")
            phases.append({"duration_s":duration,"state":state})
        if sum(p["duration_s"] for p in phases)>1200 or not any("G" in p["state"] or "g" in p["state"] for p in phases):
            raise SiteInputError("Signal cycle exceeds 1200 seconds or contains no green")
        control["phases"]=phases
        control["link_index_review_ref"]=text(control["link_index_review_ref"],"signal-link order")
    window=raw.get("window",{})
    duration=number(window.get("duration_s"),"observation window",60,7200)
    if not duration.is_integer():
        raise SiteInputError("Observation window must be whole seconds")
    session=text(window.get("survey_session_ref"),"physical vehicle count session")
    review=raw.get("review")
    if not isinstance(review,dict) or review.get("holdout_independent") is not True:
        raise SiteInputError("An independent holdout must be explicitly acknowledged")
    reviewer=text(review.get("reviewer"),"reviewer identity")
    network_ref=text(review.get("network_evidence_ref"),"network provenance")
    demand_ref=text(review.get("demand_evidence_ref"),"manual vehicle census provenance")
    holdout_ref=text(review.get("holdout_evidence_ref"),"independent holdout provenance")
    if len({session,demand_ref,holdout_ref})<3:
        raise SiteInputError("Count session, count record and holdout evidence references must differ")
    kinds={}
    for v in entries(raw.get("vehicle_types"),"measured vehicle personas",1,12):
        vid=ident(v.get("id"),"vehicle type")
        if vid in kinds or v.get("vclass") not in CLASSES:
            raise SiteInputError("Vehicle types must be unique and use supported SUMO vehicle classes")
        kinds[vid]={"id":vid,"vclass":v["vclass"],
                    "length_m":number(v.get("length_m"),"vehicle length",1,30),
                    "min_gap_m":number(v.get("min_gap_m"),"gap",.1,10),
                    "accel_mps2":number(v.get("accel_mps2"),"acceleration",.2,8),
                    "decel_mps2":number(v.get("decel_mps2"),"deceleration",.5,12),
                    "max_speed_mps":number(v.get("max_speed_mps"),"max speed",1,55),
                    "sigma":number(v.get("sigma"),"driver imperfection",0,1),
                    "measurement_ref":text(v.get("measurement_ref"),"vehicle-parameter provenance")}
    counts=[]
    unique=set()
    for d in entries(raw.get("demand"),"movement-class physical census",len(movements),200):
        a,b=pair(d); typ=ident(d.get("type_id"),"vehicle class")
        key=(a,b,typ)
        if (a,b) not in movements or typ not in kinds or key in unique:
            raise SiteInputError("Census must use unique permitted movement and declared type")
        unique.add(key)
        n=d.get("count")
        if type(n) is not int or not 0<=n<=20000:
            raise SiteInputError("Physical counts must be nonnegative integers")
        if d.get("provenance")!="FIELD_COUNT_DISTINCT_VEHICLES":
            raise SiteInputError("Tracker IDs are NOT verified physical vehicle counts")
        counts.append({"from_zone":a,"to_zone":b,"type_id":typ,"count":n,
                       "provenance":"FIELD_COUNT_DISTINCT_VEHICLES",
                       "measurement_ref":text(d.get("measurement_ref"),"count evidence")})
    if {(d["from_zone"],d["to_zone"]) for d in counts}!=movements or not sum(d["count"] for d in counts):
        raise SiteInputError("Every M3 turn needs a documented, nonempty total field census")
    checks={}
    for h in entries(raw.get("holdout"),"independent movement travel-time holdouts",
                     len(movements),len(movements)):
        a,b=pair(h)
        if (a,b) not in movements or (a,b) in checks:
            raise SiteInputError("There must be exactly one independent check per M3 turn")
        samples=h.get("samples")
        if type(samples) is not int or not 3<=samples<=100000:
            raise SiteInputError("Each movement check needs >=3 observed trips")
        check_session=text(h.get("session_ref"),"holdout session")
        if check_session==session:
            raise SiteInputError("Holdout sessions must differ from fitting/calibration census")
        checks[(a,b)]={"from_zone":a,"to_zone":b,
                       "mean_travel_time_s":number(h.get("mean_travel_time_s"),
                                                   "physical observed travel time",1,3600),
                       "samples":samples,"session_ref":check_session,
                       "measurement_ref":text(h.get("measurement_ref"),"holdout measurement")}
    if set(checks)!=movements:
        raise SiteInputError("Every movement requires an independent travel-time check")
    normalized={
        "schema_version":1,"spatial_revision":spatial["revision"],
        "coordinate_system":"LOCAL_GROUND_PLANE_METERS_NOT_GPS",
        "center_world_m":center,"arms":list(arms.values()),
        "connections":links,"control":control,
        "window":{"duration_s":int(duration),"survey_session_ref":session},
        "vehicle_types":list(kinds.values()),"demand":counts,
        "holdout":list(checks.values()),
        "review":{"reviewer":reviewer,"network_evidence_ref":network_ref,
                  "demand_evidence_ref":demand_ref,"holdout_evidence_ref":holdout_ref,
                  "holdout_independent":True},
    }
    return normalized,{
        "status":"REVIEWED_INPUTS_UNEXECUTED","real_site_sumo_allowed":False,
        "simulation_ready_for_baseline":True,"manual_distinct_vehicle_count":sum(x["count"] for x in counts),
        "tracker_ids_used_as_counts":False,
        "limitations":[
            "All physical site inputs are attested by an operator; evidence authenticity is not externally verified",
            "Junction modeled as one planar central node; actual lane curves and access points may be more complex",
            "Vehicle types are provided parameters, not implicitly inherited from Phase 1",
            "Each measured count is spaced uniformly over the observed window for baseline startup",
            "Passing mean travel-time checks will not establish causal accuracy or full site validity",
        ]}
