"""M4 canonical SUMO plain-XML compiler and baseline-fidelity reader.

See SUMO docs for --node-files/--edge-files/--connection-files/--tllogic-files
and the <flow number=...> and <tripinfo> contracts.
"""
from __future__ import annotations

from collections import defaultdict
import math
import xml.etree.ElementTree as ET

from streetlab_integration.site_inputs import SiteInputError

def xml(root):
    return ET.tostring(root,encoding="utf-8",xml_declaration=True)

def render_sumo(model):
    nodes=ET.Element("nodes")
    cx,cy=model["center_world_m"]
    controlled=model["control"]["kind"]=="FIXED_TIME_SIGNAL"
    ET.SubElement(nodes,"node",{
        "id":"j","x":str(cx),"y":str(cy),
        "type":"traffic_light" if controlled else "priority"})
    edges=ET.Element("edges")
    for a in model["arms"]:
        zone=a["zone_id"]
        x,y=a["outer_world_m"]
        ET.SubElement(nodes,"node",id="n_"+zone,x=str(x),y=str(y),type="dead_end")
        incoming=a["role"]=="APPROACH"
        ET.SubElement(edges,"edge",{"id":"e_"+zone,
            "from":"n_"+zone if incoming else "j",
            "to":"j" if incoming else "n_"+zone,
            "numLanes":str(a["lane_count"]),"speed":str(a["speed_mps"])})
    connectors=ET.Element("connections")
    for index,connection in enumerate(model["connections"]):
        attrs={"from":"e_"+connection["from_zone"],"to":"e_"+connection["to_zone"],
               "fromLane":str(connection["from_lane"]),
               "toLane":str(connection["to_lane"])}
        if controlled:
            attrs.update(tl="j",linkIndex=str(index))
        ET.SubElement(connectors,"connection",attrs)
    routes=ET.Element("routes")
    for vehicle in model["vehicle_types"]:
        ET.SubElement(routes,"vType",{
            "id":vehicle["id"],"vClass":vehicle["vclass"],
            "length":str(vehicle["length_m"]),"minGap":str(vehicle["min_gap_m"]),
            "accel":str(vehicle["accel_mps2"]),"decel":str(vehicle["decel_mps2"]),
            "maxSpeed":str(vehicle["max_speed_mps"]),"sigma":str(vehicle["sigma"])})
    movements=sorted({(d["from_zone"],d["to_zone"]) for d in model["demand"]})
    route_ids={}
    for i,pair in enumerate(movements):
        rid=f"route_{i:03d}"
        route_ids[pair]=rid
        ET.SubElement(routes,"route",id=rid,
                      edges="e_"+pair[0]+" e_"+pair[1])
    ordered_demand=sorted(model["demand"],
                          key=lambda d:(d["from_zone"],d["to_zone"],d["type_id"]))
    for index,d in enumerate(ordered_demand):
        if d["count"]>0:
            ET.SubElement(routes,"flow",{
                "id":f"flow_{index:03d}","type":d["type_id"],
                "route":route_ids[(d["from_zone"],d["to_zone"])],
                "begin":"0","end":str(model["window"]["duration_s"]),
                "number":str(d["count"]),"departLane":"best"})
    result={"nodes.nod.xml":xml(nodes),"edges.edg.xml":xml(edges),
            "connections.con.xml":xml(connectors),"routes.rou.xml":xml(routes)}
    if controlled:
        logic=ET.Element("tlLogics")
        program=ET.SubElement(logic,"tlLogic",id="j",type="static",
                              programID="site_reviewed",offset="0")
        for p in model["control"]["phases"]:
            ET.SubElement(program,"phase",duration=str(p["duration_s"]),
                          state=p["state"])
        result["signals.tll.xml"]=xml(logic)
    return result


def evaluate_tripinfo(model, raw: bytes):
    """Evaluate only completed actual SUMO trips vs separate field holdout."""
    try:
        root=ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SiteInputError("SUMO tripinfo could not be parsed") from exc
    if root.tag!="tripinfos":
        raise SiteInputError("SUMO returned an unexpected tripinfo root")
    ordered=sorted(model["demand"],
                   key=lambda d:(d["from_zone"],d["to_zone"],d["type_id"]))
    mapping={f"flow_{i:03d}":d for i,d in enumerate(ordered) if d["count"]>0}
    per_movement=defaultdict(list)
    seen=set()
    per_flow=defaultdict(int)
    for el in root:
        if el.tag!="tripinfo":
            continue
        ident=el.get("id","")
        flow,sep,number=ident.partition(".")
        if not sep or not number.isdigit() or flow not in mapping or ident in seen:
            raise SiteInputError("Unexpected or duplicate SUMO trip identifier")
        seen.add(ident)
        per_flow[flow]+=1
        if per_flow[flow]>mapping[flow]["count"]:
            raise SiteInputError("Completed trips exceed measured movement-class demand")
        duration=float(el.get("duration","nan"))
        if not math.isfinite(duration) or not 0<=duration<=14400:
            raise SiteInputError("SUMO trip duration was invalid")
        d=mapping[flow]
        per_movement[(d["from_zone"],d["to_zone"])].append(duration)
    total=sum(d["count"] for d in model["demand"])
    completed=len(seen)
    if completed>total:
        raise SiteInputError("Completed SUMO trips exceed submitted physical counts")
    results=[]
    for check in sorted(model["holdout"],key=lambda h:(h["from_zone"],h["to_zone"])):
        pair=(check["from_zone"],check["to_zone"])
        times=per_movement[pair]
        observed=check["mean_travel_time_s"]
        mean=sum(times)/len(times) if times else None
        error=abs(mean-observed)/observed if mean is not None else None
        results.append({"from_zone":pair[0],"to_zone":pair[1],
                        "holdout_mean_s":observed,
                        "holdout_trip_samples":check["samples"],
                        "simulated_completed_trips":len(times),
                        "simulated_mean_s":round(mean,4) if mean is not None else None,
                        "relative_error":round(error,6) if error is not None else None})
    completion=completed/total
    # Explicit stringent QA heuristic; does NOT establish a validated causal model.
    thresholds={"minimum_vehicle_completion_ratio":.95,
                "maximum_each_movement_travel_time_relative_error":.20}
    ok=completion>=.95 and all(
        r["relative_error"] is not None and r["relative_error"]<=.20 for r in results)
    return {"status":"BASELINE_FIDELITY_CHECKED" if ok else "BASELINE_NEEDS_REVIEW",
            "real_site_sumo_allowed":bool(ok),
            "simulated_trips_completed":completed,"manual_count_demand":total,
            "completion_ratio":round(completion,6),
            "per_movement":results,"thresholds":thresholds,
            "provenance":"SUMO_COMPARED_TO_OPERATOR_ATTESTED_INDEPENDENT_HOLDOUT",
            "scenario_executed":False,
            "limitations":[
                "QA thresholds are product heuristics, not an accredited traffic-model validation standard",
                "Survey attestations and independent holdout correctness still require field verification",
                "Mean travel times and completion are insufficient to validate all driving behavior",
                "Different observation sessions may differ in weather, incidents and vehicle mix",
            ]}
