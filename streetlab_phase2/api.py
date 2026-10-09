from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


NETWORK = {
    "nodes": [
        {"id": "W", "x": -220, "y": 0},
        {"id": "J", "x": 0, "y": 0},
        {"id": "E", "x": 150, "y": 0},
        {"id": "EO", "x": 320, "y": 0},
        {"id": "NE", "x": 150, "y": 150},
        {"id": "NM", "x": 0, "y": 150},
        {"id": "N", "x": 0, "y": 320},
    ],
    "edges": [
        {"id": "WJ", "from": "W", "to": "J"},
        {"id": "JN", "from": "J", "to": "NM"},
        {"id": "NS", "from": "NM", "to": "N"},
        {"id": "JE", "from": "J", "to": "E"},
        {"id": "EE", "from": "E", "to": "EO"},
        {"id": "EN", "from": "E", "to": "NE"},
        {"id": "N2", "from": "NE", "to": "NM"},
    ],
    "decision_movement": {"from_edge": "WJ", "to_edge": "JN"},
}


class AdvanceRequest(BaseModel):
    target_time_s: float = Field(gt=0)


class EvidenceInput(BaseModel):
    name: str
    provenance: str
    note: str = ""


class StudyCheckRequest(BaseModel):
    decision_type: str
    area: str = "demo_junction"
    baseline: str = "current_state"
    scenario_family: str = "decision_lab"
    requested_metrics: list[str]
    evidence: list[EvidenceInput] | None = None
    decision_question: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)


class EvaluateDecisionRequest(BaseModel):
    decision_type: str = "BLOCK_TURN"
    from_edge: str = "WJ"
    blocked_edge: str = "JN"
    duration_s: float = Field(default=120.0, gt=0)
    guided_share: float = Field(default=0.50, ge=0, le=1)
    members: int = Field(default=4, ge=2, le=50)
    horizon_s: float = Field(default=120.0, gt=0)
    requested_metrics: list[str] | None = None
    evidence: list[EvidenceInput] | None = None


def _explicit_payload(model: BaseModel) -> dict:
    data = {}
    for name in model.model_fields_set:
        value = getattr(model, name)
        if isinstance(value, list) and value and isinstance(value[0], BaseModel):
            data[name] = [item.model_dump() for item in value]
        else:
            data[name] = value
    return data


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>StreetLab Decision Lab</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#16181d;background:#f5f6f8}
*{box-sizing:border-box} body{margin:0} header{background:#fff;border-bottom:1px solid #ddd;padding:18px 28px}
main{max-width:1180px;margin:auto;padding:24px;display:grid;gap:18px}
.panel{background:#fff;border:1px solid #dfe2e7;border-radius:14px;padding:18px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{border:1px solid #e4e6ea;border-radius:10px;padding:14px}
.layout{display:grid;grid-template-columns:1.2fr .8fr;gap:18px} button{padding:10px 14px;border:1px solid #222;background:#222;color:#fff;border-radius:8px;cursor:pointer}
input,select{width:100%;padding:9px;border:1px solid #cfd3d8;border-radius:8px} label{display:grid;gap:5px;font-size:13px}
.controls{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.branches{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.branch{border:1px solid #dfe2e7;border-radius:10px;padding:14px}.muted{color:#666;font-size:13px}
.status{border:1px solid #dfe2e7;border-radius:10px;padding:12px;margin-top:12px}.status strong{display:block;margin-bottom:4px}
svg{width:100%;height:340px;background:#fafafa;border-radius:10px}.edge{stroke:#777;stroke-width:7}.detour{stroke-dasharray:10 8}.blocked{stroke:#111;stroke-width:10}
.node{fill:#fff;stroke:#222;stroke-width:3}.node-label{font-size:13px;font-weight:700}
@media(max-width:800px){.grid,.controls,.branches,.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><strong>StreetLab Decision Lab</strong><div class="muted">Decision assurance for counterfactual traffic testing</div></header>
<main>
<section class="panel" aria-label="Observed vehicle tracking">
<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline">
  <h2 style="margin-top:0">Observed traffic</h2>
  <span class="muted">Source-image tracking · research evidence</span>
</div>
<div class="grid">
<div class="metric"><div class="muted">Tracked IDs</div><strong id="obsTracks">—</strong></div>
<div class="metric"><div class="muted">Observed frames</div><strong id="obsFrames">—</strong></div>
<div class="metric"><div class="muted">Track observations</div><strong id="obsPoints">—</strong></div>
<div class="metric"><div class="muted">Site simulation readiness</div><strong id="obsReadiness">—</strong></div>
</div>
<p class="muted" id="obsClasses">No processed tracking report imported yet.</p>
<div class="status"><strong>Source provenance and missing data</strong><span id="obsGate">Real site geometry and demand must be supplied before reconstruction-based simulation.</span></div>
<p class="muted">The observation above is not the synthetic simulation below. Native tracking IDs are not guaranteed to be unique physical vehicles; pixels cannot measure m/s or turn demand without calibration.</p>
</section>
<section class="panel">
<h2>Current simulation <span class="muted">— Synthetic Decision Lab demo</span></h2>
<div class="grid">
<div class="metric"><div class="muted">Time</div><strong id="simTime">—</strong></div>
<div class="metric"><div class="muted">Active vehicles</div><strong id="activeVehicles">—</strong></div>
<div class="metric"><div class="muted">Mean speed</div><strong id="meanSpeed">—</strong></div>
<div class="metric"><div class="muted">Approach queue</div><strong id="queue">—</strong></div>
</div>
<p><button id="startSimulation">Start simulation</button> <button id="advanceSimulation">Advance to 60 s</button></p>
</section>

<section class="layout">
<div class="panel">
<h2>Network</h2>
<svg id="networkMap" viewBox="0 0 620 380" aria-label="Decision Lab network">
<line class="edge" x1="40" y1="300" x2="280" y2="300"/><line class="blocked" x1="280" y1="300" x2="280" y2="165"/>
<line class="edge" x1="280" y1="165" x2="280" y2="30"/><line class="edge" x1="280" y1="300" x2="430" y2="300"/>
<line class="edge" x1="430" y1="300" x2="590" y2="300"/><line class="edge detour" x1="430" y1="300" x2="430" y2="165"/>
<line class="edge detour" x1="430" y1="165" x2="280" y2="165"/>
<circle class="node" cx="280" cy="300" r="10"/><text class="node-label" x="294" y="294">Junction J</text>
</svg>
<p class="muted">Solid north movement: WJ → JN. Dashed path: temporary detour via JE → EN → N2.</p>
</div>

<div class="panel">
<h2>Test a decision</h2>
<div class="controls">
<label>Decision type
<select id="decisionType">
<option value="BLOCK_TURN">BLOCK_TURN</option>
<option value="APPLY_DETOUR">APPLY_DETOUR</option>
</select></label>
<label>Duration (s)<input id="duration" type="number" value="120" min="1"/></label>
<label>Guided share assumption<input id="guidedShare" type="number" value="0.5" min="0" max="1" step="0.05"/></label>
<label>Ensemble members<input id="members" type="number" value="4" min="2" max="50"/></label>
</div>
<div class="status">
<strong>Evidence sufficiency</strong>
<span id="studyStatus">NEEDS_DATA until checked</span>
</div>
<p><button id="checkStudy">Check evidence</button> <button id="evaluateDecision">Evaluate decision</button></p>
<p class="muted">Demo geometry, demand, turn movements and alternate-route availability are declared ASSUMED. Phase-1 speed evidence remains CALIBRATED.</p>
</div>
</section>

<section class="panel">
<h2>Scenario sensitivity</h2>
<p class="muted">Ranges are scenario-sensitivity ranges, not calibrated probabilities or confidence intervals. The authority makes the final decision.</p>
<div id="branchResults" class="branches"></div>
</section>
</main>
<script>
async function request(path, options={}) {
  const response=await fetch(path,{headers:{'Content-Type':'application/json'},...options});
  if(!response.ok) throw new Error(await response.text());
  return response.json();
}
const metrics=['max_approach_queue_vehicles','mean_network_speed_mps','rerouted_vehicles'];
const evidence=[
 {name:'geometry',provenance:'ASSUMED',note:'synthetic demo network'},
 {name:'demand',provenance:'ASSUMED',note:'synthetic demo demand'},
 {name:'turn_movements',provenance:'ASSUMED',note:'synthetic demo split'},
 {name:'alternate_route',provenance:'ASSUMED',note:'synthetic detour path'},
 {name:'speeds',provenance:'CALIBRATED',note:'Phase-1 calibration'}
];
function showState(s){
 document.getElementById('simTime').textContent=(s.simulation_time_s??0).toFixed(1)+' s';
 document.getElementById('activeVehicles').textContent=s.active_vehicles??0;
 document.getElementById('meanSpeed').textContent=(s.mean_speed_mps??0).toFixed(2)+' m/s';
 document.getElementById('queue').textContent=s.approach_queue_vehicles??0;
}
function studyPayload(){
 const decision=document.getElementById('decisionType').value;
 return {decision_type:decision,area:'demo_junction',baseline:'current_state',
  scenario_family:decision==='APPLY_DETOUR'?'temporary_route_management':'movement_restriction',
  requested_metrics:metrics,evidence:evidence};
}
async function checkStudy(){
 const r=await request('/api/study/check',{method:'POST',body:JSON.stringify(studyPayload())});
 document.getElementById('studyStatus').textContent=r.status+(r.missing_evidence?.length?' — missing: '+r.missing_evidence.join(', '):'');
 return r;
}
document.getElementById('startSimulation').onclick=async()=>showState(await request('/api/simulation/start',{method:'POST'}));
document.getElementById('advanceSimulation').onclick=async()=>showState(await request('/api/simulation/advance',{method:'POST',body:JSON.stringify({target_time_s:60})}));
document.getElementById('checkStudy').onclick=checkStudy;
document.getElementById('evaluateDecision').onclick=async()=>{
 const study=await checkStudy();
 const root=document.getElementById('branchResults'); root.innerHTML='';
 if(study.status!=='SUPPORTED'){
   root.innerHTML='<div class="branch"><strong>NEEDS_DATA</strong><p>'+study.message+'</p></div>';
   return;
 }
 const payload={decision_type:document.getElementById('decisionType').value,from_edge:'WJ',blocked_edge:'JN',
   duration_s:Number(document.getElementById('duration').value),guided_share:Number(document.getElementById('guidedShare').value),
   members:Number(document.getElementById('members').value),horizon_s:120,requested_metrics:metrics,evidence:evidence};
 const r=await request('/api/decision/evaluate',{method:'POST',body:JSON.stringify(payload)});
 document.getElementById('studyStatus').textContent=r.study.status;
 if(!r.decision_evaluated){root.innerHTML='<div class="branch"><strong>NEEDS_DATA</strong><p>'+r.study.message+'</p></div>';return;}
 for(const [name,b] of Object.entries(r.ensemble.branch_summaries)){
   const q=b.metrics.max_approach_queue_vehicles; const s=b.metrics.mean_network_speed_mps;
   const el=document.createElement('div');el.className='branch';
   el.innerHTML='<strong>'+name+'</strong><p>Queue median: '+q.median.toFixed(1)+' (p10–p90 '+q.p10.toFixed(1)+'–'+q.p90.toFixed(1)+')</p>'+
    '<p>Speed median: '+s.median.toFixed(2)+' m/s (p10–p90 '+s.p10.toFixed(2)+'–'+s.p90.toFixed(2)+')</p>';
   root.appendChild(el);
 }
};
async function showObservation(){
 const r=await request('/api/observations/latest');
 if(r.status==='NOT_IMPORTED'){
   document.getElementById('obsReadiness').textContent='Needs data';
   return;
 }
 document.getElementById('obsTracks').textContent=String(r.tracking_identity_count);
 document.getElementById('obsFrames').textContent=String(r.observed_frames);
 document.getElementById('obsPoints').textContent=String(r.observed_points);
 document.getElementById('obsReadiness').textContent=r.study_gate.status;
 document.getElementById('obsClasses').textContent='Tracked ID classes: '+Object.entries(r.tracks_by_class)
   .map(([name,count])=>name.replaceAll('_',' ')+' '+count).join(' · ');
 document.getElementById('obsGate').textContent=
   'Uncalibrated image pixels only. Missing for real-site scenarios: '+r.study_gate.missing_evidence.join(', ')+
   '. Baseline simulator remains a separate synthetic network.';
}
showObservation().catch(()=>{
 document.getElementById('obsGate').textContent='Observation receipt could not be verified. Check local artifact integrity.';
});
request('/api/simulation/state').then(showState).catch(()=>{});
</script>
</body>
</html>"""


def create_app(service: Any | None = None, *, observation_workdir: str | Path | None = None) -> FastAPI:
    if service is None:
        from .web_service import DecisionLabService
        service = DecisionLabService()

    # Read-only local ingestion receipts; never accept a user-supplied file path
    # over HTTP, and never reinterpret a synthetic network as observed geometry.
    if observation_workdir is None:
        observation_workdir = getattr(service, "workdir", Path(".streetlab-m5"))
    observation_workdir = Path(observation_workdir)

    app=FastAPI(
        title="StreetLab Decision Lab API",
        version="0.2.0",
        description=(
            "Decision-assurance API for evidence-gated traffic counterfactuals "
            "under explicit assumptions."
        ),
    )

    # Legacy sparse Phase 2 CI excludes integration packages; preserve its
    # standalone synthetic Decision Lab tests. Normal M2 deployments include
    # the product integration and must import it without suppressing errors.
    try:
        from streetlab_integration.product_api import mount_product_routes, m2_ui
    except ModuleNotFoundError as exc:
        if exc.name not in {"streetlab_integration", "streetlab_integration.product_api"}:
            raise
        def m2_ui() -> str:
            return ""
    else:
        mount_product_routes(app, observation_workdir)
        from streetlab_integration.spatial_api import mount_spatial_routes
        mount_spatial_routes(app, observation_workdir)
        from streetlab_integration.baseline_api import mount_baseline_routes
        mount_baseline_routes(app, observation_workdir)

    @app.get("/", response_class=HTMLResponse)
    def index()->str:
        spatial_link = (
            '<section class="panel" aria-label="Guided junction reconstruction">'
            '<h2>Junction reconstruction <span class="muted">— Guided calibration</span></h2>'
            '<p class="muted">Review a real source frame, enter measured ground-plane points, '
            'mark approaches and exits, and inspect evidence readiness. '
            'Site SUMO remains blocked until a validated network and demand exist.</p>'
            '<p><a href="/calibration">Open guided geometry editor →</a></p></section>'
            if m2_ui() else ""
        )
        baseline_link = (
            '<section class="panel" aria-label="Observed-site SUMO baseline">'
            '<h2>Observed-site baseline <span class="muted">— Phase M4</span></h2>'
            '<p class="muted">Convert manually measured road geometry, lane links, '
            'field-verified distinct-vehicle counts and independent travel-time '
            'checks into a source-bound, separate SUMO baseline.</p>'
            '<p><a href="/baseline">Open observed-site baseline workflow →</a></p></section>'
            if m2_ui() else ""
        )
        return _html().replace("<main>", "<main>" + m2_ui() + spatial_link + baseline_link, 1)

    @app.get("/api/health")
    def health()->dict[str,str]:
        return {"status":"ok","product_role":"decision_assurance","prediction_claim":"none"}

    @app.get("/api/network")
    def network()->dict:
        return NETWORK

    @app.get("/api/observations/latest")
    def latest_observation()->dict:
        from streetlab_integration.observation_bridge import latest_report
        try:
            result = latest_report(observation_workdir)
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            # Don't leak local filesystem paths to clients.
            raise HTTPException(status_code=409,
                                detail="Local observation report failed integrity validation") from exc
        if result is None:
            return {"status": "NOT_IMPORTED",
                    "message": "Import a native tracking export with scripts/streetlab_integration_m1.py",
                    "real_site_simulation_allowed": False}
        return result

    @app.post("/api/simulation/start")
    def start()->dict:
        try:
            return service.start()
        except Exception as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.get("/api/simulation/state")
    def state()->dict:
        return service.state()

    @app.post("/api/simulation/advance")
    def advance(payload:AdvanceRequest)->dict:
        try:
            return service.advance_to(payload.target_time_s)
        except Exception as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/study/check")
    def study_check(payload:StudyCheckRequest)->dict:
        try:
            data=payload.model_dump()
            if data["evidence"] is not None:
                data["evidence"]=[item.model_dump() for item in payload.evidence or []]
            return service.check_study(**data)
        except (ValueError,RuntimeError,NotImplementedError) as exc:
            raise HTTPException(status_code=400,detail=str(exc)) from exc

    @app.post("/api/decision/evaluate")
    def evaluate(payload:EvaluateDecisionRequest)->dict:
        try:
            return service.evaluate_decision(**_explicit_payload(payload))
        except (ValueError,RuntimeError,NotImplementedError) as exc:
            raise HTTPException(status_code=400,detail=str(exc)) from exc

    @app.post("/api/simulation/close")
    def close()->dict:
        return service.close()

    return app


app=create_app()
