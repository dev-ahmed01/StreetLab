from __future__ import annotations

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


class EvaluateDecisionRequest(BaseModel):
    decision_type: str = "BLOCK_TURN"
    from_edge: str = "WJ"
    blocked_edge: str = "JN"
    duration_s: float = Field(default=120.0, gt=0)
    guided_share: float = Field(default=0.50, ge=0, le=1)
    members: int = Field(default=4, ge=2, le=50)
    horizon_s: float = Field(default=120.0, gt=0)


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
.controls{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.branches{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.branch{border:1px solid #dfe2e7;border-radius:10px;padding:14px} .muted{color:#666;font-size:13px}
svg{width:100%;height:340px;background:#fafafa;border-radius:10px} .edge{stroke:#777;stroke-width:7}.detour{stroke-dasharray:10 8}.blocked{stroke:#111;stroke-width:10}
.node{fill:#fff;stroke:#222;stroke-width:3}.node-label{font-size:13px;font-weight:700}
@media(max-width:800px){.grid,.controls,.branches,.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><strong>StreetLab Decision Lab</strong><div class="muted">Decision assurance for counterfactual traffic testing</div></header>
<main>
<section class="panel">
<h2>Current simulation</h2>
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
<p class="muted">Solid north movement: WJ → JN. Dashed path: guided detour via JE → EN → N2.</p>
</div>

<div class="panel">
<h2>Test a decision</h2>
<div class="controls">
<label>Closure duration (s)<input id="duration" type="number" value="120" min="1"/></label>
<label>Guided share assumption<input id="guidedShare" type="number" value="0.5" min="0" max="1" step="0.05"/></label>
<label>Ensemble members<input id="members" type="number" value="4" min="2" max="50"/></label>
</div>
<p><button id="evaluateDecision">Evaluate BLOCK_TURN</button></p>
<p class="muted">The guided share is an explicit scenario assumption. StreetLab evaluates plausible counterfactuals; the authority makes the final decision.</p>
</div>
</section>

<section class="panel">
<h2>Scenario sensitivity</h2>
<p class="muted">Ranges are scenario-sensitivity ranges, not calibrated probabilities or confidence intervals.</p>
<div id="branchResults" class="branches"></div>
</section>
</main>
<script>
async function request(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
function showState(s){
  document.getElementById('simTime').textContent=(s.simulation_time_s ?? 0).toFixed(1)+' s';
  document.getElementById('activeVehicles').textContent=s.active_vehicles ?? 0;
  document.getElementById('meanSpeed').textContent=(s.mean_speed_mps ?? 0).toFixed(2)+' m/s';
  document.getElementById('queue').textContent=s.approach_queue_vehicles ?? 0;
}
document.getElementById('startSimulation').onclick=async()=>showState(await request('/api/simulation/start',{method:'POST'}));
document.getElementById('advanceSimulation').onclick=async()=>showState(await request('/api/simulation/advance',{method:'POST',body:JSON.stringify({target_time_s:60})}));
document.getElementById('evaluateDecision').onclick=async()=>{
  const payload={decision_type:'BLOCK_TURN',from_edge:'WJ',blocked_edge:'JN',
    duration_s:Number(document.getElementById('duration').value),
    guided_share:Number(document.getElementById('guidedShare').value),
    members:Number(document.getElementById('members').value),horizon_s:120};
  const r=await request('/api/decision/evaluate',{method:'POST',body:JSON.stringify(payload)});
  const root=document.getElementById('branchResults'); root.innerHTML='';
  for(const [name,b] of Object.entries(r.ensemble.branch_summaries)){
    const q=b.metrics.max_approach_queue_vehicles;
    const s=b.metrics.mean_network_speed_mps;
    const el=document.createElement('div'); el.className='branch';
    el.innerHTML='<strong>'+name+'</strong><p>Queue median: '+q.median.toFixed(1)+' (p10–p90 '+q.p10.toFixed(1)+'–'+q.p90.toFixed(1)+')</p>'+
      '<p>Speed median: '+s.median.toFixed(2)+' m/s (p10–p90 '+s.p10.toFixed(2)+'–'+s.p90.toFixed(2)+')</p>';
    root.appendChild(el);
  }
};
request('/api/simulation/state').then(showState).catch(()=>{});
</script>
</body>
</html>"""


def create_app(service: Any | None = None) -> FastAPI:
    if service is None:
        from .web_service import DecisionLabService

        service = DecisionLabService()

    app = FastAPI(
        title="StreetLab Decision Lab API",
        version="0.1.0",
        description=(
            "Decision-assurance API for testing plausible traffic "
            "counterfactuals under explicit assumptions."
        ),
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "product_role": "decision_assurance",
            "prediction_claim": "none",
        }

    @app.get("/api/network")
    def network() -> dict:
        return NETWORK

    @app.post("/api/simulation/start")
    def start() -> dict:
        try:
            return service.start()
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/simulation/state")
    def state() -> dict:
        return service.state()

    @app.post("/api/simulation/advance")
    def advance(payload: AdvanceRequest) -> dict:
        try:
            return service.advance_to(payload.target_time_s)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/decision/evaluate")
    def evaluate(payload: EvaluateDecisionRequest) -> dict:
        try:
            return service.evaluate_decision(**payload.model_dump())
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/simulation/close")
    def close() -> dict:
        return service.close()

    return app


app = create_app()
