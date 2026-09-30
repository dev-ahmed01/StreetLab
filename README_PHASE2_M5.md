# StreetLab Phase 2 M5 — Decision Lab API + UI

M5 exposes the already-validated M2-M4 simulation lifecycle through a thin
human-facing Decision Lab.

## Objective preserved

StreetLab remains decision assurance:

> Observe current traffic state, define a traffic decision, compare plausible
> counterfactual outcomes and uncertainty, then let the authority make the
> final decision.

The web layer does **not** rank branches, recommend a policy, or convert the
M4 sensitivity ranges into prediction probabilities.

## Architecture

M5 intentionally avoids introducing a separate frontend framework yet.

- `streetlab_phase2/api.py` — FastAPI HTTP surface + minimal static Decision
  Lab UI.
- `streetlab_phase2/web_service.py` — application service connecting the web
  surface to the existing runtime and ensemble engine.
- `streetlab_phase2/runtime.py` — unchanged simulation lifecycle.
- `streetlab_phase2/ensemble.py` — unchanged M4 uncertainty engine.

The UI contains no simulation or policy logic. It only sends structured
requests and renders API results.

## Current API

- `GET /` — Decision Lab UI
- `GET /api/health`
- `GET /api/network`
- `POST /api/simulation/start`
- `GET /api/simulation/state`
- `POST /api/simulation/advance`
- `POST /api/decision/evaluate`
- `POST /api/simulation/close`

The current vertical slice supports the existing demo movement:

`BLOCK_TURN: WJ -> JN`

## Decision evaluation semantics

The web service:

1. observes the current running simulation;
2. pauses at the authority's decision point;
3. captures one immutable SUMO snapshot;
4. creates BASELINE, NATURAL_RESPONSE, GUIDED_DIVERSION and MIXED_RESPONSE;
5. runs an M4 ensemble from that same snapshot;
6. returns auditable branch summaries and provenance;
7. leaves the live simulation paused at the original decision point.

No evaluated branch is written back into the live runtime.

## UI

The initial UI shows:

- current simulation time;
- active vehicles;
- mean speed;
- approach queue;
- a schematic network;
- BLOCK_TURN controls;
- guided-share assumption;
- ensemble-member count;
- branch cards with median and p10-p90 queue/speed ranges;
- explicit scenario-sensitivity language.

This is deliberately a functional Decision Lab shell, not the final polished
frontend.

## Run locally

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
$env:SUMO_HOME = "$PWD\.venv\Lib\site-packages\sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
uvicorn streetlab_phase2.api:app --reload
```

Then open:

`http://127.0.0.1:8000`

## Integration smoke

```powershell
python scripts\phase2_m5_api_smoke.py
```

Artifact:

`artifacts\phase2_m5_web_smoke.json`

## Scientific guardrails

M5 does not:

- retune Phase 1;
- vary calibrated microscopic parameters;
- claim rerouting response was empirically learned;
- treat scenario sensitivity as a calibrated probability forecast;
- choose a branch for the authority.

StreetLab continues to use the statement:

> StreetLab evaluates plausible counterfactual outcomes under explicit
> assumptions; the authority makes the final decision.
