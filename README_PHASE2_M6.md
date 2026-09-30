# StreetLab Phase 2 M6 — Study Contract, Evidence Sufficiency, Additional Decisions

M6 completes the initially locked Phase-2 roadmap.

## Objective

StreetLab must not simulate unsupported questions as though every requested
metric and decision is equally well evidenced.

M6 therefore adds an evidence gate before counterfactual execution:

`Study Contract -> Observation Package -> Evidence Sufficiency -> Simulation`

If required evidence is missing, StreetLab returns:

`NEEDS_DATA`

and does not pause, snapshot or fork the live simulation.

## Study Contract

The contract records:

- decision type;
- decision question;
- study area;
- baseline;
- scenario family;
- requested metrics;
- unsupported claims.

## Observation Package

Evidence items carry explicit provenance:

- `OBSERVED_MANUAL`
- `OBSERVED_AUTO`
- `SENSOR`
- `CALIBRATED`
- `INFERRED`
- `ASSUMED`

M6 never silently upgrades an assumption into an observation.

For the synthetic Decision Lab demo:

- geometry: ASSUMED
- demand: ASSUMED
- turn movements: ASSUMED
- alternate route: ASSUMED
- Phase-1 trajectories: CALIBRATED
- Phase-1 speeds: CALIBRATED

## Sufficiency gate

Supported prototype metrics declare evidence dependencies.

Examples:

- queue -> geometry + demand + turn movements;
- mean speed -> geometry + demand + speeds;
- rerouted vehicles -> geometry + demand + turn movements + alternate route.

Unsupported metrics such as an exact crash-reduction estimate return
`NEEDS_DATA` rather than a fabricated number or confidence score.

## Additional decision type

M6 adds:

`APPLY_DETOUR`

This reuses the existing validated route-response mechanics. It does not claim
that the direct movement is physically closed.

The current M6 detour comparison contains:

- BASELINE
- DETOUR_GUIDED
- DETOUR_MIXED

The mixed branch retains M3/M4's explicit assumed response uptake and ensemble
uncertainty.

## API

M6 adds:

`POST /api/study/check`

and extends:

`POST /api/decision/evaluate`

with optional `requested_metrics` and `evidence`.

Existing M5 BLOCK_TURN clients remain compatible.

## Refusal semantics

If the study gate returns `NEEDS_DATA`:

- `decision_evaluated = false`
- `snapshot = null`
- `ensemble = null`
- the live simulation remains unchanged;
- no confidence score is invented.

## Product boundary

StreetLab still does not choose policy.

> StreetLab evaluates plausible counterfactual outcomes under explicit
> assumptions; the authority makes the final decision.

Evidence sufficiency means only that the declared study has the inputs required
to run the supported prototype calculation. It does not certify that ASSUMED
inputs are observed truth.
