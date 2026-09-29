# StreetLab Phase 2 M1 — Decision-in-the-loop counterfactual simulator

This milestone is a vertical slice, not another calibration sprint.

It proves one product objective end-to-end:

> Start one mixed-traffic SUMO simulation, save the exact running state at a
> decision point, inject a BLOCK_TURN decision, fork multiple futures from the
> same state, and compare what happens.

## Reused Phase-1 evidence

The demo reads data/models/sumo_persona_priors.csv.

It creates all seven Phase-1 persona profiles across:
- motorcycle P1/P2/P3,
- car P1/P2,
- auto-rickshaw P1/P2.

The final Phase-1 longitudinal result is respected: tau stays globally
frozen at 0.50 s. Persona-specific speedFactor and minGapLat seed values
are retained as prototype heterogeneity.

Routing response is not claimed to be empirically learned in Phase 1.
Natural rerouting and guided diversion are explicit scenario assumptions.

## Counterfactual branches

Every branch begins from the same SUMO saveState snapshot.

1. BASELINE — no closure.
2. BLOCK_NATURAL_120 — direct north turn blocked for 120 s; SUMO reroutes.
3. BLOCK_GUIDED_120 — same closure; upstream northbound traffic is explicitly diverted.
4. BLOCK_GUIDED_60 — shorter 60 s closure with guided diversion.

The toy network contains a direct north movement and an alternate detour that
rejoins the same northbound destination.

## Metrics

The first vertical slice measures:
- arrivals after the decision,
- vehicles exposed to the blocked movement,
- rerouted vehicles,
- mean network occupancy,
- maximum stopped queue on the approach,
- mean network speed,
- mean instantaneous waiting time,
- deltas from baseline.

No LLM controls individual vehicles.

## Run locally

Windows CMD:

    set PYTHONPATH=%CD%
    set SUMO_HOME=C:\Users\Admin\Desktop\StreetLab\.venv\Lib\site-packages\sumo
    set PATH=%SUMO_HOME%\bin;%PATH%

    python scripts\phase2_decision_lab_demo.py

Output:

    artifacts\phase2_m1_decision_lab.json

## Automated validation

GitHub Actions installs SUMO on Ubuntu, runs unit tests, runs the full TraCI
integration demo, checks that blocked branches actually affect and reroute
vehicles, and uploads the JSON result as an artifact.

## M1 exit gate

M1 passes only when CI proves:
- all branches use the same saved pre-decision state;
- the baseline runs;
- the BLOCK_TURN event reaches vehicles upstream of the junction;
- at least one blocked branch reroutes vehicles;
- comparative metrics are emitted.

After M1 passes, M2 replaces the toy decision source with a reusable runtime
decision API/state manager rather than adding more calibration work.
