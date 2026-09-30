# StreetLab Phase 2 M2 — Runtime Simulation + Decision State Manager

M2 turns the M1 Decision Lab demo into a reusable runtime lifecycle.

## What M2 adds

The new `RuntimeSimulation` class can:

- create and start a SUMO simulation independently of the demo script;
- step the running simulation;
- pause and resume it;
- query the current runtime state;
- inject a structured `BLOCK_TURN` decision after startup;
- save a named snapshot tied to an exact simulation time;
- hash the snapshot for deterministic provenance;
- fork multiple counterfactual branches from that same snapshot;
- record the branch decision and snapshot identity in provenance;
- compare branch metrics against a baseline.

M2 deliberately does **not** change the scientific claims from Phase 1 or M1.

The current runtime turn-block behavior still uses M1's route-response mechanics.
Natural rerouting and guided diversion remain explicit scenario assumptions, not
empirically learned Phase-1 driver behavior.

## Main files

- `streetlab_phase2/runtime.py`
- `scripts/phase2_runtime_demo.py`
- `tests/test_phase2_runtime.py`

M1 remains available through:

- `streetlab_phase2/decision_lab.py`
- `scripts/phase2_decision_lab_demo.py`
- `tests/test_phase2_decision_lab.py`

## Run locally

Windows CMD:

    set PYTHONPATH=%CD%
    set SUMO_HOME=C:\Users\Admin\Desktop\StreetLab\.venv\Lib\site-packages\sumo
    set PATH=%SUMO_HOME%\bin;%PATH%

    python scripts\phase2_runtime_demo.py

Expected output artifact:

    artifacts\phase2_m2_runtime.json

## M2 lifecycle demonstrated

    create runtime
    -> start
    -> step to decision point
    -> pause
    -> inject structured decision
    -> capture decision snapshot
    -> resume
    -> fork BASELINE / NATURAL_RESPONSE / GUIDED_DIVERSION
    -> compare branch metrics

Every branch provenance contains:

- simulation ID;
- snapshot ID;
- snapshot simulation time;
- snapshot SHA-256;
- structured decision payload.

## M2 acceptance gate

M2 is accepted only when CI verifies:

1. existing M1 unit tests still pass;
2. M2 lifecycle unit tests pass;
3. the full M1 TraCI demo still runs;
4. the full M2 TraCI lifecycle demo runs;
5. the decision snapshot is tied to the requested simulation time;
6. all counterfactual branches use the same snapshot hash;
7. branch decisions are preserved in provenance;
8. at least one intervention branch affects/reroutes vehicles;
9. comparable metrics and baseline deltas are emitted.

## Product guardrail

StreetLab does not choose policy and does not claim exact prediction.

Use the wording:

> StreetLab evaluates plausible counterfactual outcomes under explicit assumptions.
