# StreetLab Phase 2 M4 — Scenario Ensembles + Uncertainty

M4 extends the Decision Lab from one deterministic counterfactual result to a
small ensemble of plausible counterfactual members.

## Product objective

StreetLab is still decision assurance:

> Test a traffic decision virtually, compare plausible consequences, and let
> the authority make the final decision.

M4 does not recommend a policy and does not present one ensemble as a
probability forecast.

## What varies

For the current vertical slice, only the deterministic assignment seed used by
M3's explicit mixed-response assumption varies between ensemble members.

The following remain fixed:

- the saved pre-decision SUMO snapshot;
- the structured decision;
- Phase-1 longitudinal tau = 0.50 s;
- calibrated/prototype microscopic parameters;
- the macro guided-share assumption;
- the local response trigger;
- the simulation horizon.

This means deterministic branches can legitimately have zero spread, while the
mixed-response branch can expose sensitivity to which specific affected
vehicles accept guidance versus react locally.

## Reported summaries

For comparable metrics M4 reports:

- minimum;
- mean;
- median;
- p10;
- p90;
- maximum;
- spread.

The artifact also retains every member run and its member seed so the summary
is auditable.

## Scientific interpretation

The M4 interval is a **scenario sensitivity range**, not a statistical
confidence interval and not a calibrated probability that the real road will
fall inside that range.

Use this wording:

> StreetLab evaluates a range of plausible counterfactual outcomes under
> explicit assumptions; it does not predict one exact future.

## Run

    python scripts\phase2_m4_ensemble_demo.py --members 4

Output:

    artifacts\phase2_m4_ensemble.json
