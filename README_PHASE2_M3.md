# StreetLab Phase 2 M3 — Heterogeneous Response

M3 remains a Decision Lab feature, not a new behavior-research program.

Its purpose is to make one counterfactual branch more realistic by allowing
affected vehicles to respond differently to the same intervention.

## Product objective preserved

StreetLab still answers:

> What could happen if we make this traffic decision right now?

The authority still makes the final decision. M3 does not recommend policy.

## What changes

M3 adds `HETEROGENEOUS_RESPONSE`.

Affected northbound vehicles are deterministically assigned to one of two
scenario response modes:

- `GUIDED`: accepts upstream diversion early.
- `LOCAL`: responds only near the closure, using the existing late-response
  behavior.

The same vehicle/type/seed combination receives the same assumed mode across
branches and repeated runs.

## Scientific boundary

The guided share, assignment seed, and local response trigger are explicitly:

`ASSUMED`

They are not learned from Phase 1 and must not be presented as empirical
driver-compliance estimates.

Phase-1 longitudinal tau remains frozen at 0.50 s. M3 does not retune the
validation set and does not change calibrated microscopic mechanics.

## Why this matters

M2 could compare all-natural versus all-guided response. M3 can also test a
mixed population, letting an authority see whether a proposed intervention's
traffic outcome is sensitive to partial compliance.

This is uncertainty/sensitivity support for decision assurance, not prediction
of exact individual behavior.

## Run

    python scripts\phase2_m3_heterogeneous_demo.py

Output:

    artifacts\phase2_m3_heterogeneous_response.json
