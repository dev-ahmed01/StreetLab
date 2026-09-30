# StreetLab Phase 3 P3B — FLUID Benchmark

P3B validates StreetLab's raw-video observation pipeline against FLUID's
published processed ground truth before an Indian junction is used as the
production pilot.

## Locked studies

1. fluid_fidrt_20250120 — development/calibration-debug study
   - trajectory window: ~1270.9 s
   - signal coverage: ~1295 s
   - has drone telemetry

2. fluid_fidrt_20250526 — held-out validation study
   - trajectory window: ~1729.3 s
   - signal coverage: ~1735 s
   - has drone telemetry

3. fluid_fidrt_20250525 — regression reference
   - trajectory window: ~407.4 s
   - signal coverage: ~410 s
   - has drone telemetry

The held-out validation study must not be used to tune extractor parameters.

## Benchmark outputs

For every study StreetLab computes:

- trajectory points and unique tracks
- canonical vehicle-class mix
- known movement counts
- complete vs incomplete route observations
- mean complete-route travel time
- signal coverage
- telemetry recording coverage
- camera/drone height statistics
- persona-calibration mutation guard

## Partial observations

FLUID legitimately contains vehicles that enter or leave outside the observed
camera boundary. Examples include missing in/out times and movements such as
S-Unknown.

P3B treats those as partial observations:
- retain trajectory evidence
- do not promote Unknown movement to a turning movement
- exclude incomplete routes from travel-time metrics
- report complete/incomplete route counts explicitly

## Raw-video comparison

The raw MP4 is processed by an external extraction provider such as Geo-trax.
StreetLab then compares the extracted output to FLUID ground truth using a
fixed benchmark contract.

The target P3B scorecard will contain:
- track coverage
- class agreement
- position error
- speed error
- movement agreement
- route agreement

## Behaviour boundary

FLUID ground truth validates observation extraction and site reconstruction.
It does not retrain the Phase-1 Indian behaviour personas.

MOTORCYCLE, CAR and AUTO_RICKSHAW remain the calibrated StreetLab families.
Other vehicle classes remain explicitly ASSUMED until separately calibrated.
