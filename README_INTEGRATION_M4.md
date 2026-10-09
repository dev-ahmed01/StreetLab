# StreetLab Integration Phase 4 / 8 — Observed-Site SUMO Baseline

M4 connects actual M2 source evidence and M3 reviewed local-meter geometry to an explicitly supplied site-specific network, manual traffic census and independent holdout. It never converts tracker IDs to physical counts, auto-invents routes or applies Phase 2 synthetic network geometry to a real site.

## Implemented

- /baseline: uncluttered project-bound evidence intake and independent baseline readiness, source revision and verified XML artifact links.
- M4 immutable input revisions with SHA checks and source bindings to video, tracking export and M3 geometry.
- A source-specific netconvert package: nodes.nod.xml, edges.edg.xml, connections.con.xml, optional signals.tll.xml, routes.rou.xml.
- Optional separate CLI executes actual netconvert and sumo with bounded time and deterministic seed, never inside the HTTP server.
- Trip completion and movement-specific mean travel-time comparison to a manually recorded separate physical holdout.

## JSON evidence contract

POST /api/projects/{project_id}/baseline accepts JSON with spatial_revision (exact M3 revision) and site:

- center_world_m: actual surveyed [x,y] in M3's local meters.
- arms: one per M3 zone, carrying zone_id, outer_world_m, lane_count, speed_mps and measurement_ref.
- connections: from_zone, to_zone, from_lane and to_lane covering every manually allowed M3 turn.
- control: UNSIGNALIZED_CONFIRMED with measurement_ref, OR FIXED_TIME_SIGNAL with phase duration_s/state and link_index_review_ref.
- window: duration_s (60 to 7200 integer seconds), survey_session_ref.
- vehicle_types: id, vclass, length_m, min_gap_m, accel_mps2, decel_mps2, max_speed_mps, sigma, measurement_ref. Explicit field provenance required.
- demand: from_zone, to_zone, type_id, count, measurement_ref, provenance exactly FIELD_COUNT_DISTINCT_VEHICLES; no tracker-based census.
- holdout: a separately observed travel time per M3 movement, with mean_travel_time_s, samples (at least 3), session_ref, measurement_ref.
- review: reviewer, network_evidence_ref, demand_evidence_ref, holdout_evidence_ref, holdout_independent=true.

The code fixture field_fixture() in tests/test_streetlab_integration_m4.py demonstrates software schema only. Its numerical measurements are entirely synthetic and NOT suitable for a real junction or demonstration of physical accuracy.

## Windows usage

Preserve local modifications before switching. Fetch and select codex/streetlab-integration-m4, stacked onto M3:

    cd C:\Users\Admin\Desktop\StreetLab-engine-trial
    git status --short
    git fetch origin
    git switch --track origin/codex/streetlab-integration-m4

If this local branch already exists, use git switch codex/streetlab-integration-m4 instead.
Start the existing FastAPI application and open http://127.0.0.1:8000/baseline . Save independently verified real-world field data to obtain M4 SHA revision.

    & '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000

Install actual SUMO and check sumo --version and netconvert --version. In a separate PowerShell terminal:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.site_baseline --workdir '.streetlab-m5' --project 'REAL_PROJECT_UUID' --revision 'REAL_M4_SHA256'

Replace both placeholders with real data from the UI. For a local SUMO executable folder not on PATH, add --sumo-bin-dir with that trusted path.

## Scientific and runtime gates

- Unverified or missing M3 geometry, lane mapping, real physical census, survey-backed type parameters, or independent holdout: reject package creation.
- A saved blueprint stays REVIEWED_INPUTS_UNEXECUTED with real_site_sumo_allowed=false.
- Real separate SUMO execution must succeed and produce a checksum-bound runtime and tripinfo report.
- QA passes only if at least 95% of manually entered distinct vehicle demand completes and every movement's mean simulated travel time differs by at most 20% from an operator-attested independent holdout.
- These QA thresholds are product heuristics, NOT scientific certification. The operator's physical measurements, site realism, camera distortion, vehicle dynamics, observed arrival times, signal plans and model external validity still require review.
- A passing baseline authorizes only provisional M5 scenario consideration, not any causal or actual-road performance claim.

Only the standalone CLI runs SUMO, using an explicit executable, bound on subprocess execution, no user-supplied HTTP path, an atomic runtime output directory and SHA-verified files. If SUMO isn't available no run is claimed. CI checks both mocked subprocess orchestration and ACTUAL installed SUMO/netconvert execution against two SYNTHETIC software-site fixtures (unsignalized and fixed-time signal). These passing runs verify toolchain integration but are not a real-field measured road simulation.

SUMO reference: https://sumo.dlr.de/docs/netconvert.html and https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html .

## Next

M5: observed-site counterfactuals with explicit intervention and sensitivity. M6: unified decision UI. M7: reporting/reliability/security. M8: real-field acceptance and release.