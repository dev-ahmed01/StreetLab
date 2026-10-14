# StreetLab Integration Phase 6 / 8 — Unified Project Workspace

## Delivered

M6 makes the original `/` homepage a **real-site, project-centered operational console** while preserving every existing scientific API, local video workflow and Phase 2 synthetic demonstration under a closed **Detailed video tools and synthetic research demo** section. No existing immutable research artifact, holdout or frozen W04 model is changed.

Each project follows four progressive stages: **Observe → Reconstruct → Validate SUMO → Compare**. Only one stage's detail panel is shown at a time. The right side gives a single next action and evidence integrity explanation. Every status is read from the project-specific source and original receipts, not the global latest M1 report or synthetic Phase 2 state.

At the top, select a project or create a new one. The Observe stage supports genuine source upload, durable job queueing, cancel/retry, real-frame tracking overlay and clear status. Tracking IDs are labeled tracker IDs, never physical vehicle counts.

The Reconstruct stage gives actual independent calibration status, manual control and checkpoint counts, reviewed movement definitions and a link to the existing source-frame editor. It never invents survey coordinates.

The Validate SUMO stage gives the field census count (only when reviewed), completed baseline runtime QA, per-movement field vs SUMO mean travel-time comparisons, scenario gate, and a **local-plane schematic drawn only from reviewed measured road arms**. This is not a GPS map.

The Compare stage shows only executed, source-bound and eligible paired SUMO results: count of paired conditions, count of comparable runs, mean simulated delta and the nine condition grid. All data is labeled **simulation hypotheses, not observed or causal physical effects**. If incomplete/stale, the UI gives an unavailable result rather than fabricating savings.

## Architecture and routes

- `GET /` — newly prioritized workspace with advanced detailed tools under a closed disclosure. Old `/api/network`, `/api/simulation/*`, `/api/study/check`, `/api/decision/evaluate` routes remain intact and explicitly synthetic.
- `GET /api/projects/{project_id}/workspace` — a single, read-only integrity-verified source-of-truth response. It contains verified project, observation, geometry, baseline, scenarios and the next valid action. A downstream artifact with a damaged checksum returns HTTP 409 without silently promoting it.
- `GET /assets/streetlab-workspace.css` and `/assets/streetlab-workspace.js` — static, local, allowlisted assets without external CDN or frontend build pipeline.
- Existing `/calibration`, `/baseline` and `/scenarios` routes remain intact. Project-id query parameters preserve context across screen transitions.
- Video upload and worker queue endpoints remain the existing bounded native M2 ones, with no additional CPU inference or SUMO execution inside HTTP handlers.
- `/?advanced=1#m2Panel` automatically expands the earlier video processing and synthetic demo tools when the user needs them.

### Data-state guarantees

Stored original video and successful job receipt are always matched by project; M2 native run and verified tracking receipt establish pixel evidence. The M3 revision must bind the currently verified tracking export. M4 must bind the current M3 revision. M5 must bind the currently eligible M4 baseline. Changing geometry or baseline makes downstream stage statuses explicitly **STALE_GEOMETRY** / **STALE_BASELINE**, blocking stale decision summaries.

Top-line data preserves scientific boundaries: not all tracker IDs are distinct vehicles; a fitted local homography is not GPS; a baseline QA pass is a software heuristic; a paired experiment delta is **not real-world causal evidence**. The workspace does not use Phase 2 synthetic metrics or fabricate any local site statistics.

## Start on Windows

Inspect and preserve uncommitted changes, then fetch M6 stacked on M5:

    cd C:\Users\Admin\Desktop\StreetLab-engine-trial
    git status --short
    git fetch origin
    git switch --track origin/codex/streetlab-integration-m6

If that branch already exists, run `git switch codex/streetlab-integration-m6` instead.

Run the same verified local Python environment and existing app:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000

Open `http://127.0.0.1:8000`. For a project with an uploaded video, its standalone frozen W04 model worker must still be running to produce the tracking export. M4 baseline and M5 experiment tools also require a local SUMO/netconvert installation and actual reviewed field evidence.

## CI and acceptance

Dedicated M6 CI compiles Python, checks frontend JavaScript syntax and exercises M1–M6 software regression, homepage rendering, verified source and stage readiness, source isolation, SHA tampering, and stale revisions. The actual native SUMO M4/M5 synthetic-fixture smoke was previously verified in its own dedicated CI.

This phase verifies product wiring, software gates and presentation—not field calibration or the real source model on the user's Windows machine. These remain acceptance tasks for M8, including independent field survey, physical counts, true holdout and external validation. The application must continue to display NEEDS_DATA until those inputs exist.

## Remaining integration work

M7: secured and reproducible reports, project authorization, performance and long-run resilience.

M8: end-to-end actual-source Windows acceptance, verified field data, deployment packaging and final release criteria.