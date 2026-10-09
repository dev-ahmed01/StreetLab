# StreetLab Integration M1 — Native traffic tracking evidence in Decision Lab

**Milestone:** the first real product integration slice, independent of the frozen holdout and not another detector optimization.

## What the repository currently connects

| Stage | Existing source | Present boundary |
|---|---|---|
| Raw fixed-camera video | streetlab_phase3/video/unseen_dual_lane.py | Can produce genuine 14-column trackers; no user-facing job runner yet |
| Tracking import | streetlab_integration/observation_bridge.py (new) | Import stable source-pixel ByteTrack ID stream into an immutable observation receipt |
| Site observation normalization | streetlab_phase3/ingestion.py | P3A georeferenced providers need actual world-scale coordinates; pixels must not be used as meters |
| Junction and demand | streetlab_phase2/demo_network.py | Current network and routes are synthetic, NOT automatically reconstructed |
| Scenarios | streetlab_phase2/web_service.py and study.py | Existing assumed-input SUMO Decision Lab; real-site evidence gate remains NEEDS_DATA |
| UI | streetlab_phase2/api.py | New read-only Observed traffic section and /api/observations/latest |

## The end-to-end M1 slice

A genuine native 14-column ByteTrack primary export is validated for frame order, original nonnegative track IDs, per-frame uniqueness, four-class mapping, source-pixel geometry and confidence. We create a SHA-anchored immutable receipt including report.json, track_points_pixels.csv and SHA256SUMS.txt. The local Decision Lab reads this receipt through a fixed SHA-addressed location, not through an arbitrary HTTP path.

The report displays track ID counts, observation points, frame coverage, class changes, class mix and limited source-image pixel previews. Track IDs are NOT verified unique physical vehicles. FPS determines duration only and is NOT used to infer meters per second.

Only pixel trajectories enter the real Phase 2 M6 evidence gate as OBSERVED_AUTO. Real geometry, demand, turn movements, alternate route and world-scale speeds remain missing, so the true site study says NEEDS_DATA. This is deliberate: the existing SUMO demo is a separate, explicitly synthetic junction.

## Windows: run the first visible integration on the saved W04 primary output

Do not alter your existing frozen holdout lock. W04 is just local development/demo input for the product UI, not unseen validation.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git fetch origin
git switch codex/streetlab-integration-m1
git pull --ff-only origin codex/streetlab-integration-m1

$tracks = Get-ChildItem '.\artifacts\phase3\sahi_detector_trials' `
  -Recurse -File -Filter 'hard_nms_ios_0.30.txt' |
  Select-Object -First 1
if (-not $tracks) { throw 'Stable W04 14-column tracker export not found.' }

.\.venv\Scripts\python.exe scripts\streetlab_integration_m1.py `
  --tracks $tracks.FullName `
  --workdir '.streetlab-m5' `
  --fps 30 `
  --label 'W04_INTERNAL_DEVELOPMENT_NOT_HOLDOUT'

.\.venv\Scripts\python.exe -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and observe the tracking summary, then use the synthetic SUMO demo separately. Inspect the source report directly at http://127.0.0.1:8000/api/observations/latest. If .venv lacks dependencies, install the existing requirements.txt in your chosen Python environment.

The immutable source receipt lives under .streetlab-m5/observations/<SHA256>/. A second import of the same bytes refuses to overwrite. There is no video upload, source georeferencing or real-site simulation in M1.

## Acceptance

One frozen primary export imports; actual source hash and immutable files can be checked; the UI shows source identity class mix, and the Phase 2 real-site study truthfully says NEEDS_DATA. CI runs the bridge, API, unsafe source rejection and source immutability tests.

## Next product-first work (no research treadmill)

**M2 Guided video processing:** file selection/upload, safe queued job, progress state and one bounded frozen-model pass feeding this report. Use existing OpenVINO exporter; no threshold tuning.

**M3 Actual road geometry and demand:** calibrate pixel/world transform, verified lane/road graph and per-movement demand. No fake pixel-to-meter conversion or assumed observations.

**M4 SUMO observed-site baseline:** generate and fidelity-check a genuine site network and routes using validated observation evidence plus Phase 1 calibrated persona models; refuse impossible scenarios.

**M5 Scenario and report:** a complete user journey with scenario comparison, uncertainty, saved projects, downloadable report, runtime p95, testing and deployment.

No W04 primary IDs, FLUID labels, old PR or preregistered independent holdout algorithms are modified in M1.
