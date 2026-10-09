# StreetLab Integration Phase 2 / 8 — Video processing

## Scope and provenance

M2 adds local projects, source video ingestion, a SQLite-backed queue, a separately running CPU worker, frozen original IoS 0.30 OpenVINO/SAHI/ByteTrack, four real-source preview overlays, and M1's verified observation receipts. Research shadow policies, W04 FLUID annotations, the independent holdout lock, and Phase 1/2 calibrated behavior are untouched.

M1 remains separate from the synthetic SUMO demo. Actual site simulation continues to say NEEDS_DATA until world geometry, demand, turns, scale and physical speeds are established. Tracker IDs are **not** a confirmed census of distinct vehicles.

**Branch history:** M2 is based on codex/streetlab-integration-m1 (PR #11), itself stacked on research PR #10. Neither branch is merged or promoted.

## Components

- API/UI: streetlab_integration/product_api.py mounted in streetlab_phase2/api.py. Source videos are posted as bounded raw bytes, not remote filesystem paths. Existing M1 UI and the synthetic demo are preserved.
- SQLite: .streetlab-m5/video_jobs.sqlite3, durable WAL rows for projects, sources, jobs, SHA, config and status.
- Local files: .streetlab-m5/projects and .streetlab-m5/runs; sources are immutable UUID-scoped files, output is staged then atomically published with a SHA manifest.
- Worker: streetlab_integration/worker.py is a separate process, validates SHA-pinned W04 model and input, claims one job at a time with BEGIN IMMEDIATE, persists progress and a 600-second lease. An expired lease requeues on next claim. Runs reuse independently verified completed artifacts.
- Frozen runner: streetlab_integration/primary_runner.py reuses original SAHI per-tile model inference, hard IoS .30 suppression and actual ByteTrack IDs. No shadow candidate promotion or threshold sweeps.
- Observation: verified source-pixel native 14-column tracks are imported via the existing M1 immutable receipt mechanism. Each project's report can be verified separately.
- Job states: QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED. Cancellation is cooperative between frames; retry is explicit. Source video hashing occurs before and after processing.

**Limits:** a source is at most 2 GiB, up to one hour by decoder metadata. Upload uses a disk spool so allow extra free space. The worker requires the actual existing OpenVINO model, SAHI, supervision, trackers and OpenCV runtime; CI does not pretend to have those model assets. The local API has no user login: keep it bound to localhost only. Previews are sample frames, not a fully annotated MP4.

## Windows (no need to redo your M1 import)

Check local Git state before changing branches:

~~~powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git status --short
git branch --show-current
git fetch origin
git branch -a --list '*streetlab-integration-m2*'
# If no local M2 branch exists:
git switch --track origin/codex/streetlab-integration-m2
# If it already exists use: git switch codex/streetlab-integration-m2
git pull --ff-only origin codex/streetlab-integration-m2
~~~

Do not modify the frozen CORRESPONDENCE_LOCK_V1.json. Keep using the environment that successfully ran the original SAHI tracking model. Replace the model path below with the *actual existing immutable W04 model directory*. The provenance path is the default already used by the existing holdout launcher.

Terminal 1, worker:

~~~powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
$modelDir = 'C:\PATH\TO\YOUR\FROZEN\W04\openvino_model'
$provenance = '.\artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01\source_provenance.json'
Test-Path $modelDir
Test-Path $provenance
& '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.worker --workdir '.streetlab-m5' --model-dir $modelDir --frozen-provenance $provenance
~~~

Terminal 2, dashboard:

~~~powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
& '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000
~~~

Browse http://127.0.0.1:8000, create a project, upload a video, and start analysis. Leave the last-frame field blank to process the whole video; use a smaller end frame for a bounded real-source smoke test. Refreshes do not cancel the queued worker job.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET / POST | /api/projects | List/create projects |
| GET | /api/projects/{id} | Project + video metadata |
| POST | /api/projects/{id}/video | Binary video and X-StreetLab-Filename header |
| GET / POST | /api/projects/{id}/jobs | List/queue source-frame analysis |
| GET | /api/jobs/{id} | Persistent progress |
| POST | /api/jobs/{id}/cancel | Safe cancellation |
| POST | /api/jobs/{id}/retry | Explicit retry |
| GET | /api/jobs/{id}/report | Verify selected M1 observation receipt |
| GET | /api/jobs/{id}/overlays | List SHA-verified preview images |
| GET | /api/jobs/{id}/overlays/{name} | Read a verified JPEG preview |

## Phase status and roadmap

**M2 is implemented but not scientifically verified on real footage yet.** CI can check database lifecycle, hashing, HTTP, native-track receipt and a locally generated OpenCV decoder smoke, but this is not a real-video OpenVINO benchmark. We still need a local CPU run against original model/video, p50/p95 measured runtime and manually inspected overlays to accept the phase.

Integration roadmap: M1 observation bridge (completed); M2 video processing (this phase); M3 guided spatial calibration; M4 observed-site SUMO fidelity; M5 real scenarios and uncertainty; M6 unified dashboard and calibration UI; M7 exports/security/performance; M8 full validation and delivery.
