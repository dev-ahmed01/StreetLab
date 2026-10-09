# StreetLab Integration Phase 3 / 8 — Guided junction reconstruction

## Shipped software

M3 extends the same local StreetLab application. **Open the dashboard at http://127.0.0.1:8000 and choose "Open guided geometry editor"**, or visit http://127.0.0.1:8000/calibration directly.

- Choose an M2 project and a successfully completed real-video job. Display a genuine source-video frame, validated against the SHA-pinned stored source; the SVG UI maps clicks back to exact original-source pixel dimensions.
- Define four manually measured correspondences between source image pixels and a **local two-dimensional ground-plane in meters**. Enter the measurement/survey basis. A separate manually measured check point is required to pass the metric-export gate. Control points must be finite, inside the source image, nonduplicated and nondegenerate; check residual must be at most 1 m. Four fitted controls alone cannot independently verify scale.
- Draw and edit source-image **approach and exit polygons** on the actual frame. Add explicitly reviewed approach lane counts and allowed approach-to-exit links. Inputs are normalized and rejected when malformed, self-crossing, missing, inconsistent or too large.
- Preserve geometry, model/source hashes, camera/coordinate conventions, manual provenance and quality decisions in an immutable SHA-256 content-addressed revision tied to the exact M2 project, source, job, frozen model and native tracking receipt. New revisions never alter earlier revisions; the latest pointer persists in SQLite.
- Only if the documented ground-plane calibration has an independent passing check, emit a verified CSV containing **source-pixel coordinates and calibrated LOCAL-meter ground points**, labeled CALIBRATED. Otherwise save the draft and mark NEEDS_DATA, without outputting fabricated meter coordinates.
- Visit real vehicle tracker trajectories against manual approach/exit polygons. Emit **tracker-ID movement candidates and ambiguity statuses**, never conflate them with distinct physical vehicles or verified traffic demand.
- Fetch project-specific reconstruction status, quality gates and verified calibration export from the API. Provide a clear list of missing evidence and downstream simulation blockers.

## What M3 deliberately does not claim

The app does **not** automatically identify real roads, survey GPS, infer a city coordinate reference, solve unknown elevation, validate camera distortion, know signal phasing or treat tracker identities as unique physical traffic counts.

A local homography maps points lying on a common planar road surface only when the camera is sufficiently stationary and the measurement source is genuine. The additional check point is an **internal consistency check of operator-supplied observations**, not external certification. The nominal 1 m tolerance is a product QA gate, not an established statistical uncertainty range; field measurement and distorted camera geometry may still invalidate it.

M3 marks a geometrical source as GUIDED_GEOMETRY_REVIEWED only after all required checks and manually provided approaches, exits, lane counts, and movement links are present. **real_site_sumo_allowed remains false for every M3 revision.** Route topology, reviewed physically distinct movement demand, signal plans/assumptions and baseline fidelity will be added in M4 and tested against actual observations.

Current scope draws approach/exit polygons and permitted links, not a final lane-centerline, node-edge SUMO network. The latter is a separate observed-site simulation integration gate. No Phase 1 calibrated Indian traffic behavior, Phase 2 demo network, frozen W04 detector policy, native 14-column originals, holdout lock or FLUID annotations are altered.

## API

| Method | URL | Result |
|---|---|---|
| GET | /calibration | Guided visual geometry editor |
| GET | /api/jobs/{job_id}/source-frame?frame={index} | Genuine source frame, aligned and SHA-verified |
| GET | /api/projects/{project_id}/reconstruction | Latest source-bound model and quality |
| POST | /api/projects/{project_id}/reconstruction | Create verified immutable reconstruction revision, body {"job_id":"...","model":{...}} |
| GET | /api/projects/{project_id}/reconstruction/export | Integrity-checked source-to-local-meters CSV (only after independently verified metric calibration) |

The POST shape has "anchors" and "checks" as arrays of {"pixel":[x,y],"world_m":[X,Y],"provenance":"OBSERVED_MANUAL"}, "scale_basis" as text, "zones" with "id", "role" APPROACH/EXIT, "polygon_pixels", "lane_count", and "movements" joining a known approach and exit. No caller path or fabricated real-world geometry is accepted.

## Windows operator workflow

This is stacked on M2 PR #12. Do **not** switch branches with unsaved working-directory changes. Inspect current state, fetch, and switch safely:

~~~powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git status --short
git branch --show-current
git fetch origin
if (-not (git branch --list 'codex/streetlab-integration-m3')) {
    git switch --track origin/codex/streetlab-integration-m3
} else {
    git switch codex/streetlab-integration-m3
}
git pull --ff-only origin codex/streetlab-integration-m3
~~~

Use your **existing M2 source and verified analysis job**. Launch the same existing FastAPI server with the working Python environment:

~~~powershell
& '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000
~~~

The M3 editor does not require rerunning a video or modifying the frozen holdout. The server requires OpenCV only when you request a video-frame preview and requires NumPy for homography. Source-frame retrieval checks the entire stored source hash to prevent silently displaying tampered footage; on very large videos this makes each preview an expensive I/O action and should be optimized only with a defensible integrity mechanism.

## Verification status

Dedicated CI runs M1, M2, M3 and existing Phase 2 evidence-gate tests with compiled source and an OpenCV decoder runtime. M3 includes positive/rejected/ambiguous geometry, no-calibration refusal, independent scale checks, real-project/job binding, SQLite persistence, immutable versioning, SHA tampering and API-export guards. Synthetic geometry tests establish **software behavior**, not physical accuracy.

An end-to-end local evaluation on actual CCTV with field-surveyed points, independently checked geometry and expert-reviewed approach/exits remains necessary before deeming the site calibration reliable. M2's full OpenVINO Windows acceptance is still open. The independent Phase 3 tracking holdout remains a parallel research task, not a dependency of M3 UI/software verification.

## Remaining milestones

M4: observed-site SUMO network, source-lane mapping, verified OD demand and baseline fidelity.

M5: counterfactual site scenarios, controls, uncertainty.

M6: consolidated uncluttered operations UI and spatial/simulation charts.

M7: reports, security, multi-project performance and robustness.

M8: integrated real-world acceptance, deployment and reproducibility.
