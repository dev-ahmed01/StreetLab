# StreetLab — Evidence-First Traffic Observation and Decision Support

StreetLab is a **local-first, research-backed traffic analysis platform** joining source-video observations, verified junction geometry, field-supported traffic demand, and reproducible SUMO experiments. It is not a certified road-design system, nor does a simulator prove an intervention's real-world outcome.

## Current product

The application provides a source-bound four-step project workspace:

1. **Observe** — create a project, upload an original fixed-camera traffic video, run a separate frozen OpenVINO/SAHI/ByteTrack worker, and inspect verified source-pixel tracks.
2. **Reconstruct** — manually calibrate a local ground-plane using independent measurements and describe entry/exit polygons, lanes, and allowed movements. Pixel coordinates are not GPS coordinates. Tracker IDs are not physical vehicle counts.
3. **Validate SUMO** — supply an independent field vehicle census and observed trip holdout; execute a real native SUMO baseline in a separate CLI and review model-to-field agreement.
4. **Compare** — execute controlled, nine-condition (18 SUMO runs) simulation experiments on a verified model. Modeled differences are not observed causal improvements.

Other pages: `/fieldwork` (prepare **blank** genuine fieldwork forms), `/reports` (SHA-bound provenance export) and `/release` (formal operator acceptance blockers).

**Implementation:** M1–M9 integration code is delivered. A genuine local Windows W04 model run, independent physical site measurements, and external review are still prerequisites to scientific acceptance or deployment.

## Quick start — Windows PowerShell

Use a fresh clone of the **main** branch once the integration PRs have merged:

```powershell
git clone --depth 1 https://github.com/dev-ahmed01/StreetLab.git
cd StreetLab
py -3.12 -m venv .venv-sahi-audit
& .\.venv-sahi-audit\Scripts\python.exe -m pip install -r requirements.txt pytest opencv-python-headless
.\scripts\STREETLAB_M8.ps1 -Mode environment
.\scripts\STREETLAB_M8.ps1 -Mode web
```

Open **http://127.0.0.1:8000**. The local app is **not authorized for public exposure**. The video worker is a second process requiring the actual frozen W04 OpenVINO model, SHA-pinned provenance and original runtime dependencies; these are intentionally not included in GitHub. The project database and original media are stored in local `.streetlab-m5/` and are never automatically copied from previous research environments.

```powershell
.\scripts\STREETLAB_M8.ps1 -Mode worker -ModelDir 'C:\REAL\W04_MODEL' -FrozenProvenance 'C:\REAL\source_provenance.json'
```

On Windows, use `scripts/STREETLAB_M9.ps1 -Mode plan|kit|hashes|check` after real M3 reconstruction. See [M9 operator guide](README_INTEGRATION_M9.md) and [M8 formal acceptance](README_INTEGRATION_M8.md). Do not create synthetic counts or trip times to clear the gates.

## Clean installation and testing

```powershell
& .\.venv-sahi-audit\Scripts\python.exe scripts\repository_hygiene.py --root .
& .\.venv-sahi-audit\Scripts\python.exe scripts\repository_smoke.py
& .\.venv-sahi-audit\Scripts\python.exe -m pytest -q tests\test_streetlab_integration_m*.py tests\test_phase2_m6_api.py
```

CI also validates the real native SUMO executable against synthetic test fixtures; see the [Actions](https://github.com/dev-ahmed01/StreetLab/actions). Real-site video inference and physical field survey are NOT tested on GitHub runners.

## Repository and data hygiene

The main code contains no committed Python installation or virtualenv. `.gitignore` blocks these, Python bytecode, private footage, local databases, original field records and newly generated simulation artifacts. A tracked set of **original Chennai research trajectories and generated research outputs** is retained under `data/` for historical scientific reproducibility, subject to its separate permission/provenance review.

The Git history still holds old tracked virtualenv binaries; deleting them from the current tree is **not a history rewrite**. To remove historical sensitive blobs or shrink all historical clone objects requires a separate coordinated rewrite after archived backups and collaborator migration. Do not assume that public Git history becomes private when files are removed.

## Documentation

- [M1–M9 field-to-simulation sequence](README_INTEGRATION_M9.md)
- [M8 release acceptance and original-file schema](README_INTEGRATION_M8.md)
- [M7 security, reports and backup limits](README_INTEGRATION_M7.md)
- [Phase 1 traffic behavior work](README_PHASE2_M1.md)
- [Phase 3 video / holdout research](README_PHASE3_P3B.md)
- [Repository security and release controls](docs/SECURITY_RELEASE.md)

**Scope warning:** Frozen research artifacts, W04 correspondence lock, independent holdout and prior scientific evidence must not be silently changed by maintenance. Software test success is not independent ground truth, approved real-road intervention or deployment authorization.
