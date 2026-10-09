# W04 — locate an original same-window Geo-trax / T000 track file

**Read-only search. Not a baseline certification and not a new inference request.**

The W04 Build 4 source-integrity report is now verified (`original_source_sha256_verified=true`), but `batch_report.json.same_window_T000 = null`. We must find the *original, unmodified* Geo-trax output corresponding to the May-26 source video before comparing the experimental 25 policies to T000. The required source-video frame window is **10750–10950 inclusive** (FLUID truth frames **10751–10951**, frozen +1 and 50px).

After pulling the latest `codex/phase3-engine-shootout` branch, run from the StreetLab root:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only origin codex/phase3-engine-shootout

$roots = @(
  ".\artifacts\phase3\geotrax",
  "C:\Users\Admin\Desktop\StreetLabData\Video_2"
)
& ".\.venv-sahi-audit\Scripts\python.exe" scripts\phase3_discover_t000_tracks.py `
  --roots $roots `
  --output ".\artifacts\phase3\sahi_detector_trials\W04_T000_CANDIDATES_01.json"
```

Upload **only** `W04_T000_CANDIDATES_01.json` to analyze candidate paths, original 14-column Geo-trax layout, frame index coverage, duplicate frame/ID indicators, file size and SHA256 if complete. This command **does not** run Geo-trax, need the OpenVINO model, alter the original .txt files, or declare any candidate the true T000 baseline. Add any other known original Geo-trax/T000 folders to `$roots` before running. Avoid pointing it at your SAHI trial directory; that would list experimental policy files, not original T000.

An original full-video T000 track file may contain rows outside the 201-frame interval; that is normal. Some legitimate frames can have zero detections, so `window_frames_present < 201` is **not by itself proof** the candidate is invalid. A candidate filename or 14-column format alone also cannot establish its source video, model/config, coordinate transform, or actual original-baseline status; cross-check original experiment provenance and score it independently before scientific comparison.

If zero viable original candidate files exist, first reconstruct the precise original Geo-trax/T000 configuration and source provenance; only then decide whether a fresh baseline extraction is scientifically necessary. Do not overwrite any prior T000 artifact or substitute the W04 candidate tracks as T000.

## October 9 follow-up: original two-root inventory returned zero

The user submitted `W04_T000_CANDIDATES_01.json`, reporting **zero scanned candidate files** and **zero compatible 14-column files** under only:

- `artifacts\phase3\geotrax` (relative to the **experimental** `StreetLab-engine-trial` root)
- `C:\Users\Admin\Desktop\StreetLabData\Video_2`

This does **not** demonstrate that the authentic historical source track is absent. Earlier StreetLab work recorded a distinct **original** checkout at `C:\Users\Admin\Desktop\StreetLab`. Its expected May-26 full Geo-trax output path was `artifacts\phase3\geotrax\fluid_fidrt_20250526\full\20250526_video.txt`, and a separate T000 experiment candidate location was `artifacts\phase3\geotrax_tuning\experiments\T000_BASELINE`. These historical paths are leads, not verified existence or authenticated T000 runs.

**Preferred targeted next command** (from `C:\Users\Admin\Desktop\StreetLab-engine-trial` after pulling the experimental branch):

```powershell
git pull --ff-only origin codex/phase3-engine-shootout
& ".\scripts\RUN_W04_T000_HISTORICAL_DISCOVERY.ps1"
```

This reports `FOUND/MISSING` for each original/experimental candidate directory and generates `artifacts\phase3\sahi_detector_trials\W04_T000_HISTORICAL_CANDIDATES_01.json` without rerunning model inference. Upload the JSON plus the console's FOUND/MISSING lines. If a candidate `.txt` is found, assess its recorded file provenance, full-video frame IDs, frozen source/FLUID hashes, original Geo-trax/T000 settings and baseline score using the exact 10750–10950 window before claiming comparability.

**Scientific restriction:** `20250526_video.txt` from a generic original Geo-trax full run might be a useful control but must not automatically be named `T000` unless its original T000 model/config provenance is established. A valid track file may omit zero-detection frames; `frame_coverage_complete=false` alone is not conclusive. Even with a baseline, production promotion remains blocked by independent held-out physical review and CPU p95 evidence.



## Follow-up: historical scan returned zero again

The user supplied `W04_T000_HISTORICAL_CANDIDATES_01.json`: four historical and experimental paths targeted, **zero `.txt` track candidates**. The original scanner silently skipped missing directories, so it does **not** establish whether the older checkout or T000 output directory currently exists.

Prior T000 Stage-A work reportedly required **12,093.078 seconds (~3h22m)** and generated full-run metrics of **58.2575% point recall**, **96.6231% point precision**, and **30.0339% motorcycle recall**. These historical full-run values are *not* the same scoring cohort as the current W04 201-frame continuous benchmark; never compare them directly.

### Last bounded recovery pass, rather than another filename guess

From the experimental repo:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only origin codex/phase3-engine-shootout
& ".\scripts\RUN_W04_T000_FORENSIC_INVENTORY.ps1"
```

Upload `artifacts\phase3\sahi_detector_trials\W04_T000_FORENSIC_INVENTORY_01.json`. The standalone read-only inventory reports existence for historical paths and searches relevant artifacts, outputs, runs, experiments and benchmark directories in StreetLab-named Desktop checkouts. It recognizes track files, summaries, manifests, compressed archives and experiment configurations. Virtual environments and large model/frames folders are excluded; errors and scan truncation are disclosed. No source file is changed and no candidate is certified as the original T000.

**After that single inventory:** If a genuine T000 track is found, authenticate source video and model/config/Geo-trax version and score its exact original frame window 10750–10950 (+1 FLUID offset, 50px rule). If only summaries survive, archive them as historical evidence but do not invent missing tracks. If nothing survives, record original T000 as unavailable and reconstruct an explicitly *new* baseline only after verifying its exact processing and frame-alignment method. Proceed with cached-box class-aware ablations without new OpenVINO inference in parallel. Production promotion still requires same-window baseline, independent holdout, physical review and CPU p95 evidence.
