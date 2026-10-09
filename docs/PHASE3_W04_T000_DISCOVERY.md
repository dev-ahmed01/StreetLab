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
