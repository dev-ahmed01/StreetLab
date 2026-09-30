# StreetLab Phase 3 P3B — Windows Geo-trax Runner

This runbook processes a local FLUID aerial video with Geo-trax and compares
the extracted pixel-space tracks against FLUID ground truth.

## 1. Keep the benchmark data outside Git

Recommended local layout:

C:\Users\Admin\Desktop\StreetLabData\Video_1\
  20250120_video.mp4
  20250120_video_Traj.csv
  20250120_video_signal.csv
  20250120_video_Route.csv
  FIDRT_20250120_video.csv

Video 1 is the P3B development/calibration-debug study.

Video 2 is held out for validation and must not be used to tune extraction
settings:

C:\Users\Admin\Desktop\StreetLabData\Video_2\
  20250526_video.mp4
  20250526_video_Traj.csv
  20250526_video_signal.csv
  20250526_video_Route.csv
  FIDRT_20250526_video.csv

## 2. Open PowerShell in the StreetLab repository

Example:

cd C:\Users\Admin\Desktop\StreetLab

Activate the normal StreetLab environment if you already use one:

.\.venv\Scripts\Activate.ps1

The Geo-trax extractor is deliberately installed into its own environment,
.venv-geotrax, so its computer-vision dependencies do not alter StreetLab's
simulation environment.

## 3. Install/verify Geo-trax

Run once:

powershell -ExecutionPolicy Bypass -File scripts\phase3_setup_geotrax_windows.ps1

This creates .venv-geotrax and installs the pinned version:

geo-trax==1.5.1

The default detector weights download automatically on first extraction.

## 4. Run Video 1 smoke benchmark first

Use only the first 600 frames:

powershell -ExecutionPolicy Bypass -File scripts\phase3_run_geotrax_windows.ps1 -StudyId fluid_fidrt_20250120 -VideoPath "C:\Users\Admin\Desktop\StreetLabData\Video_1\20250120_video.mp4" -FluidTracks "C:\Users\Admin\Desktop\StreetLabData\Video_1\20250120_video_Traj.csv" -Mode Smoke -SmokeFrames 600

The runner:

1. checks/installs Geo-trax;
2. runs Geo-trax with --no-geo;
3. writes its tracks under artifacts\phase3\geotrax;
4. finds the generated <video-stem>.txt;
5. compares it against FLUID cx/cy ground truth;
6. writes pixel_scorecard.json.

The first benchmark intentionally uses pixel coordinates because the FLUID
trajectory table contains both image-space cx/cy and world-space cx_m/cy_m,
while Geo-trax requires an orthophoto for defensible world georeferencing.

## 5. Inspect the smoke scorecard

Expected output location:

artifacts\phase3\geotrax\fluid_fidrt_20250120\smoke\pixel_scorecard.json

Metrics:

- point_recall
- point_precision
- track_coverage
- class_agreement
- pixel_mae
- pixel_rmse
- detected frame_offset

This smoke benchmark is a setup check, not the final extraction result.

## 6. Run the full Video 1 benchmark

Only after the smoke run completes successfully:

powershell -ExecutionPolicy Bypass -File scripts\phase3_run_geotrax_windows.ps1 -StudyId fluid_fidrt_20250120 -VideoPath "C:\Users\Admin\Desktop\StreetLabData\Video_1\20250120_video.mp4" -FluidTracks "C:\Users\Admin\Desktop\StreetLabData\Video_1\20250120_video_Traj.csv" -Mode Full -SkipSetup

Do not tune on Video 2.

## 7. Freeze extraction settings, then run Video 2

After Video 1 settings are accepted:

powershell -ExecutionPolicy Bypass -File scripts\phase3_run_geotrax_windows.ps1 -StudyId fluid_fidrt_20250526 -VideoPath "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4" -FluidTracks "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv" -Mode Full -SkipSetup

Video 2 is the held-out validation set.

## Benchmark boundary

The raw-video benchmark validates:

raw video -> detection -> tracking -> class normalization -> pixel trajectory

It does not recalibrate StreetLab's Phase-1 Indian behaviour personas.

MOTORCYCLE, CAR and AUTO_RICKSHAW remain the calibrated behaviour families.

## Next P3B step after pixel validation

Once pixel extraction generalizes to Video 2, add georeferencing:

video + orthophoto -> Geo-trax world coordinates -> FLUID cx_m/cy_m comparison

That unlocks direct speed/trajectory validation in metres and then the SUMO
baseline-reconstruction stage.
