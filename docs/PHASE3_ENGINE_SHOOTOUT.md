# StreetLab P3B — Engine shootout (experimental, not deployed)

The existing **Geo-trax 1.5.1 extraction and its FLUID baseline remain untouched**.
This is an independent Python experiment adapter, not a replacement for Geo-trax
or a claim of better accuracy. It does not modify Phase 1/2, calibration or
the original FLUID labels.

## Why these engines?

Evidence reviewed from upstream projects:

- Geo-trax: pinned aerial-view detector already used in StreetLab
  (`hf://rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt`).
- ByteTrack: low-confidence detection rescue; cheap association for mostly
  stationary-camera sequences. Original: https://github.com/ifzhang/ByteTrack
- BoT-SORT: association with camera-motion compensation, optional ReID.
  Original: https://github.com/NirAharon/BoT-SORT
- OC-SORT: observation-centric recovery, worth a later candidate **only after**
  checking the installed Ultralytics version supports `ocsort.yaml`.
  Original: https://github.com/noahcao/OC_SORT
- BoxMOT: broader pluggable tracker alternatives, but AGPL obligations and
  added dependencies mean it is not installed as part of this first trial.
  https://github.com/mikel-brostrom/boxmot

For a defensible comparison, keep the **same aerial detector weights**, source
video, test windows, scale, and FLUID benchmark, and vary only the tracker.
A fresh pretrained COCO model is not automatically better for small overhead
motorcycles. If detector recall remains low with both trackers, audit/retrain
the aerial detector next. Trackers cannot recover objects never detected.

## What is implemented

- `streetlab_phase3/video/engine_shootout.py`: external Ultralytics experiment
  adapter with no mandatory torch/OpenCV dependency on importing StreetLab.
- `scripts/phase3_engine_shootout.py`: video/window CLI and fixed-offset FLUID
  pixel + identity scoring.
- `config/phase3/geotrax_class_ids.json`: explicit checkpoint-specific IDs for ONNX models without class labels.
- `config/phase3/bytetrack_recall.yaml` and `botsort_recall.yaml`: explicit
  **untested** candidate presets; neither is a production default.
- `tests/test_phase3_engine_shootout.py`: mock-engine contract regression suite.
- Geo-trax-compatible **14 columns**; absolute zero-based video frame indexes;
  90 prior sequential warm-up frames; no fabricated IDs; native bbox centers;
  no false claim of stabilized/georeferenced positions.
- Safeguards: failed/exhausted trials cannot overwrite baseline exports,
  and the promotion gate checks same score window, +1 frame offset, recall,
  precision, class agreement, pixel error, coverage and identity fragmentation.
  A candidate must beat baseline recall or fragmentation by over 1 percentage
  point while staying within regression limits.

**Important:** Geo-trax original exports stabilized coordinates too. This
adapter sets stabilized fields equal to raw centers strictly to satisfy the
legacy 14-column parser and marks `stabilized_coordinates=false` in manifest.
P3B's pixel evaluator uses raw centers. Do not use these experimental exports
for world-coordinate site calibration until actual stabilization is provided.

## Data availability

GitHub main has benchmark code and `data/benchmarks/fluid_fidrt/manifest.json`,
but **does not store the MP4 videos, parsed FLUID trajectory CSVs or the completed
T000–T002 local extraction artifacts**. Model inference and quantitative
comparison cannot be executed in GitHub-only CI. The available tests verify the
adapter, export schema, failure handling and promotion logic, not real recall.

The Stage-A W01–W07 windows and T000 results live locally on the user's
Windows machine. In particular, `W04` is frames 10750–11350, with warm-up
10660–10749. Keep exact same-window T000 scorecards for comparison.

## Local Windows run (PowerShell from StreetLab root)

Use an isolated environment so Geo-trax 1.5.1 dependencies cannot be broken.

```powershell
py -3.12 -m venv .venv-engine-trial
.\.venv-engine-trial\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-engine-trial\Scripts\python.exe -m pip install "ultralytics>=8.3,<9" opencv-python pytest

# Locate the aerial detector .pt already cached by Geo-trax.
Get-ChildItem "$env:USERPROFILE\.cache\huggingface" -Filter "geotrax_hbb_yolov8s_1920_v1.pt" -Recurse -ErrorAction SilentlyContinue |
  Select-Object -First 5 FullName

.\.venv-engine-trial\Scripts\python.exe -m pytest tests/test_phase3_engine_shootout.py -q
.\.venv-engine-trial\Scripts\python.exe -m pip freeze > artifacts\phase3\engine_shootout_dependencies.txt
```

The official aerial detector also has an [ONNX export](https://huggingface.co/rfonod/geo-trax/blob/main/geotrax_hbb_yolov8s_1920_v1.onnx).
If the older `.pt` checkpoint is incompatible with a newer Ultralytics/Torch
runtime, use this official `.onnx` checkpoint instead and install `onnxruntime`:

```powershell
.\\.venv-engine-trial\\Scripts\\python.exe -m pip install onnxruntime
# Set $weights to the downloaded official .onnx file.
# For this Geo-trax checkpoint ONLY, add --class-map config/phase3/geotrax_class_ids.json
```

Never apply the numeric Geo-trax mapping to a generic COCO model, where class
0 is PERSON and would otherwise be mislabeled as CAR. Model class mapping is
audited in the trial manifest.

Substitute the real `$weights` path found above and the existing FLUID CSV
path. **Do not use the original Stage-A aggregate scorecard as the
single-window baseline**: only W04's T000 pixel and identity scorecards match
this trial.

```powershell
$python = ".\.venv-engine-trial\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\path\to\20250526_video_Traj.csv"
$weights = "C:\path\to\geotrax_hbb_yolov8s_1920_v1.pt"

& $python scripts/phase3_engine_shootout.py `
  --video $video --weights $weights `
  --tracker bytetrack.yaml `
  --output artifacts/phase3/engine_shootout/W04/bytetrack_default.txt `
  --start-frame 10750 --end-frame 11350 --warmup-frames 90 `
  --confidence 0.10 --image-size 1920 --device cpu `
  --fluid-tracks $truth

& $python scripts/phase3_engine_shootout.py `
  --video $video --weights $weights `
  --tracker config/phase3/bytetrack_recall.yaml `
  --output artifacts/phase3/engine_shootout/W04/bytetrack_recall.txt `
  --start-frame 10750 --end-frame 11350 --warmup-frames 90 `
  --confidence 0.10 --image-size 1920 --device cpu `
  --fluid-tracks $truth
```

When W04's verified T000 pixel and identity scorecards exist, add
`--baseline-pixel <same-window-T000-pixel.json>` and
`--baseline-identity <same-window-T000-identity.json>`. The CLI writes
`.pixel.json`, `.identity.json`, `.manifest.json`, and, when both baseline
scorecards are supplied, `.gate.json`. Exit code **2** is a rejected
candidate, not a code crash. Do not promote a candidate without same-window
ground truth, and do not overwrite existing experiment files.

When an installed Ultralytics version lacks the custom tracker backend, use
`bytetrack.yaml` built in or install a compatible release and pin it in the
dependency snapshot. Alternative tracker YAML definitions are deliberately
separate and reversible.

## Evaluation milestones

1. **Gate 0 — integration smoke:** run the tests and 100-200 sequential frames
   with model checkpoint. Abort and remove failed trial-only outputs.
2. **Gate 1 — controlled comparison:** W01 control and W04/W05 problem windows
   using exact 90-frame warm-up. Compare default ByteTrack and recall preset
   against the same-window Geo-trax T000; inspect especially moped/motorcycle
   recall and mean predicted IDs per truth track.
3. **Gate 2 — holdout:** evaluate winning engine on untouched recorded intervals.
   Reject if precision, class agreement, positional error or ID fragmentation
   deteriorate beyond guardrails. Preserve failed reports for reproducibility.
4. **Gate 3 — next sprint:** if detection recall, not ID assignment, is
   limiting, add a domain-fine-tuned detection checkpoint candidate. Do not
   merge new engine into main until same-window real-video evidence passes.

## Licensing and reproducibility

Ultralytics and BoxMOT distribute AGPL-3.0 licensed software; a commercial
StreetLab deployment requires an explicit license/compliance decision. This
experimental module does **not vendor** their source. BoT-SORT/OC-SORT original
repositories are MIT, but the adapter's optional Ultralytics runtime still has
its own license. Record exact pip versions and the model SHA-256 before
comparing performance. Trial manifests embed the checkpoint hash when it is a
local file. No undocumented post-processing is performed.
