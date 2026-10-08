# StreetLab Phase 3 — SAHI + ByteTrack continuous-track candidate

**EXPERIMENTAL / ISOLATED / NO PRODUCTION PROMOTION**

This sprint joins two independent, documented upstream components:

- [SAHI](https://github.com/obss/sahi): full-image and sliced detection modes; the
  API returns full-frame `ObjectPrediction` boxes.
- [Roboflow Trackers](https://github.com/roboflow/trackers): modern
  `ByteTrackTracker.update(sv.Detections)` API (Apache-2.0). We deliberately
  avoid the deprecated `supervision.ByteTrack` wrapper. Unconfirmed detections
  reported as tracker ID `-1` are omitted from trajectory exports.
- [FLUID P3B benchmark](../streetlab_phase3/pixel_benchmark.py): frozen
  offset **+1**, fixed 50px matching threshold, full fixed evaluation window.

The original Geo-trax model, existing production configuration, FLUID labels
and phase 1/2 behavior calibrations are unchanged. The new trial consumes the
same aerial detector checkpoint for both modes and writes fresh files only.

## Continuous semantics

Do **not** call a multiobject tracker once every tenth sampled frame: that
would distort lifecycle and fragmentation. Tracking trials decode EVERY
consecutive source frame from `max(0,start-90)` through `end`, including
blank frames; the first 90 frames merely warm up. Only the declared central
window is exported. There is no optical stabilization, georeferencing,
interpolation, or invented identity.

Tracking preset (not a demonstrated optimal choice): detector confidence 0.15,
ByteTrack activation 0.20, high-confidence association split 0.15,
`lost_track_buffer=45`, `minimum_consecutive_frames=2`. Tracker's internal
frame-rate is deliberately fixed to 30 so the buffer counts 45 per-frame
updates regardless of the video fps; actual video frames are never skipped.
All values are in a manifest alongside the checkpoint SHA-256. This
configuration intentionally differs from the original BoT-SORT baseline, and
is a *candidate*, not a production policy.

The pipeline implements:
`raw video -> RGB -> SAHI detection -> sv.Detections -> ByteTrackTracker ->
confirmed IDs -> 14-column pixel tracks -> fixed-window FLUID scoring`.

## Real validation locally on FLUID

The 17,293-frame May-26 video, its FLUID trajectory CSV and T000 extraction
results are on your Windows machine; none are present in GitHub. Never
represent GitHub unit or public-image smoke tests as FLUID accuracy.

Create the independent experiment worktree as described in
[engine shootout](PHASE3_ENGINE_SHOOTOUT.md), then:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git fetch origin
git checkout --detach origin/codex/phase3-engine-shootout

$python = ".\.venv-sahi-audit\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python -m pip install "sahi[ultralytics]" "trackers==2.6.0" opencv-python

$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\path\to\geotrax_hbb_yolov8s_1920_v1.pt"

# W04: warm up 10660–10749; evaluate 10750–11350.
& $python scripts/phase3_sahi_track.py `
  --video $video --weights $weights --fluid-tracks $truth `
  --output artifacts/phase3/sahi_tracker_trials/W04_sliced.txt `
  --start-frame 10750 --end-frame 11350 --mode sliced `
  --confidence 0.15 --image-size 1920 `
  --slice-height 640 --slice-width 640 --overlap 0.20 --device cpu

# Matched non-sliced detection mode through EXACT SAME TRACKER.
& $python scripts/phase3_sahi_track.py `
  --video $video --weights $weights --fluid-tracks $truth `
  --output artifacts/phase3/sahi_tracker_trials/W04_standard.txt `
  --start-frame 10750 --end-frame 11350 --mode standard `
  --confidence 0.15 --image-size 1920 --device cpu
```

If and ONLY if the checkpoint uses unknown numeric names (e.g. the official
four-class ONNX export), add
`--class-map config/phase3/geotrax_class_ids.json`.
Never apply that numeric map to a generic COCO checkpoint (class 0 is person).

To compare either run against **the exact same-window T000 file**, pass
`--baseline-tracks "C:\path\to\W04_T000_absolute_frame_tracks.txt"`
on a fresh run with a new `--output`. This scores the baseline and candidate
in the same fixed frame window and produces a fail-closed promotion gate.
Previously generated results are never overwritten. Exit code 2 means the
candidate failed a scoring gate, not a successful promotion.

Run W01 (4850–5250) and W05 (13250–13850) using fresh unique output paths.
The same 90-frame warm-up rule applies. The first test can be smaller (e.g.
40–100 evaluation frames) to check model compatibility and runtime cost, but
small pilots **cannot** serve as the final comparison.

## Diagnose motorcycle misses without rerunning video inference

The class diagnostic reads an EXISTING Geo-trax-compatible track export and
FLUID annotation CSV, applying the identical **fixed** +1 frame alignment
and <=50 px spatial Hungarian matching used in `PixelBenchmark`.

```powershell
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
& $python scripts/phase3_class_diagnostics.py `
  --tracks artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt `
  --fluid-tracks "C:\\Users\\Admin\\Desktop\\StreetLabData\\Video_2\\20250526_video_Traj.csv" `
  --start-frame 10750 --end-frame 10759 `
  --output artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01_classes.json
```

To compare against T000, pass its real original Geo-trax tracking file as
`--tracks` with **the same frame bounds**, saving to a separate JSON path.
This does not change or rerun either engine.

The report distinguishes **spatial recall** (did any vehicle prediction
occupy a truth location?) from **correct-class recall** (was it assigned the
right class?). In a wrong-class spatial association, the truth contributes
a class error, *not* a spatial false positive. This preserves the frozen
benchmark's class-agnostic spatial association and avoids falsely claiming
a motorcycle was detected correctly when it was labeled as a car.

### CPU tiling safety

For the 3840×2160 May-26 video, running every 640-pixel slice with
`--image-size 1920` causes redundant resizing and excessive CPU work.
Choose **`--image-size 640`** with 640×640 slices for the first SAHI pilot.
The unsliced control may retain `--image-size 1920`; this explicitly
tests two *inference configurations* with the same trained model weights,
not identical image input sizes. Keep the manifest settings visible so
accuracy and compute comparisons remain honest. Never use tiny warm-up or
10-frame identity results as production quality evidence.

## First actual May-26 W04 pilots and fair offline A/B

The user has now executed the experiments locally on the same authentic video:
3840x2160, 10 FPS, original official overhead checkpoint SHA-256
`7d462ae523b15f3679a83f74ab08aa79d2978f319ca2f0cf2892fbbb60df79da`.

| Run | Source video frames | Warm-up | Model image size | Point recall | Point precision | Elapsed |
|---|---|---:|---:|---:|---:|---:|
| Standard+ByteTrack | 10750–10759 (10) | 5 | 1920 | 246/368 = 66.85% | 246/262 = 93.89% | 32.97s / 15 processed |
| SAHI+ByteTrack | 10750–10752 (3) | 2 | 640 tile and inference | 98/113 = 86.73% | 98/127 = 77.17% | 65.89s / 5 processed |

**These two headline recalls are not comparable**: the denominator and
interval differ. The 10-frame standard per-class diagnostic showed
MOTORCYCLE: 65/186 spatial matches (34.95%), 121 spatial misses, 11
unmatched motorcycle predictions, zero spatially matched class confusions;
CAR: 171/172 spatial matches (99.42%), with 21 car→bus class mismatches.
Thus motorcycle detection coverage is the primary observed weakness, while
car/bus classification is a secondary target. Identity fragmentation
measured over three or ten frames is too short to support a tracking claim.

**Do not run more inference yet.** Both raw `.txt` exports already contain
the shared source frames 10750–10752. The new offline comparator re-scores
both on the SAME three FLUID frames, reports motorcycle recall and unmatched
predictions and refuses unmatched checkpoint hashes or tracker settings:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\\.venv-sahi-audit\\Scripts\\python.exe"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
& $python scripts/phase3_compare_existing_trials.py `
  --standard-tracks "artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --fluid-tracks $truth --start-frame 10750 --end-frame 10752 `
  --output "artifacts/phase3/sahi_tracker_trials/W04_existing_3frame_comparison.json"
```

It reports both per-class and identity summaries in a new JSON, plus the
sliced-minus-standard differences.

### Paired rescue/loss and unmatched-confidence forensic report

After the first W04 comparison, an additional offline-only analysis is
available using the exact same inputs. Because results are immutable, give
the `--output` a NEW filename:

```powershell
& $python scripts/phase3_compare_existing_trials.py `
  --standard-tracks "artifacts/phase3/sahi_tracker_trials/W04_standard_pilot01.txt" `
  --sliced-tracks "artifacts/phase3/sahi_tracker_trials/W04_sliced_smoke01.txt" `
  --fluid-tracks $truth --start-frame 10750 --end-frame 10752 `
  --output "artifacts/phase3/sahi_tracker_trials/W04_paired_forensics01.json"
```

This reports **paired** truth-observation outcomes for each class: matched
by both, standard only, SAHI only, or neither; the same accounting with
correct class required; and concrete FLUID motorcycle track IDs/video frames
rescued or lost. It separately groups unmatched prediction observations by
category, confidence ([0,.25), [.25,.5), [.5,1]) and track ID appearing on
one versus multiple frames **within this 3-frame evaluation cohort**.
An ID's short presence inside the restricted cohort is NOT proof it is an
ephemeral false positive on a longer video.

Do not treat confidence groupings as calibrated probabilities or derive a
production threshold from these three tuned frames. The purpose is to
identify whether the additional 23 unmatched SAHI predictions look like
low-confidence motorcycles, wrong-class vehicles or high-confidence false
alarms before spending CPU on a controlled, longer experiment. Raw FLUID
files and private video stay on the local machine.
 It deliberately records **unequal
warm-up (5 versus 2)** and **image-size (1920 versus 640)** as confounders;
thus even the same-three-frame comparison is exploratory, not a controlled
component-level ablation or evidence for production promotion. To isolate
the effects of tiling, later rerun both configurations on identical image
size, confidence, source window and 90-frame tracker warm-up.

`T000` original Geo-trax track export has not been located in the earlier
searched `C:\Users\Admin\Desktop\StreetLab\artifacts` directory. Do
not invent its score or use a different-window summary as a replacement.

## Evidence produced

- `*.txt`: Geo-trax 14-column track file with confirmed real IDs.
- `*.manifest.json`: exact model hash, versions, bounds, runtime and export counts.
- `*.pixel.json` and `*.identity.json`: fixed-window P3B report.
- `*.baseline_pixel.json`, `*.baseline_identity.json`, `*.gate.json`:
  optional same-window baseline and promotion decision.

Important: `x_stabilized_px` and `y_stabilized_px` duplicate raw coordinates
for **parser compatibility only**. The manifest explicitly marks
`stabilized_coordinates=false`. Never feed trial tracks directly into
georeferenced simulation or world-coordinates without separate calibration.

## Evidence gates

1. **Real API integration:** public SAHI sample with real YOLO checkpoint
   produces at least one confirmed ByteTrack ID and 14-column record.
2. **Development windows:** W01/W04/W05 fixed intervals with actual FLUID
   labels; track recall, precision, class agreement, pixel MAE, fragmentation
   and elapsed processing seconds.
3. **Unseen validation:** May-26 was already used for tuning and is **not**
   an untouched holdout. Require another recording before claiming
   generalization.
4. **Promotion:** only a model that satisfies same-cohort quality and latency
   guardrails and an untouched test can be considered for production.

Any failed or regressive experimental candidate should remain isolated.
There is no silent fallback that would mislabel Geo-trax outputs as SAHI.
