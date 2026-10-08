# StreetLab P3B Sprint 2 — SAHI small-object detection audit

Status: **EXPERIMENTAL / NOT PROMOTED**. This does **not** modify Geo-trax,
installed weights, tracking output, Phase 1/2 simulation, or old scorecards.

## Provenance

- Upstream SAHI: https://github.com/obss/sahi
- Official SAHI quick start: https://obss.github.io/sahi/quick-start/
- Original paper: https://arxiv.org/abs/2202.06934
- Official Ultralytics tiled inference instructions:
  https://docs.ultralytics.com/guides/sahi-tiled-inference/

SAHI runs **detector inference**, not persistent multiobject tracking. Each
frame yields independent bounding boxes and no identities. A rise in SAHI
detection recall is **not** proof that trajectory recall or IDF1 improves.
Direct sliced inference on every video frame may also be much slower.

## The bounded experiment

Use the exact same checkpoint, threshold, video frame and class taxonomy for
two independent inference modes:

1. **STANDARD**: whole image with `sahi.predict.get_prediction`.
2. **SLICED**: `get_sliced_prediction`, e.g. 640 × 640 pixel tiles and 20%
   overlap. Use class-aware NMS with IoU 0.5 after tile merging, without
   an additional whole-image pass (isolates sliced inference).

Convert OpenCV BGR to RGB before both predictors. The sparse frame sampler
uses absolute source frame numbers, verifies decoded position and aligns
FLUID annotations by a frozen **+1 frame offset**. If there are no annotated
motorcycles at the sampled positions, fail rather than report a misleading
recall. No synthetic trajectories or track IDs are emitted.

Metrics: precision, recall, false positives and misses by four supported
classes; total precision/recall; median/mean wall-time latency per frame.
Spatial association uses maximum-cardinality, class-aware Hungarian matching
at fixed <=50 px distance. **This detection-only matching is intentionally
different from the legacy Geo-trax pixel benchmark.** Never splice these
detection scores into the original tracking scorecards.

The decision gate is intentionally conservative: sliced mode should improve
motorcycle point recall by at least 2 percentage points, not sacrifice total
recall or car recall outside tolerances, not lose more than 2 percentage
points of precision, and not exceed 5× standard median latency. Passing only
authorizes an **experimental tracking integration**. Production promotion
requires separately measured track continuity and untouched held-out data.

## Run with existing local FLUID May-26 data

The repository checkout does NOT include the user's MP4s or local FLUID
trajectory CSV, so GitHub unit tests cannot establish real accuracy.
Use a separate optional environment:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
py -3.12 -m venv .venv-sahi-audit
.\.venv-sahi-audit\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-sahi-audit\Scripts\python.exe -m pip install "sahi[ultralytics]" opencv-python pytest
# If using the official Geo-trax ONNX instead of the .pt checkpoint:
.\.venv-sahi-audit\Scripts\python.exe -m pip install onnxruntime

$python = ".\.venv-sahi-audit\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$weights = "C:\path\to\geotrax_hbb_yolov8s_1920_v1.pt"

# First controlled W04 pilot: 61 sampled frames over 601 source frames.
& $python scripts/phase3_sahi_audit.py `
  --video $video --fluid-tracks $truth --weights $weights `
  --output-dir artifacts/phase3/sahi_audit/W04_640 `
  --start-frame 10750 --end-frame 11350 --sample-step 10 `
  --slice-height 640 --slice-width 640 --overlap 0.20 `
  --confidence 0.15 --image-size 1920 --device cpu

# If using the official Geo-trax ONNX checkpoint, ALSO include:
# --class-map config/phase3/geotrax_class_ids.json
```

If the local `StreetLab-engine-trial` worktree doesn't exist, create it
using the PR branch instructions from `docs/PHASE3_ENGINE_SHOOTOUT.md`.

Outputs in `artifacts/phase3/sahi_audit/W04_640/`:

- `report.json` — exact checkpoint and annotation hashes, sampled frame IDs,
  dependency versions, detector comparison, gate and configuration
- `standard_detections.csv`, `sliced_detections.csv` — **detections without IDs**

The command returns 2 when the candidate fails its benchmark gate, 0 when it
is eligible for a future tracking experiment, and nonzero on an error.
Previous trial folders are never overwritten. To test a revised parameter,
provide a NEW `--output-dir`.

If no motorcycles occur at the chosen 61 frames, rerun a different fixed
sample set and record that selection before viewing its scores. Do not
cherry-pick only frames where the candidate appears to help.

## Next gate

1. Compare matching W01 control window to W04/W05 difficult windows using
   frozen file hashes and the same sampling recipe.
2. If sliced recall improves sufficiently, integrate detections into an
   independently benchmarked ByteTrack/BoT-SORT candidate, keeping original
   Geo-trax results untouched.
3. Evaluate W01–W07 separately and repeat on untouched validation footage;
   reject candidate if regression or latency is too high.
4. Only then consider an optional runtime engine switch.

The P3B tracking metric fix is also available in this PR: the new
`evaluation_frame_range` argument to PixelBenchmark/IdentityBenchmark
ensures every annotated frame in the predeclared window contributes to recall,
including frames with zero detections. Historical callers retain their
original inferred-window behavior for backward compatibility.

**Licensing:** SAHI is MIT; the Ultralytics dependency has its own licensing
(AGPL-3.0 or enterprise). Do not deploy commercially before reviewing those
requirements. Preserve source, checkpoint and annotation hashes.
