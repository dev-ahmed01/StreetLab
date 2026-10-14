# Phase 3 integrated box optimization — Build 1/4

**Status:** experimental, unpromoted, reversible. All work remains on
`codex/phase3-engine-shootout` / draft PR #10. The Geo-trax T000
baseline, frozen P3B scoring, existing FLUID labels, Phase 1/2,
and production tracking are unchanged.

## What has been implemented

1. **True per-tile box capture (pre-SAHi global merge):**
   `streetlab_phase3/video/premerge_box_capture.py` runs SAHI's own
   `get_slice_bboxes` and `get_prediction` separately per tile on the
   21 source W04 frames. It records absolute x1/y1/x2/y2 coordinates,
   0-based source frame, tile ID, confidence, class, and source index.
   Local Ultralytics model-level NMS may already have happened; these
   are **pre-global-merge**, NOT raw model logits/pre-model NMS.
   Unrecognized model category names are counted and not silently
   reclassified. Six declared model families supported: CAR, BUS,
   HEAVY_VEHICLE, MOTORCYCLE, PEDESTRIAN, BICYCLE.

2. **Deterministic 25-candidate geometry matrix:**
   `streetlab_phase3/video/integrated_box_lab.py` implements one
   unmerged control and Hard-NMS, linear Soft-NMS, Gaussian Soft-NMS,
   and score-weighted box fusion, each at IoU and IoS thresholds 0.30,
   0.50, 0.70. This comparison is **class-aware by default**.
   A cross-class WBF attempt fails closed. Unlike the older 8/25-pixel
   center-only proxies, all policies here operate on actual xyxy boxes.
   The decision rules never receive FLUID truth or unmatched flags.

3. **Frozen separate scoring:** after each geometry candidate has
   been finalized, the original fixed (+1 frame, 50px,
   class-aware one-to-one maximum-cardinality) FLUID benchmark is
   recomputed with original 739 denominated W04 annotations, excluding
   unsupported classes and unprojected pixel centers. The original
   670/857, 78.18% precision, 90.66% recall scorecard from prior
   SAHI experiments stays recorded unchanged for comparison.
   The new per-tile inference is a distinct capture, so those old
   predicted box counts are **not** assumed to be identical.

4. **Immutable proof bundle:** full source ZIP members have SHA256
   checksums and a predeclared frame span 10750–11350 step 30.
   The actual loaded local OpenVINO XML/BIN export's stable tree hash
   must equal the bundle's recorded export SHA. Output staging is
   atomic and the destination may not exist. Every candidate gets its
   own CSV, plus a complete score matrix, capture provenance and raw
   pre-global-merge JSONL. Actual video decoding enforces accurate
   frame seeking. No result is eligible for tracking or promotion
   merely because it improves FLUID precision.

5. **Adjudication template:** generated `adjudication_template.json`
   prepopulates 15 original known W04 review cases with nearest raw
   FLUID types, but all new human judgments are `UNREVIEWED`. It does
   not invent missing BUS labels, modify the raw truth, or assume that
   one detector marker corresponds to one physical motorcycle.

## Container tests completed

Locally, the module passed **12 unit/integration tests**, including
original W04 cached source scores of **739 truth, 857 predictions,
670 matched, MOTORCYCLE 320/380**, exact-box IoU/IoS, close separate
motorcycles not automatically deleted, six-class tile coordinate
offsets, deterministic Soft-NMS/fusion, forged model SHA rejection,
altered bundle SHA rejection, all-25 CSV export and immutable outputs.
The real-box forward pass cannot run in the current container
(OpenVINO package unavailable): this module's OpenVINO inference is
**implemented but not yet runtime-validated**. Current GitHub CI is
source/tests and optional fake-model testing, not actual 21-frame
OpenVINO inference.

## One batched Windows invocation — when ready

No new download of the raw 2GB video is necessary. The original video,
W04 source bundle, and exported OpenVINO model remain local.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$bundle = "artifacts/phase3/sahi_detector_trials/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip"
$model = "artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01/checkpoint_openvino_model"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"

& $python scripts/phase3_integrated_box_lab.py `
  --bundle $bundle `
  --runtime-model $model `
  --video $video `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_integrated_box_lab21_01"
```

If the user saved the bundle at a different local path, adjust only
`$bundle`. If `--video` is omitted, the CLI instead runs on the
bundle's JPEG-compressed source frames, which are **not pixel-identical**
to original decoded MP4 frames. Never compare those two sources as a
single controlled inference experiment.

Outputs:

- `pre_global_merge_boxes.jsonl`: all six categories, tile IDs, xyxy
- `candidate_boxes/<policy>_<metric>_<threshold>.csv`: ALL 25
  candidates, including source contributor IDs for merged boxes
- `matrix_report.json`: original scorecard as context, candidate
  class-aware observations, all motorcycle/car denominators, guardrails
- `capture_metadata.json`: video provenance, model SHA validation,
  frame timing and output-file digests
- `adjudication_template.json`: 15 **UNREVIEWED** source cases

The intended output is a **single experiment** rather than 25 local
runs or a succession of one-line diagnostics.

## Unfinished and deliberately gated

**Build 2/4:** visually adjudicate balanced samples and ensure
motorcycle/car/bus physical-presence labels with at least one
independent reviewer; compare candidate box overlays without selecting
by FLUID unmatched flags. Include 21 W04 frames plus unseen data.

**Build 3/4:** connect the best predeclared validated boxes to a
continuous Roboflow ByteTrack run with >=90 contiguous warm-up
frames, genuine track IDs and the frozen pixel/identity benchmarks.
No synthetic IDs.

**Build 4/4:** independent held-out CCTV replication, realistic CPU
throughput/10fps estimation, fragmentation, stopline/crossing and
junction reconstruction checks; explicit go/no-go decision.

**No production promotion, existing score overwrite, or modification
to source videos/labels/tracker.** Class-aware FLUID precision is not a
validated measure of physical false-positive detections; the sampled
May-26 W04 labels include visibly genuine buses tagged CAR and a
motorcycle detection co-located with a PEDESTRIAN label.
