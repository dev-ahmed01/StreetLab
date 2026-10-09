# Phase 3 Build 3/4: continuous OpenVINO box-policy ByteTrack experiment

**Experimental only.** All scripts under `scripts/phase3_continuous_box_tracking_lab.py` are separate from production Geo-trax/T000. No original FLUID annotations, frozen matches, source media, prior SAHI report, or production tracker is mutated. Draft PR #10 remains unmerged.

## What Build 3 delivers

One REAL **contiguous** source-video frame stream passes through the SHA-verified W04 OpenVINO model exactly once per frame (per-tile inference, model-local NMS, absolute bounding boxes). The results are fanned out to 25 existing Build 1 geometry policies, each owning an entirely separate **Roboflow ByteTrack** instance. They receive exactly one update per decoded frame, including warmup and empty-detection frames. No fake identities or sparse 21-frame tracker evaluations are permitted.

Each candidate emits frozen **14-column Geo-trax pixel tracks** with real confirmed tracker IDs, independent +1/50px PixelBenchmark and IdentityBenchmark metrics, separate class-aware diagnostic reports (including **MOTORCYCLE correct-class recall**), and a manifest including SHA256 of all candidate track files. Tracking timing and box-policy timing are measured separately from one shared detector inference timing stream. Original FLUID is parsed **once** for all candidates, avoiding 25 reads of the whole CSV.

All 25 postprocessing candidates are scored in a single pass: the unmerged control and Hard NMS / linear Soft NMS / Gaussian Soft NMS / weighted fusion, across IoU and IoS with thresholds 0.30 / 0.50 / 0.70. **These methods are not guaranteed physically safe**: overlapping real motorcycles might be merged. The output never auto-selects a winning policy or promotes production. The original class-mapping ontology is retained; two-wheeler vs bicycle and car vs bus are never silently merged.

## Actual local run (later, only when ready)

The Python environment must contain SAHI, OpenVINO, supervision and the `trackers` ByteTrack package already used by the existing Phase 3 trial. The 4K source must be a real continuous MP4; the sample JPEGs from the W04 archive are insufficient to evaluate identity stability. Set `$model` to the SHA-matching exported OpenVINO directory used in the W04 experiment.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"
$video = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4"
$truth = "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video_Traj.csv"
$model = "artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01/checkpoint_openvino_model"
$bundle = "artifacts/phase3/sahi_detector_trials/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip"

& $python scripts/phase3_continuous_box_tracking_lab.py `
  --video $video --bundle $bundle --runtime-model $model `
  --fluid-tracks $truth `
  --start-frame 10750 --end-frame 10950 `
  --warmup-frames 90 --fps 30 `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_box_tracking_batch201_01"
```

The `--bundle` path is the exact file produced earlier and uploaded to the conversation; if the ZIP remains elsewhere on the local PC, use its actual path. For valid source-camera frame rate, pass the measured FPS rather than guessing. A long 4K CPU batch is expected to take substantial time; the code **does not** claim a real throughput improvement before the local run. `--policies raw_unmerged hard_nms_iou_0.50 ...` can intentionally select fewer from the fixed matrix for a bounded smoke test, not for favorable post-hoc selection.

Optionally provide `--baseline-tracks <unchanged same-window T000.txt>` to score a same-frame baseline; the system fails closed if the cohort differs. Baseline evidence alone is not a promotion gate: the May-26 W04 sequence is previously tuned, Build 2 lacks real independent reviewers, and unseen CCTV remains unvalidated. Full FLUID original CSV SHA must match W04 provenance; no relabeling is performed.

## Evidence and limitations

- `batch_report.json`: continuous and warmup frame counts, per-policy tracks/pixel/identity/class scorecard, detector median and per-policy postprocess/tracker medians, no promotion.
- `source_provenance.json`: original video, FLUID and model/bundle SHA256 plus exact frame interval, written transactionally with the report.
- `<policy>.txt` / `<policy>.score.json`: exact tracked vehicles and per-policy evidence, written within one atomic staged output directory. Any gap, duplicate/invalid track row, invalid label cohort, or model hash mismatch fails the trial; no partial output directory is published.
- Existing frozen P3B matcher remains unchanged. Pixel and identity benchmark matches are **class-agnostic spatial associations**; `class_diagnostics` reports correct-class recall independently. The tracker is a real ByteTrack when the local inference path is executed; CI tests use explicit synthetic tracker doubles and **do not claim physical identity performance**.
- Captured boxes are per-tile **post local YOLO NMS** and pre SAHI global merge. No raw network logits, optimization of model weights, or guaranteed safe suppression is implied.
- The 25-policy matrix on **already tuned W04** is for development. Independent adjudicated images and untouched holdout footage are required before selecting a production rule.

## Next Build 4

A heldout evaluation harness should consume evidence from this Build 3, the Build 2 *real two-reviewer* agreement file (when available), and untouched reference tracks, then enforce per-class recall, physical vehicle preservation, identity fragmentation and CPU-budget gates without any automatic production promotion.