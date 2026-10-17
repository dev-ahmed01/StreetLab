# StreetLab W04 — frozen asset readiness (read-only)

The local research SUMO track can run without W04 media. W04 is a **separate, previously tuned FLUID video development study**, not a blind held-out evaluation. Once SUMO works on Windows, use this audit to verify available local model/video evidence before scheduling a new comparison.

## Windows source audit (no inference)

From a current local `main` checkout, with the already-installed Python environment:

```powershell
cd C:\Users\Admin\Desktop\StreetLab-final
$video = 'C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4'
$model = 'C:\Users\Admin\Desktop\StreetLab-engine-trial\artifacts\phase3\sahi_detector_trials\W04_openvino_export640_01\checkpoint_openvino_model'
$provenance = 'C:\Users\Admin\Desktop\StreetLab-engine-trial\artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01\source_provenance.json'
.\scripts\STREETLAB_W04_READINESS.ps1 -Video $video -ModelDir $model -FrozenProvenance $provenance
```

If you have the exact original FLUID source truth referenced by the `source_provenance.json` fingerprint, supply `-Truth '...\20250526_video_Traj.csv'`. Do not substitute a derived, re-exported, changed, or different-day CSV.

The script hashes the entire original video and frozen model tree (it can take time for 1.7 GB video) and creates a **new** `artifacts/w04_readiness_*.json` file. It checks against SHA-256 fields in the existing local frozen provenance; matching a supplied provenance file does **not** independently authenticate the provenance itself. It does not decode frames, run the detector, alter the media, mutate model weights, or compute precision/recall.

Missing truth is reported honestly as a separate blocker. A valid video/model status means only *development asset integrity*, never held-out pass or production approval.

## What remains for genuine W04/holdout science

1. **Original Build 3 verification:** If the full `W04_box_tracking_batch201_01` directory, original truth and original source paths remain available, use the existing `scripts/phase3_holdout_validation.py inspect` command to verify all 25 saved policy outputs and original source fingerprints. The 201 evaluation frames are **10750–10950** (90 prior warmup frames). These measurements were used during development.
2. **Predeclare policy, then evaluate untouched footage:** The provisional W04 development leader is `hard_nms_ios_0.30`, but its 86.70% point precision and 91.37% recall are development FLUID-label metrics, not blind generalization. Use the existing `lock` stage **before** opening new holdout results. Do not treat another window of the tuned W04 footage as automatically independent.
3. **Same-window independent baseline:** Recover an authentic original Geo-trax/T000 baseline only if it exists and matches the holdout window. Otherwise transparently label a newly generated same-window *replacement* baseline; do not misrepresent it as recovered original T000.
4. **Independent human review and performance:** Need matching original FLUID trajectories, a separate untouched test video, two-reviewer physical-object checks including rare classes, tracking-ID stability, and CPU end-to-end p95. The existing `validate` gate remains fail-closed where evidence is missing.

## Evidence boundaries

This audit is **not** a new W04 inference run and does **not** turn the separate Chennai behavior data and FLUID observations into a calibrated junction. The Chennai holdout failure and field-acceptance conditions remain untouched. No physical site inspection is required for the *limited research prototype*, but real-site-certified behavior still needs external validation.
