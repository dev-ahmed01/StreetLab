# May 25 cross-date research regression — one frozen model and policy

**Purpose:** Evaluate the *historical May 25, 2025 FLUID regression-reference video* under the May 26 W04 model and the preregistered postprocessing/tracking policy **without retuning**. It can provide a second real-video *regression diagnostic*, but **it is not a genuinely unseen blind holdout**.

This phase addresses the 18 Oct 2026 dataset inventory showing only two existing local video/trajectory pairs:

- \`Video_1/20250525_video.mp4\` + \`20250525_video_Traj.csv\`: historical **regression reference**
- \`Video_2/20250526_video.mp4\` + \`20250526_video_Traj.csv\`: already tuned/development W04

A historical manifest label of \`held_out_validation\` for May 26 does not reverse its use in method selection. No third video exists in that inventoried directory.

## Single Windows run

\`\`\`powershell
cd C:\Users\Admin\Desktop\StreetLab-final
git pull --ff-only origin main
.\scripts\STREETLAB_W04_MAY25_REGRESSION.ps1
\`\`\`

This runs an actual 150-frame CPU pipeline if all preflight checks pass. To inspect only the source, policy and FLUID readiness **without inference**, append \`-PreflightOnly\`. The default paths point at the previously inventoried May 25 original files and the previously verified frozen May 26 OpenVINO model. Alternate locations can be supplied using \`-Video\`, \`-Truth\`, \`-ModelDir\` or \`-ProtocolFolder\` while keeping the original exact historical filenames and contents.

### Preregistered cohort

- **Source frames 30–59**: 30 tracker warmup frames.
- **Source frames 60–179**: 120 contiguous scored frames.
- **Original FLUID frames 61–180**: frozen one-frame alignment, 50px gate.
- **Model:** the exact frozen OpenVINO model-tree SHA from \`policy_lock.json\`.
- **Policy:** the exact \`hard_nms_ios_0.30\` configuration in the lock, no additional candidates.
- **Detector:** same unmodified 640px SAHI tiles and 20% overlap, threshold .15, one shared detector feed into one genuine ByteTrack stream.
- **Environment:** local CPU and existing research tracker/SAHI/Ultralytics/OpenVINO dependencies in \`.venv-sahi-audit\Scripts\python.exe\`.

Why 120 frames? It is the previously locked **minimum holdout numerical window** (120) and allows a conservative, bounded, reproducible cross-date regression. It is **not** a holdout merely because 120 continuous frames were scored. The selected May 25 window is predeclared independent of any May 25 model results; this does not establish source novelty.

### Fail-closed preflight

Before detector loading it checks the original video/FLUID filenames and distinct fingerprints from tuned W04; hash-matches the frozen model directory against the real policy lock; verifies that the annotation file contains the expected \`frame,id,cx,cy,type\` fields and supported class points in the fixed window; and uses OpenCV metadata to require a 3840×2160, approximately 30 FPS source covering the cohort.

If anything fails, it **does not run inference** or silently change labels, model, frame rate or source window. A successful preflight generates an exclusive \`artifacts/w04_may25_crossdate_*/preflight.json\`.

A successful full run additionally generates:

- \`batch/\`: immutable original pre-global-merge raw boxes, absolute-frame ByteTrack IDs, one frozen policy scorecard, batch report, and May 25 SHA provenance.
- \`cross_date_summary.json\`: historical regression-reference precision/recall/F1, ID switches, per-class FLUID correctness, CPU detector/postprocess/tracker stage medians, hashes and explicit no-holdout flags.

Failed CPU inference may leave a completed \`preflight.json\` and no batch or score; do not claim a partial experiment as a pass.

### Original FLUID blank-coordinate handling

Some May 25 trajectory rows have empty cx/cy cells. The frozen `normalize_fluid_pixel_truth` scorer excludes those rows; this preflight now applies the **same exclusion** and reports both `supported_fluid_rows_before_blank_coordinate_filter` and `skipped_fluid_rows_with_blank_coordinates`. No CSV is modified, and no coordinates are interpolated or fabricated.

Nonempty invalid centers (including NaN/Infinity) still fail closed. A fixed cohort with zero usable labels cannot be evaluated; the script never chooses a different or more favorable interval automatically. The preflight receipt is created only if preflight **passes**. If checks fail before that, the Windows launcher now explicitly says no valid preflight JSON was written.

### May 25 native video geometry and frame rate

The separate May 25 regression **does not assume the W04 May 26 video resolution is universal**. The runner reads original OpenCV metadata and accepts a native source between 640×360 and 7680×4320 at 10–60 FPS, provided the fixed video frames 30–179 exist and all usable FLUID ground-truth centers are within the original video bounds. The valid source frame rate is passed unchanged to ByteTrack instead of silently declaring every video 30 FPS. Each decoded frame is checked against its verified native dimensions. The source is **never resized, cropped for full-frame reference coordinates, resampled, or rewritten**; SAHI uses the already-frozen 640px tile geometry and overlap on the original pixels. The core W04 decoder retains its strict 3840×2160 default.

If the source or annotation geometry is incompatible, the preflight error reports the actual width, height, frame count, FPS, and/or number of FLUID centers outside the native dimensions. **Do not automatically scale or repair the FLUID truth**. Native coordinate bounds alone do not establish genuine video/trajectory correspondence; that remains an independent visual-review requirement.

Because May 25 may differ from May 26 in native resolution, FPS, content, class distributions and sampled windows, the output remains descriptive cross-date regression evidence only, not a controlled site-generalization result. Measured detector latency is for the source's actual pixel dimensions.

## Scope of conclusions

A May 25 FLUID-label score can be compared **descriptively** with the May 26 tuned W04 metric. The two dates differ in scene composition and sampled window length (120 vs 201 frames), so the numeric difference is **not** a controlled same-window baseline delta, treatment effect, or calibrated generalization certificate. Correct-class precision/recall inherits known FLUID annotation disagreement. No original same-window T000 baseline or two-blind-reviewer physical-object adjudication is available; the sum of detector/tracker/postprocessing medians is not end-to-end p95.

To obtain *independent* validation afterwards, acquire genuinely untouched source footage and matching labels, establish chronology externally, keep the locked model/policy, compare on a properly matched baseline and separately measure p95/human-truth robustness. The independent Chennai generalization failure and synthetic-only native SUMO status remain unchanged.
