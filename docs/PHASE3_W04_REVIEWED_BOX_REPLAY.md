# Phase 3 — Accepted W04 dual-review → cached detector replay → actual box matrix

**Development evidence only.** W04 is already tuned. The source FLUID dataset, original 21-frame OpenVINO/PyTorch score, T000, and production Geo-trax remain untouched. Never interpret case-level matching as physical precision or exhaustive recall.

## Verified real W04 cached replay (October 9, 2026)

User supplied two completed blind human-review CSVs. The SHA-checked original Build 2 adjudicator produced **13 agreed present** and **2 unresolved** out of 15 cases. OpenVINO and PyTorch *already-cached* sliced center-only detections, byte-verified against the uploaded W04 source archive, were compared with the 13 reviewer-drawn bounding boxes. A detection is counted in a case only if its saved **center** is inside the independent consensus box.

| Cases | OpenVINO | PyTorch |
|---|---:|---:|
| Any predicted center inside physical review box | 13/13 | 12/13 |
| At least one center with reviewers' exact physical class | 12/13 | 11/13 |
| Multiple model centers inside physical review box | 5/13 | 5/13 |

One reviewed AUTO_RICKSHAW is outside both backends' four vehicle classes, and both model caches detect it as CAR. OpenVINO alone covers W04 case 1 (frame 10990) inside the human-reviewed motorcycle box. The other two unresolved cases are not forced into either score. These numbers **cannot** be used to compute detector precision, frame-level recall, physical FPs or tracking ID switches.

## New single-command verified real OpenVINO 25-candidate box experiment

After the source bundle and validated dual-review consensus are saved locally, run one PowerShell command (from the StreetLab repo root and the existing compatible `.venv-sahi-audit`). All 21 W04 sampled frames, 25 predeclared box policies and 13 accepted independent-review comparisons run automatically without needing 25 separate invocations.

```powershell
cd C:\Users\Admin\Desktop\StreetLab-engine-trial
git pull --ff-only
$python = ".\.venv-sahi-audit\Scripts\python.exe"

& $python scripts/phase3_w04_integrated_review_run.py `
  --bundle "artifacts/phase3/sahi_detector_trials/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip" `
  --review-dir "artifacts/phase3/sahi_detector_trials/W04_BLIND_REVIEW_15_02" `
  --consensus "artifacts/phase3/sahi_detector_trials/W04_manual_dual_consensus_20261009_01.json" `
  --model-dir "artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01/checkpoint_openvino_model" `
  --video "C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4" `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_25policies_dualreview_realvideo_01"
```

**Local evidence placement is required**: the bundle, original SHA-verified blinded reviewer folder and official consensus JSON live only in the user's local workspace or ChatGPT uploads. They are intentionally NOT published in the GitHub repository. Locate the actual files before running; adjust paths only as needed. The `--video` option uses original pixel-exact decoded frames and is strongly preferred; omitting it uses bundled JPEGs and is explicitly marked as not pixel-identical.

**CRITICAL complete model:** Do NOT use the zipped `model/checkpoint.xml` and `.bin` as an OpenVINO directory. That archive omitted other model export metadata, so its reconstructed tree does **not** pass the original export SHA. Use the **original complete directory** recorded by the W04 detector source report. The runner and Build 1 both hash it and refuse mismatches before executing inference.

Outputs live only in a newly created directory and are atomically published on success:

- `cached_review.json`: real W04 cached OpenVINO/PyTorch centers against agreed human review boxes
- `box_lab/pre_global_merge_boxes.jsonl`: real per-tile post-local-model-NMS boxes (all supported model classes, absolute xyxy)
- `box_lab/candidate_boxes/*.csv`: all 25 alternative global merge policies
- `box_lab/matrix_report.json`: frozen FLUID class-aware 50px, +1 frame *development* score and CPU observations
- `reviewed_candidate_comparison.json`: 13 accepted human-reviewed case coverage for every candidate
- `one_batch_summary.json`: source/consensus hashes, run provenance and non-promotion status

No policy is selected automatically on W04. No original artifact is overwritten. An error leaves no partially published experiment directory.

For a deterministic no-inference repeat of the existing result, add `--cached-only` and a **different** unique `--output-dir`. This command needs only NumPy/OpenCV/Python libraries in the existing repo; it does not require OpenVINO, SAHI or tracking dependencies.

## What this does not establish

W04's 21 frames are separated by 30 source frames, so they do not support true track identity. Real 25-policy box capture is **not** the Build 3 continuous tracking validation: that requires its separate consecutive video run and true same-window T000. Physical review on the tuned 15 selected samples is neither unbiased precision nor a substitute for independent unseen footage. Holdout Build 4 remains fail-closed until those real sources and human reviews exist. Draft PR #10 must remain unmerged.