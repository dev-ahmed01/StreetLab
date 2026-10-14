# StreetLab Phase 3 — Build 2/4 Independent Physical-Object Review

## Scope and evidence

This build is a **manual-review protocol and case-level comparison tool**, not a labeled dataset or corrected FLUID ground truth. It preserves the original T000/P3B and May26 W04 class-aware scores. All reviewer-facing case images are blinded: original raw crop left; target anchor ring right. Predicted classes and FLUID types are hidden. The 15 cases were deliberately selected during prior error investigation, so neither physical precision nor exhaustive recall can be inferred.

## Generate 15-case review pack (can run on user PC or container)

```powershell
$python = ".\.venv-sahi-audit\Scripts\python.exe"
& $python scripts/phase3_manual_review.py prepare `
  --bundle "artifacts/phase3/sahi_detector_trials/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip" `
  --output-dir "artifacts/phase3/sahi_detector_trials/W04_blind_manual_review_15_01"
```

Or use the ready-to-review ZIP built from the previously uploaded W04 source bundle; do not generate the pack again unless you need local reproducibility.

## Two independent reviewers

1. Unzip review pack locally. Reviewer A and reviewer B must **not** consult the model predictions or FLUID labels, discuss disputed items, or copy from each other.
2. Open `reviewer_ui.html` in a local browser. It loads `images/case_*.jpg` without network requests.
3. Enter a distinct reviewer ID and inspect **each** image. Set `PRESENT` only when a physical object exists at the ring; draw a box around the object on the *left* crop; select one visual class. Use `UNKNOWN` if the object is present but its class is visually unclear; it will remain unresolved. Select `ABSENT` when no object is at the target; `UNCERTAIN` for ambiguity/occlusion.
4. Export each independent CSV. Do not change case IDs. Alternatively, edit `reviewer_A_TEMPLATE.csv` and `reviewer_B_TEMPLATE.csv` manually with `PRESENT`, `ABSENT`, or `UNCERTAIN`. For `PRESENT`, `x1,y1,x2,y2` are measured on the left (480px) crop, not absolute 4K-frame pixels.
5. The two reviewers' CSVs must be stored separately. Unreviewed templates will be rejected.

## Generate consensus (after real people have reviewed)

```powershell
& $python scripts/phase3_manual_review.py adjudicate `
  --review-dir "artifacts/phase3/sahi_detector_trials/W04_blind_manual_review_15_01" `
  --review-a "path/to/real_reviewer_A.csv" `
  --review-b "path/to/real_reviewer_B.csv" `
  --output "artifacts/phase3/sahi_detector_trials/W04_manual_dual_consensus_01.json"
```

Only two independent decisions agreeing on PRESENT with the **same visible class** and bbox IoU >= .50, or both agreeing ABSENT, produce agreed cases. All conflicting, UNKNOWN, and UNCERTAIN cases remain UNRESOLVED. No automatic or model-derived human labels.

## Compare with Build 1 raw box candidates (after actual inference and review)

```powershell
& $python scripts/phase3_manual_review.py compare `
  --review-dir "artifacts/phase3/sahi_detector_trials/W04_blind_manual_review_15_01" `
  --consensus "artifacts/phase3/sahi_detector_trials/W04_manual_dual_consensus_01.json" `
  --box-lab-dir "artifacts/phase3/sahi_detector_trials/W04_premerge_box_lab_01" `
  --output "artifacts/phase3/sahi_detector_trials/W04_reviewed_case_box_comparison_01.json"
```

The comparator reads Build 1's immutable SHA-checksummed 25 candidate CSV box files. It records whether any output box center is inside an agreed physical-object box and whether the predicted class matches the visually reviewed class. `AGREED_ABSENT` cases are checked for candidate centers within 15px of the anchor. This does **not** measure whole-frame recall, physical precision, ID-switches, throughput, or production readiness. The Build 1 OpenVINO capture has not been measured in this container.

### Integrity / restrictions

- SHA-256 of source bundle manifest, case index, frame JPEG bytes, blinded gallery JPG, individual reviewer sheets, candidate boxes, and linked matrix reports.
- Immutable output paths; no overwrite of pre-existing review artifacts.
- All candidate rules are geometry-only, no FLUID labels in suppression or review decisions.
- W04 is tuned source footage; require unseen full-frame annotation, real box inference and consecutive tracker evaluation for any promotion. PR remains draft.