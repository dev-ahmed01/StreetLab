# StreetLab W04 Phase — blinded source-video adjudication

**Status:** Two-reviewer visual-review tooling implemented as experimental tooling on draft PR #10. It is NOT production code, not a replacement for original FLUID ground truth, and does not modify the authoritative IoS .30 ByteTrack output.

## Why this stage exists

The real unified hybrid experiment showed the rare IoU .50 stream improves heavy-class recall but damages primary tracker continuity. The separate evidence lane identified 191 unmatched rare-class observations across 40 tracklets; **seven candidate IDs** (five HEAVY_VEHICLE and two MOTORCYCLE) passed the original persistence/confidence screen. All 115 reviewed-tracklet observations had same-class raw detector support but five tracklets carried competing-class raw detections, so none is automatically a newly missed physical vehicle.

Two independent human reviewers should inspect original **source video pixels** before any object-level claim. Crops are actual source-video frames, not synthetic visualizations and never a replacement for whole-frame context. The original zero-based source frame index differs from FLUID annotated frames by +1.

## Implemented end to end

`scripts/phase3_w04_visual_review_pack.py` uses the already executed original 25-policy continuous batch directory, the existing unified hybrid replay directory and the original 4K `20250526_video.mp4`. It verifies Build 4 original batch evidence and video/FLUID source hashes; all 25 original policy tracking exports; four new hybrid track+sidecar hashes; original-vs-replay control byte-for-byte identity; the 21,104 cached raw detector boxes; and the **exact 191 unmatched / 40 shadow tracklets / 7 review targets cohort**. Any drift is a hard failure.

It deterministically samples up to five real frames per candidate from earliest, latest, middle, high-confidence and class-ambiguous observations, fills additional gaps with genuine track frames, and uses a single sequential video decode to produce a scene view, an unclipped context crop with target outline, and a neutral original-primary-tracker overlay for each case. The original MP4 is opened **read-only**. No newly invented camera images, track IDs, truth labels, model training or 4K OpenVINO inference is involved.

Reviewers receive two **separate** standalone offline HTML ZIP packets. Each includes `index.html` and JPEG evidence assets, but excludes original detector class, confidence, hybrid tracker ID, source manifest and other reviewer's responses. Reviewers independently mark visible YES/NO/UNCLEAR; observed image class CAR, BUS, HEAVY_VEHICLE, MOTORCYCLE, AUTO_RICKSHAW, BICYCLE, PEDESTRIAN, OTHER, UNKNOWN or NA; and whether the object is UNTRACKED, ALREADY_TRACKED, DUPLICATE_SHADOW, UNCLEAR or NA. They add notes and download their reviewer-specific CSV from the local browser. Do not place both reviewers together or exchange partial judgments.

`scripts/phase3_w04_review_consensus.py` validates complete **exactly seven** independent R01 and R02 CSV entries, authenticates the internal case manifest, rejects duplicates, swapped reviewer IDs, malformed fields and inconsistent classes, and produces explicit per-case agreement or review-needed states with immutable hashes. Even two independent human reviewers who agree about a possible untracked vehicle do **not** automatically rewrite FLUID truth or production counts.

## One-time Windows packet preparation

From `C:\Users\Admin\Desktop\StreetLab-engine-trial`:

```powershell
git switch codex/phase3-engine-shootout
git pull --ff-only origin codex/phase3-engine-shootout
& '.\scripts\RUN_W04_VISUAL_REVIEW.ps1'
```

The script expects the source video at `C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4`, and the previously executed folders:

- `artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01`
- `artifacts\phase3\sahi_detector_trials\W04_unified_hybrid_replay_01`

It will produce a new immutable folder `artifacts\phase3\sahi_detector_trials\W04_VISUAL_TWO_REVIEWER_KIT_01`, with `REVIEWER_R01_ONLY.zip`, `REVIEWER_R02_ONLY.zip`, `INTERNAL_case_manifest.json`, review templates, assets and SHA manifest. Unzip and distribute only one reviewer ZIP to each reviewer. **Never distribute the internal manifest** with the reviewer packet, because it reveals original class hypotheses. The video hash check can take time for a large source MP4; this is file I/O, not new inference.

## After both independent CSVs have been returned

```powershell
& '.\scripts\RUN_W04_VISUAL_REVIEW.ps1' -Mode Consensus -Reviewer01 'C:\path\W04_R01_blind_review.csv' -Reviewer02 'C:\path\W04_R02_blind_review.csv'
```

Artifacts: `artifacts\phase3\sahi_detector_trials\W04_VISUAL_CONSENSUS_01\case_level_consensus.csv`, `requires_adjudication.csv`, `consensus_summary.json` and `SHA256SUMS.txt`. Unclear or discordant cases require third-party adjudication. A possible missed physical object is still a W04-development human hypothesis, not validated on an independent video.

## Safety and data handling

- Original primary tracker export SHA remains `6c695e384e05421ec25dbe450f00a83fb6a72a659a2d162a1122d67c7d0ce418`. Frozen video provenance SHA remains `57105b564ff3f9c68d88e6df05790654a49a48f626b1adaf39b62ea61c262ce0`.
- Source imagery potentially contains identifiable people, vehicles and other personal information. Keep generated review packs local and send only to authorized reviewers; don't publish source crops in GitHub PR or external services.
- Independent physical review is **outstanding** until both CSVs are completed. This workflow makes no promotion decision and is not an unseen holdout.
- Separate release gates remain generalization on new footage, balanced class-level physical proof, replacement Geo-trax comparison if required and measured end-to-end CPU p95.

## Test coverage

`tests/test_phase3_w04_visual_review_pack.py` exercises representative-frame selection, anonymous blind IDs, static HTML reviewer ZIP isolation, image crop and original-primary overlay on synthetic frames, strict reviewer completeness, ID-swapping failure, independent agreement and uncertainty escalation. The normal Phase 3 GitHub CI sparse checkout includes both new scripts. CI does not and cannot validate the original 4K video without a local machine run.