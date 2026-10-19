# May25 W04 research: autonomous completion — no more human-review loops

This phase replaces the repeated 14-crop review cycle with one reproducible **full 120-frame, all-objects analysis** of the original FLUID labels and frozen OpenVINO/ByteTrack tracking evidence. It neither asks for manual physical reviews nor makes the unsound claim that software can establish physical truth independently of source annotations.

## Run once when ready to inspect the entire research result

\`\`\`powershell
cd C:\Users\Admin\Desktop\StreetLab-final
git pull --ff-only origin main
.\scripts\STREETLAB_W04_AUTONOMOUS_RESEARCH.ps1
\`\`\`

The command automatically finds the latest original completed May25 run and its source-backed forensic evidence. It runs the automated regression tests, SHA-verifies **the original 505 MB MP4, FLUID CSV, raw detection batch, every saved tracking file and policy scorecard**, then replays the unchanged +1-frame/50px class-agnostic Hungarian correspondence over every one of the 120 scored frames. This can take time reading original video bytes, but it does **not** run SAHI/OpenVINO inference, modify a source CSV, select a new detector policy, or request further human reviews.

A newly created, non-overwriting \`artifacts/w04_autonomous_research_*\` folder contains:

- \`research_findings.md\`: human-readable final historical regression outcome, unique object/track distribution of class mistakes, matching ambiguity and scientific restrictions.
- \`automated_object_audit.json\`: complete numerical evidence, frozen source SHA fingerprints, all matched class diagnostics, error concentration, predicted class persistence, unique BUS track identities and explicit no-holdout/no-production flags.
- \`all_truth_track_diagnostics.csv\`: per-original-FLUID-object summary across the **whole** cohort, separating unmatched frames, class disagreement frames, dominant model category, distinct predicted IDs and competing spatial assignments.
- \`predicted_track_class_stability.csv\`: model-side track identity class persistence, multiple-FLUID-track associations and BUS model identities.

The audit fails closed if any source SHA, original 120-frame cohort, frozen detector policy, expected confusion matrix, or recorded experimental class counts differs from original evidence. That is deliberate; a changing label CSV must **not** silently create a more flattering result.

## What this can settle programmatically

- **Unique-object versus repeated-frame errors.** The existing May25 diagnostic counted 236 matched HEAVY_VEHICLE → CAR **frame-level** disagreements. They need not represent 236 unique vehicles. This workflow computes the true number of FLUID track IDs behind those 236 and how much is attributable to the top two or five objects.
- **Track stability.** Distinguishes consistent CAR classification of one persistent FLUID HEAVY track from model categories changing frame by frame. Track stability is not physical-class accuracy.
- **Matching ambiguity.** Counts disagreements with another eligible spatial association no more than 10px farther away within the unchanged 50px gate, without reoptimizing model thresholds or changing results.
- **BUS evidence limitation.** Separates 185 BUS *frame detections* into actual **distinct predicted track IDs**, records which had a spatial association with any FLUID class, and never turns zero BUS ground truth into measured bus precision or recall.
- **Cohort-wide object counts.** Uses all 120 original frames, not only 18 selected review frames or 14 cropped images.

## What it cannot legitimately settle

FLUID labels cannot independently establish whether a yellow vehicle is a passenger car, commercial goods vehicle, truck or bus. An automated model repeatedly saying CAR does not prove the source annotation HEAVY is wrong; an open-bed goods vehicle is not necessarily a passenger BUS. The user requested no external physical reviews, so the completed deliverable **retains physical taxonomy uncertainty** instead of fabricating a two-person consensus or model improvement.

May25 is a previously inspected **historical regression reference**, May26 W04 is **tuned development**, their 120/201-frame F1 difference is **unpaired/descriptive**, and both may contain source annotation disagreements. Native SUMO is synthetic-only. The Chennai independent holdout transfer failed. No same-window original T000, independently unseen true vehicle labels, external field calibration, or end-to-end CPU p95 exists. **Production remains blocked.** The original W04 model and policy stay frozen until a distinct, genuinely independent evaluation can support a new experiment.

No reviewer forms are necessary to complete this bounded engineering/research analysis. The missing field-quality evidence remains a documented limit, not another task loop.

## Optional read-only web integration

The existing `/research` page includes a May25 full-cohort result panel. The page does **not** compute results, fetch private Drive evidence, or fabricate unique-object counts. When the original-source autonomous runner has successfully created `automated_object_audit.json`, set `STREETLAB_MAY25_AUDIT_PATH` to the absolute path of that file **before starting the local web server**. The endpoint `/api/research/may25-audit` then loads only the completed source-fingerprinted artifact and performs fail-closed object-count and source-method checks. Missing or inconsistent files return HTTP 409 and the page shows *Not verified*. This does not replace the runner's complete source SHA verification.

The preview counts are annotated FLUID track identities and repeated matching events, **not independently confirmed physical vehicles**. BUS recall remains unmeasurable, the Chennai transfer remains failed, the SUMO demonstration remains synthetic, and production release remains blocked.
