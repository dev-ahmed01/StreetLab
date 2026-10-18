# May 25 physical-object review: label-blinded independent comparison

The supplied labeled May 25 contact sheet is a useful *engineering* diagnostic, but it visibly shows both **FLUID HEAVY_VEHICLE** and **Model CAR** before review. Therefore it must **not** be given to independent blinded reviewers as if it were an unbiased truth task.

The 14 user-visible examples are **selected, error-enriched, repeated-track** observations from a historical regression sample, not representative independent heldout data. In particular, R001–R006 repeat two FLUID track identities, and BUS samples contain uncertain/missing annotation. An object may also be an auto-rickshaw, light commercial vehicle, delivery truck, passenger car, bus, motorcycle or nonvehicle artifact; these should not be forced into FLUID's four-class ontology *before* physical inspection.

## Create independent blind packets (no new detector inference)

Run after \`git pull --ff-only origin main\`:

\`\`\`powershell
cd C:\Users\Admin\Desktop\StreetLab-final
.\scripts\STREETLAB_W04_BLIND_REVIEW.ps1
\`\`\`

The script finds the completed original May25 cross-date batch, source-bound forensics and existing track-diverse image review packet. It rechecks the original source video/FLUID SHA hashes, review sample correspondence and earlier image hashes, then **re-decodes the original video** to make NEW crops without the FLUID/model category, event, frame, or track IDs. Only **one black/white neutral ring** indicates the object to be identified. The crops are scaled for display only, never fed back into scoring.

This creates \`artifacts/w04_may25_blind_review_*/\` containing:

- **\`reviewer_1/\`** and **\`reviewer_2/\`**: separate but visually equivalent label-blinded contact sheets, individual numbered \`B001...\` images, instructions and an initially empty \`independent_reviewer_blank.csv\`. Share **only** the respective reviewer subfolder with each person.
- \`sealed_reference.json\`: preserves randomized sample-to-FLUID/model identities and original frame numbers. **DO NOT send this file to reviewers.**
- \`blind_review_receipt.json\`: source hash and sample provenance, explicitly \`independent_holdout_pass=false\` and \`production_release_approved=false\`.

**The user or anyone who saw the labeled contact sheet can still contribute an engineering review, but cannot truthfully claim that review was blinded.** For a genuine blind exercise ask two different people who have not seen the labeled images or research predictions. They should work separately without sharing observations.

Classify the object *inside the neutral ring* with one of these more descriptive physical types (rather than presupposing the four-class FLUID taxonomy): \`passenger_car\`, \`light_commercial_vehicle\`, \`heavy_goods_vehicle\`, \`passenger_bus\`, \`two_wheeler\`, \`three_wheeler\`, \`bicycle\`, \`pedestrian\`, \`fixed_object\`, \`not_visible\`, \`uncertain\`. Record whether the circle actually marks the object, occlusion, confidence and any uncertainty. Two different reviewer codes and a self-attested \`reviewed_without_FLUID_or_model_labels=yes\` are required. Their identity and blindness **cannot be externally proven by CSV files**, so independent adjudication remains a separate scientific obligation.

## Compare returned review sheets

After two independent reviewers have filled every sample row and returned their completed forms:

\`\`\`powershell
.\scripts\STREETLAB_W04_BLIND_REVIEW.ps1 -Mode Summarize -Reviewer1 'C:\PATH\reviewer_1_completed.csv' -Reviewer2 'C:\PATH\reviewer_2_completed.csv'
\`\`\`

The script checks unchanged sealed reference hashes, forms, reviewer identity claims, complete sample coverage, allowed values and truthfully declared blinding. It then exports a new \`artifacts/w04_may25_two_reviewer_consensus_*.json\` showing agreements, disagreements and aggregated *track groups*, not inflated frame-level counts. It never edits source labels, model weights or historical scores, and never upgrades a previously used May 25 regression into blind holdout validation.

A true HEAVY→CAR detector error, FLUID taxonomy mismatch, missing BUS annotation, or correspondence problem can **only be provisionally diagnosed** from physically reviewed objects, and visual reference selection itself is biased. No population-wide class recall can be inferred from the selected 14 crops. F1 remains the saved historical FLUID-label F1 (May25 91.42%; W04 development 88.97% on different windows). Full independent footage and adjudicated labels, baseline comparisons and CPU p95 remain pending; Chennai heldout generalization failed and native SUMO is synthetic-only.
