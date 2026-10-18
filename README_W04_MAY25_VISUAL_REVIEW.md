# May 25 source-backed visual review packet — physical-object investigation before tuning

After the May25 **real-video** FLUID-label regression and source-backed forensic replay, the candidate error queue contains **182 frame-level events** across 18 selected source frames. Its 36 selected HEAVY_VEHICLE → CAR disagreements involve just **two recurring FLUID track IDs, 7 and 36**, matched to predicted CAR tracks 26 and 29, repeated across 18 frames each. Of the 39 selected unmatched predicted BUS records, 30 come from predicted track IDs 97 and 110. This suggests an **object-persistent class/taxonomy review**, not a blind postprocessing parameter sweep. The full 120-frame cohort contains 236 spatially matched HEAVY→CAR disagreements; the 18-frame selection does **not** establish how many unique physical vehicles caused all 236.

The source scorer records **91.42% overall May25 F1** and 60.89% HEAVY_VEHICLE correct-class recall (436/716 FLUID truth points). The May26 W04 F1 is **88.97% on a different 201-frame development sample**. These differences remain descriptive and previously exposed regression evidence, not an independent heldout test.

## Execute once on Windows — no OpenVINO inference

\`\`\`powershell
cd C:\Users\Admin\Desktop\StreetLab-final
git pull --ff-only origin main
.\scripts\STREETLAB_W04_MAY25_REVIEW_PACKET.ps1
\`\`\`

The launcher uses the latest complete original \`artifacts/w04_may25_crossdate_*\` batch and \`artifacts/w04_may25_forensics_*\` report. You can pass exact paths as \`-RunDir\` and \`-ForensicsDir\` if you have more than one research copy.

The generator hashes and checks the **original May25 MP4 and FLUID trajectory CSV**, crosschecks prior forensics against saved batch/summary/preflight checksums and original source provenance, and verifies \`review_events.csv\` is byte-equivalent in events to the audit JSON. It does not load the frozen model or run inference and does not modify the original video, annotations, tracks, policy or scoring.

It decodes only the original May25 source frames needed for its report, preserving exact source frame indices (FLUID index = video frame + 1). Original 4K source pixels are **cropped/resized for display only**; this is not part of model scoring and does not alter FLUID coordinate scales. A reviewer-friendly contact sheet has separate green FLUID label coordinates and orange prediction coordinates, with distinct class/ID labels and explicit **not physically adjudicated** warnings.

## Output

The script creates a unique \`artifacts/w04_may25_review_packet_*\` folder:

- \`review_contact_sheet.jpg\`: one sheet of up to **14 distinct evidence samples**.
- \`crops/R001.jpg\` … individual context crops around original FLUID/predicted centers.
- \`reviewer_1_blank.csv\` and \`reviewer_2_blank.csv\`: identical unfilled forms for **independent** physical identification; original predictions and FLUID class labels are hints, **not** adjudicated truth.
- \`review_packet_manifest.json\`: original-video/FLUID hashes, saved research source correspondence, timestamp, exact source frame and IDs per sample, cropped-image checksums, and explicit \`independent_holdout_pass=false\` / \`production_release_approved=false\`.

The sample selection is **grouped by event and tracked identities**, not by number of consecutive error rows. It prefers **three temporally spread samples** for each of the two recurring HEAVY class-confusion pairs, plus unmatched HEAVY tracks, possible BUS label issues and distinct MOTORCYCLE misses. This sharply reduces repeated review burden but is **biased by design**, so the reviewed set must never be converted to representative field-wide class accuracy.

## How to use the evidence

Open the contact sheet first, or upload the \`review_contact_sheet.jpg\` (and optionally the relevant individual crops) for additional image-level diagnosis. Establish whether FLUID “heavy” corresponds to an actual physically heavy vehicle, whether predicted BUS items are true missing annotations, and whether motorcycle misses are occlusion/tiny-object failures.

Only after **two independent reviewers** examine the original frames and reconcile disagreements should the project classify persistent errors as physical label inconsistencies versus genuine detector errors. Reviewers should not edit or overwrite the original FLUID trajectory CSV; store adjudication in a separate review layer, by individual \`sample_id\` and original frame/track ID.

If the video is private/sensitive, keep crops local and share only files you are authorized to distribute.

**Remaining scientific blocks:** genuinely unseen video, independent human truth, original same-window T000 or a transparently created replacement baseline, real end-to-end CPU p95 measurement, and unchanged failed Chennai generalization. Native SUMO remains synthetic. This stage is research-only, never production validation.
