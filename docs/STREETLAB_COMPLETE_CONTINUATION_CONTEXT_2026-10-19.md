# STREETLAB — COMPLETE CONTINUATION AND HANDOFF CONTEXT
**As of:** 19 October 2026 (India/IST)  
**Project:** StreetLab — evidence-first video traffic observation, track reconstruction, traffic simulation, and reproducible research prototype  
**Repository:** https://github.com/dev-ahmed01/StreetLab  
**Single persistent data workspace:** https://drive.google.com/drive/folders/12L3N-a2VUYojuT3cXICkpK9X6K5KsEIQ  
**Repository main at handoff verification:** 2936a95950e68dfae2eccbac278bdf4014b3bc42 — PR #37, Drive-portable original-source audit.  
**Purpose of this file:** A new ChatGPT conversation should take ownership of the project and CONTINUE DIRECTLY from the exact unfinished task, using connected Google Drive and GitHub, without restarting discovery, making the user shuttle files between chats, or creating another endless sequence of manual reviews.

---

## 0. READ THIS FIRST — HOW THE USER WANTS THE WORK DONE

The user's repeated instruction, in substance, is:

> I do not want to be stuck in a loop uploading files, running one script, copying errors, and approving one tiny phase at a time. My files are all in Google Drive so you can work with them. Research, implement, test, fix, integrate and finish the whole deliverable yourself as far as your actual tools allow. Take the time necessary. Come back with a genuinely finished result for me to check. Do not make me perform physical/manual vehicle reviews.

**Operating rules for the next assistant:**

1. **Own implementation, not just prompting.** Use the connected GitHub integration to inspect the actual repository, create branches and PRs, run/inspect GitHub CI, fix failures and merge verified changes. Use Drive for original project data, source evidence, and results. No Codex prompt handoff unless the user explicitly asks for one.
2. **Stop the upload loop.** The exact Drive folder above is the user's chosen persistent source. Search/browse that folder and reuse files already there. Do not ask the user to upload video, annotation CSV, source provenance, previous scorecards, or cropped images already present and accessible.
3. **Stop physical-review loops.** No further requests for reviewer_1/reviewer_2 sheets, manual bounding-box reviews, one screenshot at a time, or repeated qualitative "is this a truck?" prompts. The user explicitly rejected these. Keep physical-label uncertainty as a documented scientific limitation rather than repeatedly treating it as the next blocking task.
4. **Work in meaningful batches.** Implement, verify, and iterate on a coherent deliverable; avoid "one PR per tiny error" or multiple nearly identical scripts that force the user back to the terminal. Bundle necessary fixes with tests and one clear final execution path.
5. **Do not fabricate results or claim access.** GitHub CI with synthetic fixtures is NOT the same as running the real 505 MB May25 MP4. The connected Drive source is real; execution is only "complete" if it actually ran and the real source hashes, exact scores, and outputs were verified. The ChatGPT sandbox/container recently returned ServerError/ClientError intermittently. Check before relying on it; if down, report it accurately and use other available safe paths.
6. **Freeze research evidence.** No changes to original MP4, FLUID trajectory CSV, original saved raw boxes or tracks, selected W04 policy/model, or held-out failed evidence to obtain better numbers. Generate new versioned/non-overwriting outputs only.
7. **Finite roadmap, no moving goalposts.** The user asks how many phases. Use the FIXED SIX DELIVERABLE PHASES in section 9. When a phase is complete, advance within the same work session when feasible. Don't invent new manual-review phases.
8. **Status must be meaningful.** Distinguish implemented/merged/CI-green from actually run on original Drive data, and research-demo readiness from field-production approval. Include exact remaining blockers, not vague percentages.
9. **Do not unnecessarily ask for decisions.** The user's priorities and architecture are established. Ask only when a genuinely essential permission, access, or unresolvable scientific input is missing.
10. **Favor a working finished application and academic artifact.** The user ultimately wants a polished, runnable, understandable StreetLab research prototype—not only increasingly elaborate audit reports.

### FIRST RESPONSE IN A NEW CHAT

After reading this file, do not merely re-summarize or ask "what next?" Verify GitHub main and the Drive source, recover the analysis environment if possible, then continue the current outstanding full-cohort analysis. Tell the user the fixed six-phase scope and what is realistically remaining. Make actual progress before returning with a result.

---

## 1. HIGH-LEVEL VISION: WHAT STREETLAB IS

StreetLab is a **local-first evidence-first traffic observation and decision-support/research platform** for fixed-camera road traffic studies. It is NOT just YOLO visualization and not a certified engineering design or production traffic-control product.

The intended scientific chain:

- **Observe** — ingest source traffic footage; run frozen, source-provenanced OpenVINO/SAHI detector + ByteTrack; preserve source pixel coordinates, vehicle class hypotheses, identities, counts and observed trajectories.
- **Reconstruct** — define geometry and lane/entry/exit polygons with independent actual ground-plane measurements. Pixel coordinates are NOT GPS or ground-truth meters; raw tracker IDs are NOT automatically verified unique physical vehicle counts.
- **Validate SUMO** — construct genuine native SUMO simulation on a grounded junction network; real-site claims require independent vehicle census and observed travel time/trip holdout. Synthetic junction demonstration is a separate restricted research path.
- **Compare** — controlled traffic-planning/signal/detour simulation scenarios with reproducible experiments and comparisons; modeled improvements must not be presented as observed field causal gains.
- **Report/Release** — SHA-bound evidence exports, explicit release acceptance gates, local web UI for work stages, reproducible research package and known limitations.

The implementation includes M1–M9 application integration and a local web server with project workflows; see the current repository README. Typical server start from project root:

    .\scripts\STREETLAB_M8.ps1 -Mode web -Port 8001

Open localhost only. The research submission runner is:

    .\scripts\STREETLAB_RESEARCH.ps1

The user successfully ran this on Windows AFTER a corrected package/module import fix; it ran **actual native SUMO on a synthetic/assumed junction scenario**, generating immutable reports with **Chennai real-site holdout still failed** and no production approval. These are separate evidence channels from May25/May26 FLUID video.

### Product philosophy

A credible controlled research demonstration is acceptable provided its provenance and limits remain true. An attractive "working application" must not claim to have independently verified field results, never misrepresent stored-video replay as live inference, and must not hide failed scientific gates.

---

## 2. RESEARCH, DATA, AND SCIENTIFIC STATUS

### W04 development source — May 26, 2025

- Source: Video_2 / 20250526_video.mp4, around **1.70 GB**, original FLUID trajectories around **143.6 MB**.
- A frozen W04 OpenVINO model and detector output provenance exists under the former StreetLab-engine-trial artifacts, also represented in Drive.
- Selected policy **hard_nms_ios_0.30**, frozen after extensive exploration.
- Continuous development sequence: 291 source frames total including 90 warmup; 201 scored frames **10750–10950**.
- Original development metrics: approximate FLUID-label spatial F1 **88.97%**, class diagnostics and **71 contiguous ID switches** over that longer window. Reported detector CPU median around **1.245 seconds/frame** (not real-time throughput or end-to-end p95).
- Earlier development class-level values included motorcycle correct-class recall ~83.32%, heavy correct-class recall ~79.75%; zero BUS FLUID truth points in that specific W04 window.
- Previous experiment broad sweeps: roughly 25 policy candidates, raw premerge box/provenance archives, class stabilization and geometry hypotheses. Many candidate explorations made May26 W04 a **tuned DEVELOPMENT cohort**, even if an older dataset manifest calls it "held_out_validation." Do NOT retroactively describe it as truly unseen.
- The model/video/FLUID fingerprints were verified by a previous read-only local readiness audit (W04_LOCAL_DEVELOPMENT_ASSETS_VERIFIED). Independent holdout pass remains false.

### Historical cross-date regression — May 25, 2025

- Source: Video_1 / 20250525_video.mp4, approximately **504.6 MB** (sometimes rounded to 505 MB); FLUID original trajectory CSV ~**41.5 MB**.
- Previously used historical **regression reference**, not untouched independent holdout.
- Native original May25 footage: **3840x2160 at 10 FPS**. The first runner incorrectly assumed 30 FPS; that bug was fixed.
- Frozen May26 W04 model and policy **hard_nms_ios_0.30** were run against original May25 video+its own FLUID file.
- Actual local completed May25 experiment: 30 warmup frames (source 30–59), 120 evaluated source frames **60–179**, FLUID alignment **+1** (truth frames 61–180); **150 total decoded frames**.
- Correct saved real-video FLUID-label outcomes:

| Metric | May25 original real-video result |
|---|---:|
| Spatial precision | **90.38%** |
| Spatial recall | **92.49%** |
| Spatial F1 | **91.42%** |
| Contiguous ID switches | **21** |
| CAR correct-class recall | **95.99%** on 4,391 CAR truth points |
| MOTORCYCLE correct-class recall | **82.44%** on 2,785 MOTORCYCLE truth points |
| HEAVY_VEHICLE correct-class recall | **60.89%** on 716 HEAVY truth points |
| BUS correct-class accuracy | Not measurable — **0 BUS FLUID truth points** |
| Median detector stage | About **1.400 sec/source frame** |
| Source FPS | **10** |

- Of 716 HEAVY FLUID truth points: **672 spatially associated**, of which **436** predicted HEAVY and **236** associated with **CAR** predictions; remaining **44** were spatially unmatched. These are **frame-level label-relative disagreements**, NOT necessarily 236 different physical vehicles.
- Around **185 predicted BUS frame records** occurred with zero BUS FLUID truth points; one predicted BUS associated spatially with a non-BUS FLUID label and 184 were unmatched. Their physical accuracy remains unknown; zero BUS truth does NOT certify that all unmatched BUS predictions are false physical buses.
- May25 minus May26 development F1 ≈ **+2.45 percentage points**, but periods (120 vs 201 frames), traffic density, dates and class mixes differ. This is **unpaired descriptive difference**, NOT scientific proof of generalization improvement.
- Stage timing medians cannot establish real end-to-end CPU **p95**.

### May25 forensic/review history

- Automated forensic replay verified original source hashes, original model/policy/provenance and re-scored tracks against FLUID (+1 frame/50px class-agnostic Hungarian).
- Saved **regression_forensics.json**, **review_events.csv**.
- Reviewed candidate queue had **182 frame-level events** on 18 selected source frames. In that SELECTED subset, **36 HEAVY→CAR errors involved just FLUID truth track IDs 7 and 36**, corresponding to CAR prediction tracks 26 and 29. This does **NOT** establish how many distinct FLUID objects account for ALL 236 full-cohort disagreements.
- Similarly 30 of 39 selected unmatched BUS entries repeated predicted BUS tracks 97 and 110.
- A 14-crop contact sheet was created (8 distinct source frames). Looking at it showed plausible light utility/car/heavy-vehicle taxonomy disagreements and ambiguous BUS labeling. The user then explicitly said **STOP ALL PHYSICAL REVIEW LOOPS**.
- PR #35 implemented blinding/two-reviewer forms before that instruction; **do not continue requesting these forms**. The forms are optional historical artifacts only, not part of the requested completion route.
- PR #36 implemented full-cohort autonomous track-level diagnostics, designed to answer object uniqueness, persistent label mismatches, model-side class consistency, spatial matching ambiguity and distinct predicted BUS track counts across ALL 120 frames. It is code-complete/CI-passed, but original-source execution of those new diagnostic outputs has NOT been confirmed.

### Independent site/production boundaries

- An unrelated real-site **Chennai** source/validation holdout FAILED. Do not change its outcome, transfer claims, or source limits.
- Native SUMO was actually executed on synthetic/assumed junction scenes. It is NOT a calibrated field causal-effect result.
- Original same-window T000 baseline is unavailable; do not represent a freshly created replacement as recovered original T000.
- Independent physically adjudicated labels, independent unseen video source, field calibration and end-to-end p95 remain unavailable/unverified.
- A finished and demonstrable **RESEARCH PROTOTYPE** is still possible. Scientific/production claims must remain bounded.

---

## 3. GITHUB: EXACT IMPLEMENTATION/PR HISTORY TO REUSE

Repository: https://github.com/dev-ahmed01/StreetLab

Major recent merged steps:

- PR #23 — research report exporter, figures/CSV/Markdown and genuine native SUMO synthetic scenario experiments.
- PR #24 — fixed Windows research runner import/package behavior without manual PYTHONPATH.
- PR #25 — source/model/truth readiness audit, read-only SHA integrity.
- PR #28 — frozen policy preregistration/holdout source intake that cannot claim blind data is available when it is not.
- PR #29 — offline directory inventory for real video/FLUID candidate pairs.
- PR #30 — May25 real-video cross-date regression runner.
- PR #31 — skip blank FLUID cx/cy consistently with frozen scorer, without editing CSV.
- PR #32 — accept May25 native video FPS/resolution instead of hardcoded W04 30 FPS, enforce no resampling/false label scales.
- PR #33 — source-verified forensic replay and objective class mismatch review queue.
- PR #34 — representative original-frame visual packet.
- PR #35 — neutral-ring blind review/reviewer templates; DO NOT reopen manual review dependency.
- PR #36 — complete full-cohort automated object-level diagnostic code, tests, one-run script and research report templates, NO reviewer requirement.
- **PR #37** — relocated original Google Drive video/FLUID source support while enforcing original SHA hashes and frozen evidence. **Latest verified main commit: 2936a95950e68dfae2eccbac278bdf4014b3bc42**. Reported all 15 CI workflows passed.

Important current implementation files:

- streetlab_integration/w04_autonomous_diagnostics.py
- streetlab_integration/w04_crossdate_regression.py
- streetlab_integration/w04_crossdate_forensics.py
- streetlab_integration/w04_visual_review_packet.py
- streetlab_phase3/pixel_benchmark.py
- streetlab_phase3/video/class_diagnostics.py
- streetlab_phase3/geotrax_pixel.py
- streetlab_phase3/video/holdout_validation.py
- streetlab_phase3/class_mapping.py
- scripts/STREETLAB_W04_AUTONOMOUS_RESEARCH.ps1
- tests/test_w04_autonomous_diagnostics.py
- README_W04_AUTONOMOUS_RESEARCH.md
- README.md
- .github/workflows/research-submission.yml

Latest Windows command from PR #37, ONLY as contingency if no remote execution path works:

    cd C:\Users\Admin\Desktop\StreetLab-final
    git pull --ff-only origin main
    .\scripts\STREETLAB_W04_AUTONOMOUS_RESEARCH.ps1

It also accepts BOTH -SourceVideo and -SourceTruth to point to relocated, Drive-materialized original files, without changing frozen Windows source provenance, and accepts the original -RunDir and -ForensicsDir. The next assistant should prefer DOING THIS WORK itself with Drive + a working tool runtime; don't reflexively bounce it back to the user.

GitHub CI tests synthetic fixtures and code regression/integration; it cannot establish that the new full-cohort May25 audit has actually processed the original 505MB recording.

---

## 4. GOOGLE DRIVE — PERSISTENT SOURCE OF TRUTH

**User's root folder:**  
https://drive.google.com/drive/folders/12L3N-a2VUYojuT3cXICkpK9X6K5KsEIQ

The Google Drive connector is connected, and the assistant successfully accessed the folder through it. This was the user's explicit strategy to end repeated manual uploads and downloading local artifacts.

Top-level children and Drive folder IDs:

| Folder | Drive folder ID |
|---|---|
| StreetLabData | 1CHa-wnWoM8xAM7To-74-ZuOyYOgmIQjk |
| StreetLab-main | 1vFsfmaoksiBYkFUDcopagj7pcsERX_we |
| StreetLab-final | 1cmSrHbB15jgvPfBcYNJFJnfX4qqDyiVV |
| StreetLab-engine-trial | 1rCDWoSEk4B_D3mkD8980QELlR83QoSP7 |

Data folders:

- Video_1 (May25): ID **1JPHpvHm97IobZfRa-1WFnNqq7u2fwX4e**.
- Video_2 (May26): ID **1Uq3gKBb6jkLkSCvuQQiCf2F3asslRJlG**.

Important observed Google Drive **file IDs** (verify metadata/title before processing):

| File or artifact | Drive file ID |
|---|---|
| May25 original video 20250525_video.mp4 | 1ndrZqMBAMKgN5FqNRgiw5cAfNG90pnFR |
| May25 original FLUID 20250525_video_Traj.csv | 1TZwZGLHHw_-nQbF2VwkSpXJVPCT1TZKm |
| May25 saved tracked-point text (~727KB) | 1zr324It7K-CG1_Fh8KPyJScLDVHv7l-0 |
| May25 raw capture boxes (~3.25MB) | 1H0r1a5Py48ooZaunPbBnrEv1fMPixum0 |
| May25 cross_date_summary.json | 1qhyt_KQvRnSI7DhGX65A79ySnTDiuejf |
| May25 preflight.json | 1eM3PgGoTn3qTk3mtZqy5P57BsdVXP790 |
| May25 batch_report.json | 14gLrHDv1CKRxBoUK5hz31z-1IJVwUo_M |
| May25 saved source_provenance.json | 1q-0FZee2qyhegOotvwkA7C9KXauqaC2B |
| Frozen W04 policy_lock.json | 1LeQA_HkKb6NtPxJrjdGWj7rSNrbsVP5n |
| Frozen holdout_protocol.json | 1hkOd0TWDL8b7VCBbwxg-Zh0cHCLgNS9F |

Important subfolders:

- May25 completed run: **1k-Yc9iGaYo81xd9C59QiYimnuMODqI15**
- May25 source-backed forensics: **1sF7sJuwo6-r5kYsK9l6ki4_GJ26OvN7D**
- Earlier May25 forensics copy: **1bxcjFpBwASeA8_pXsmysUUXLSAk19wKf**
- May25 image packet: **1LNNG0ULNthZY6reDgpZLk19q4hcRy-3_**
- W04 policy lock folder: **1bR033ZvVj8QYrPn77PSXL7aOK1ExsiDB**
- Engine-trial Phase3: **1AbypNSeY7BzbRVkW26bcvdczz-xHSe86**
- Main engine artifacts: **1Im0EcoS1WJEZ5WqJECRaJ4tXe22W07o9**

**Known connector nuance:** Google Drive search with a parent-child filter sometimes returned folder names but NO files inside large folders. Directly opening those folder URLs via the Google Drive fetch connector (or the mounted Files Library at /Google Drive/Streetlab/...) returned actual files, including the 504.6MB May25 video, 41.5MB CSV and 143.6MB May26 truth. Do NOT infer an empty folder from one empty search call. Prefer listing/opening the known exact folder, then materialize exact original files.

**Mount nuance:** The Files Library showed an external Google Drive mount under /Google Drive and a project folder /Google Drive/Streetlab (capitalization differs from user-shared URL). Files Library materialization of the original FLUID CSV, saved tracking text, raw boxes and video was attempted; small originals were successfully materialized in previous turns. The container then began intermittently failing with ServerError/ClientError, so **do not assume those temporary /mnt/data paths survive the new conversation**. Rematerialize exact needed files from Drive once a working environment is available. The Drive folder/file IDs persist and do not need reupload.

**Original-source integrity:** Use the hashes inside the May25 preflight/source_provenance/frozen protocol. They, not historical Windows absolute paths, define source identity. PR #37 enables read-only relocation from Drive while enforcing exact original bytes. Never guess a missing SHA or bypass the lock check.

---

## 5. EXACT STOPPING POINT OF THIS CHAT

The immediate question in progress at the user's latest request was:

> Now that Drive provides the original May25 data, finish the FULL-COHORT automated object attribution and deliver its real outputs, without reviewer workflows or further user file-upload loops.

Earlier assistant:
1. Verified current StreetLab GitHub main PR #37 code and source paths.
2. Used Google Drive connector to confirm original video/truth/saved model-tracking artifacts.
3. Materialized some original data into a temporary environment, then observed Python/container tools intermittently failing (ServerError/ClientError).
4. Switched to the Files Library connector to read portions of original May25 FLUID (reported 222,818 source CSV lines) and model-tracking file (reported 8,076 lines), not just the short 182-event review queue.
5. Inspected the exact frozen scorer in streetlab_phase3/pixel_benchmark.py: class-agnostic Hungarian one-to-one within each aligned frame, **+1 frame offset, 50px gate**, and then class correctness assessed after matching.
6. Inspected the class-mapping aliases; e.g., FLUID moped maps to MOTORCYCLE, truck maps to HEAVY_VEHICLE, passenger classes to CAR. Skip blank original center columns consistent with the existing benchmark normalizer.
7. Began an attempt to parse the entire FLUID/track data in buffered chunks using Files Library + in-tool JavaScript. The most recent large chunk-processing tool call **did not return a verified result in the recorded chat before the user requested this handoff**. DO NOT invent its output, unique-track counts, concentration or matched score.

**The programmatic full-cohort audit is CODED and GitHub-CI-tested, but a successfully produced, source-bound original May25 automated_object_audit.json and final research_findings.md is still the key missing actual deliverable.**

Do not start another May25 OpenVINO detector pass. The original track output and labels are already there. Replay the saved tracks/FLUID and verify the original scorecard, minimizing repeated reading of the large MP4 (SHA once if needed).

### The NEXT technical actions, in order

1. Verify the latest GitHub main and Drive folder/source evidence availability. Do not rediscover dozens of branches when this has been done; verify changes since 2936a95.
2. Recover container/Python access; when working, materialize original small May25 inputs via Drive/Files (CSV, 727KB tracks, original provenance, preflight, batch report, policy lock, forensic JSON). Materialize the May25 MP4 if SHA validation or permitted real frame inspection requires it. Process bytes rather than dumping 222,818 CSV lines into the chat.
3. Run (or fix then run) the genuine original-source, full-cohort w04_autonomous_diagnostics.py, using PR #37's portable source overrides. Ensure it reproduces source hashes, frozen +1/50px scores and original 236 heavy class-confusion frame matches.
4. Compute and VERIFY novel unique-object metrics: number of distinct May25 FLUID HEAVY track IDs underlying ALL 236 HEAVY→CAR frame points; top-two/top-five concentrations; track class persistence; unmatched/matched BUS records by distinct predicted IDs; Hungarian near-alternative associations; fragmentation and object-level issues.
5. Produce final source-verified output bundle: **automated_object_audit.json**, **all_truth_track_diagnostics.csv**, **predicted_track_class_stability.csv**, **research_findings.md**, and execution manifest with hashes/status. Verify artifacts exist and that counts conserve against the original scorecard; treat source mismatch as an error.
6. Then finish the research-prototype application/release verification with proper web UI, tests, native SUMO research demonstration and truthful reports (see next section); user should see a complete product-oriented delivery rather than more review packet prompts.

**If sandbox is still broken:** do not claim an original-video audit ran. Work through the connected tools and source-backed GitHub implementation. Consider a safe runnable workflow on an available compute host only if genuinely supported and permitted; GitHub hosted CI does not automatically have access to private Drive files. If no real execution environment exists, clearly identify this single blocker instead of returning to the same repeated upload/Powershell loop.

---

## 6. PREVIOUS LOOP AND ERRORS — DO NOT REPEAT

The project has repeatedly suffered avoidable serial experiments and one-error-at-a-time workflows:

- Earlier Windows SUMO research runner initially required manual PYTHONPATH. PR #24 corrected module execution. User verified it.
- May25 cross-date preflight originally threw on blank FLUID cx. PR #31 aligned it with existing normalizer. User verified progression.
- May25 preflight then threw because video did not satisfy hardcoded 4K/30 FPS assumption. PR #32 fixed video-native metadata/FPS and saved-frame decoder without label resampling; user then completed actual full May25 inference successfully.
- Several separate PRs implemented readiness, intake, discovery, preflight, regression, forensics, visual samples and blind reviewer forms. This was scientifically careful but operationally extremely repetitive. The user explicitly asked to stop physical reviews and to "build everything and check it out" before returning.
- More recently a 505MB video transfer caused the code-execution container to fail intermittently. The user set up Google Drive specifically to avoid repeated file uploads and local Windows back-and-forth. Do not keep requesting the same files.
- Avoid celebrating synthetic CI while neglecting the real original-source run and finished usable app.

**Principle:** Fix entire failure categories and verify source-to-output compatibility at once. Use objective numbers and reproducible tests. Never abandon previous approaches without evidence, but stop speculative hyperparameter exploration on already-tuned development footage.

---

## 7. WHAT THE FINISHED USER-FACING PRODUCT SHOULD BE

StreetLab should feel like a coherent, functional local research application, not disconnected scripts and scattered PRs.

### User-oriented minimal workflow

1. **Project/source selection:** see original recording and source provenance; choose Observe or preloaded historical research data.
2. **Traffic observation:** play video and review computed classes/paths/counts if genuine recorded tracks exist; clearly distinguish a prerecorded trace replay from live inference. Use source pixel matching and meaningful vehicle statistics, not explanatory text overload.
3. **Junction reconstruction:** a usable geometry and movement view with calibrated versus assumed inputs distinguished.
4. **SUMO experiment:** one native SUMO simulation runner and controlled comparison showing baseline vs alternatives as *modeled* differences.
5. **Research evidence & diagnostics:** authoritative report with metrics, class error concentration by unique tracked object, native source status, p95 unknown as appropriate, and reproducible export.
6. **Release/status:** clear scientific acceptance gates; honest failed Chennai transfer, missing unseen holdout, synthetic-only scenario origin, no production endorsement.

### UI/UX direction

- Calm, professional and uncluttered; minimal modules and straightforward navigation.
- Statistics should be numbers with brief labels, not walls of explanatory UI text.
- Two complementary fonts: **Poppins for body** and a coherent modern heading family selected by the implementation.
- Clean visual hierarchy, consistent spacing, simple back/forward behavior; avoid redundant dashboard widgets, overloaded menus and noisy repetitive descriptions.
- Original source video and traffic visualization should be prominent where useful.
- Do not conceal scientific failure flags in the data; simplify how they are presented to end users.

The codebase already has project web flows; audit and improve the existing app before inventing a second competing frontend.

---

## 8. SOURCE / METRIC QUALITY RULES

- Never identify FLUID track IDs as definitely unique real-world vehicles without qualification: within FLUID they are annotated track identities, and physical counts can be affected by fragmentation and annotation quality.
- Match class-agnostically in native pixel space using the locked +1 offset and 50px gate; class errors are diagnosed after spatial association.
- Zero BUS ground-truth points means BUS accuracy/recall unmeasurable, even with persistent predicted BUS objects.
- Source labels can disagree with visual taxonomy (light commercial vs CAR vs HEAVY); do not rewrite FLUID labels or automatically "correct" them based on model class consistency.
- Unique-track concentration from **ALL 120 frames** may differ dramatically from 18-frame shortlisted crops; never extrapolate only the two observed FLUID track IDs 7/36 to the entire cohort.
- Native stage medians ~1.4s/frame are not full end-to-end p95. Never claim real-time 30 FPS from those numbers.
- May25 91.42% and W04 May26 88.97% are label-relative, unpaired distinct cohorts.
- Never rebrand development cohort or historical regression as genuinely unseen independent holdout.
- Chennai generalization FAILED and native SUMO synthetic-only conditions must survive final documentation and UI reports.
- Synthetic CI success proves code behavior on test fixtures, not actual source-video performance.
- Only use actual verified SHA hashes and source paths. Never guess or fabricate metrics.

---

## 9. SIX FIXED DELIVERY PHASES — STOP ADDING REVIEW PHASES

These are **six delivery-level workstreams**, not the historical internal M1–M9 or W04 experimental numbering. New bugs are fixed INSIDE their respective phase rather than extending the roadmap indefinitely.

| Phase | Objective | State at handoff | Objective done condition |
|---|---|---|---|
| **1. Source & reproducibility foundation** | Original May25/May26 + W04 model/provenance, frozen tests, project/data hygiene | **COMPLETED for known original files**. Drive access and GitHub portable-source support merged. | Original inputs discoverable, SHA referenced, no upload loop. |
| **2. Frozen perception baseline & cross-date replay** | W04 201-frame development; real May25 fixed 120-frame regression, actual native geometry and timing | **COMPLETED as historical regression**. May25 genuine 91.42% saved. | Exact source/eval settings, immutable score/provenance; limitations stated. |
| **3. Autonomous full-cohort diagnostics** | Compute unique object/track error concentration, class stability, BUS ambiguity and output scientific report without human-review dependency | **CODE/TESTS COMPLETE; ORIGINAL-SOURCE EXECUTION NOT VERIFIED**. THIS IS THE IMMEDIATE ACTIVE PHASE. | Full source-hashed 120-frame metrics reproduced; saved JSON/CSVs/Markdown generated and verified. |
| **4. Scientific research integration** | Tie original real-source observations, frozen failures, genuine native SUMO synthetic scenario, and research reporting into one reproducible package | **PARTIALLY IMPLEMENTED**, native SUMO reports already run; final coherent all-source package pending. | A reproducible clear experiment/report pipeline; explicit distinctions among development, regression, synthetic simulations and failed Chennai. |
| **5. Finished application & UI end-to-end** | Verify and simplify the actual StreetLab web UX and source selection/playback/analysis/report journey | **BASE APP IMPLEMENTED M1–M9; FINAL FINISH/REAL DATA E2E NOT VERIFIED**. | Stable local app, polished minimal UI, usable real-source workflows, automated app/API/browser checks where supported. |
| **6. Release hardening & user handoff** | Repo cleanup, merged reviewed changes, CI, deployment/local run book, finished research deliverables, evidence-status honesty | **PENDING FINAL END-TO-END ACCEPTANCE**. | One coherent tested release/PR; all code checks passed; actual source runs truthfully reported; known scientific exclusions documented; easy start guide. |

**No seventh phase** simply because independent physical labels or field measurements are absent. If missing, state that this completed item is a **bounded research prototype**, not field-certified production. Any later genuine field deployment is a distinct future project requiring actual field evidence.

**Completion target:** Deliver all feasible working prototype features and automated research diagnostics; provide a FINAL, accurate list of externally unverified scientific capabilities. Do not put the user in another infinite loop.

---

## 10. MOST IMPORTANT FILES THE NEXT CHAT SHOULD READ FIRST

1. This handoff file.
2. https://github.com/dev-ahmed01/StreetLab and its current README.md and latest main commit.
3. streetlab_integration/w04_autonomous_diagnostics.py — currently code-ready but original Drive dataset results pending.
4. scripts/STREETLAB_W04_AUTONOMOUS_RESEARCH.ps1 — supports two relocated Drive source paths.
5. streetlab_phase3/pixel_benchmark.py — exact matching implementation, not an assumed matching algorithm.
6. streetlab_phase3/video/class_diagnostics.py — true-class vs predicted-class per-class metrics.
7. streetlab_integration/w04_crossdate_forensics.py — source-hashed earlier completed forensic replay.
8. May25 source-provenance, cross_date_summary.json and regression_forensics.json in the user's Drive folder.
9. All original FLUID/track data from the Drive source folder, not truncated conversational CSV snippets.
10. README_W04_AUTONOMOUS_RESEARCH.md and README_RESEARCH_SUBMISSION.md for scientific boundaries.

---

## 11. COPY-PASTE STARTER REQUEST FOR THE NEW CHAT

I am continuing my StreetLab project. Read the complete context file in this repository first. Repo: https://github.com/dev-ahmed01/StreetLab . My persistent Google Drive data and evidence folder is https://drive.google.com/drive/folders/12L3N-a2VUYojuT3cXICkpK9X6K5KsEIQ .

Do NOT restart the upload loop, demand repeated PowerShell outputs, send me one tiny fix per turn, ask me to perform manual/physical vehicle review, or give me a prompt for another coding agent. I want you to own implementation, analysis, tests, PRs, merges and eventual finished app/research deliverable, doing as much as your connected tools genuinely permit. You have explicit permission to proceed without step-by-step confirmation for ordinary repository work. Do not overclaim runs that did not occur.

We have a fixed SIX-PHASE delivery roadmap. Phases 1–2 are substantially completed. Phase 3 is the immediate unfinished original-source run: the May25 120-frame automated object-level audit from saved tracking CSV and FLUID labels, with source hash/provenance checks. PR #36 created the analyzer; PR #37 added Drive-portable original source paths. Previous actual May25 regression measured 90.38% precision, 92.49% recall, 91.42% F1 and 236 HEAVY→CAR frame-level confusion points. The unique-track distribution across the FULL cohort is NOT YET VERIFIED. Please recover execution using connected Drive, calculate the true whole-cohort metrics, output authoritative files, and then continue phases 4–6. Complete code and CI where possible, address blockers yourself, deliver a working polished research prototype, and preserve known scientific limits (Chennai failed, native SUMO synthetic-only, no unseen holdout/physical adjudication/production approval).

When you respond, give meaningful achieved results and the remaining bounded work. No more open-ended review loops.

---

## 12. FINAL TRUTHFUL STATUS SNAPSHOT

**The data is available in connected Drive. The main repository already has a sophisticated, CI-passing research/product implementation. The real May25 detector/tracker pass SUCCEEDED previously and produced credible historical FLUID-label scores. The FULL May25 unique-object automated forensic REPORT is coded but has NOT been verified as executed against the original Drive files as of this handoff. The execution runtime had intermittent failures; the last attempted bulk Drive text parsing did not yield a verified numerical result in the chat.**

Focus effort there first; then consolidate research, finish app E2E and release. Preserve all original evidence, stop manual reviewer gates, and give the user a FINISHED, verifiable deliverable rather than another chain of instructions.
