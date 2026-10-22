# StreetLab fixed completion roadmap — six phases, no repeated reviewer/upload loop

This release milestone produces a single objective acceptance verdict from the existing local/Drive-staged research receipts. It **never** treats technical tests or historical FLUID scores as independent traffic-site validation.

## Six fixed phases

1. **Source and provenance:** May25 source/video and FLUID SHA correspondence previously attested by source-backed forensics, along with non-overwriting preflight and score hashes.
2. **Frozen W04 policy:** original versioned `hard_nms_ios_0.30` lock and preregistration remain byte-aligned and unpromoted.
3. **May25 real regression:** 150 decoded original frames, 120 scored, historical reference only, and saved forensics score replay.
4. **Autonomous entire-cohort analysis:** run the already merged full 120-frame object attribution on the original source-backed tracking and FLUID files (not 14 selected crops). If its receipt does not exist, this phase stays **BLOCKED**. No subjective vehicle label repairs.
5. **Native SUMO and Chennai:** actual native SUMO has run **on a synthetic assumption-based network** and frozen Chennai independent generalization has **failed**. Both statements are preserved.
6. **Final research handoff:** complete only after phases 1–5 are verified; produces a ready-to-submit *research-prototype* evidence bundle, never a field or production release.

## One no-review status command

```powershell
cd C:\Users\Admin\Desktop\StreetLab-final
git pull --ff-only origin main
.\scripts\STREETLAB_W04_COMPLETION.ps1
```

The script automatically selects the most recent **hash-consistent chain** of May25 run, matching forensic replay, matching full-cohort audit and actually executed synthetic SUMO study, rather than mistakenly treating a later partial/failed folder as authoritative. It verifies frozen policy and SHA links between preregistration, original May25 run, reproduced forensics, full-cohort audit and native SUMO study, and writes unique non-overwriting `artifacts/w04_completion_*/completion_verdict.json` and `completion_verdict.md`.

This **does not require a further physical review or fresh upload**, and it does not run OpenVINO. With missing files, it writes exact **BLOCKED** phase statuses rather than silently passing. The saved JSON's `original_source_bytes_rehashed_in_this_report=false` flag distinguishes this completion receipt from the separate original source hash-verifying autonomous audit.

The existing [connected Google Drive workspace](https://drive.google.com/drive/folders/12L3N-a2VUYojuT3cXICkpK9X6K5KsEIQ) provides original source artifacts, but a GitHub CI runner cannot directly access private Google Drive OAuth files just because ChatGPT has a connector. Runtime execution of the May25 original full-cohort audit still requires those bytes materialized on an execution machine; do not claim a complete Phase 4 until that source-backed audit JSON is actually produced.

## Scientific release restriction

The project is a real-video observation and *synthetic SUMO* research prototype, not an operational digital twin. May25 is a historical regression reference, May26 W04 was tuned development, FLUID physical categories have not been independently adjudicated, Chennai held-out transfer has failed, and an authentic same-window original T000 and end-to-end CPU p95 are not yet proven. **Independent holdout=false; production_release_approved=false** even if all six research-bundle phases complete.
