# StreetLab — Phase 3 W04 prospective candidate intake (no video inference)

Use this after the **25/25 W04 policy score replay** and the **SHA-bound policy lock**. The intake is an inexpensive, read-only prerequisite for a new evaluation. It **never** claims that an existing source was truly unseen merely because the filename or SHA-256 differs.

## No new footage yet: produce a truthful blocker receipt

Run after pulling \`main\`, from \`C:\Users\Admin\Desktop\StreetLab-final\`:

\`\`\`powershell
git pull --ff-only origin main
.\scripts\STREETLAB_W04_HOLDOUT_INTAKE.ps1
\`\`\`

The launcher finds the newest local \`artifacts/w04_policy_lock_*\` directory and verifies that \`policy_lock.json\` SHA-256 matches \`holdout_protocol.json\`. It refuses altered or promoted policy evidence. Without candidate sources it creates a fresh \`artifacts/w04_holdout_intake_*.json\` with status \`W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH\`.

## If you already possess a distinct, legitimately usable source video + matching original labels

\`\`\`powershell
$originalLockDir = 'C:\Users\Admin\Desktop\StreetLab-final\artifacts\w04_policy_lock_YYYYMMDD_HHMMSS_mmm'
.\scripts\STREETLAB_W04_HOLDOUT_INTAKE.ps1 -ProtocolFolder $originalLockDir -CandidateVideo 'C:\EXISTING\candidate.mp4' -CandidateTruth 'C:\EXISTING\candidate_Traj.csv'
\`\`\`

If original same-window Geo-trax/T000 track bytes are available, add \`-BaselineTracks 'C:\EXISTING\original_t000.txt' -BaselineKind original_t000\`. If you generate a new comparison instead, use \`-BaselineKind new_replacement\` and do **not** describe those tracks as recovered original T000. The intake only hashes that file; it cannot independently establish its origin or scoring window.

**Possible statuses:**

- \`W04_CANDIDATE_REJECTED_REUSED_DEVELOPMENT_SOURCE\`: the MP4 or FLUID SHA is identical to tuned May 26 W04.
- \`W04_HISTORICAL_DATASET_ROLE_REQUIRES_EXTERNAL_REVIEW\`: filename corresponds to a known FLUID study with a previous declared role. It may have been examined before.
- \`W04_DISTINCT_BYTES_INDEPENDENCE_UNVERIFIED\`: both candidate files exist and have different SHA-256s, but a distinct hash **cannot** prove untouched provenance.
- \`W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH\`: there is no complete, accessible candidate video+truth pair.

The original manifest labels May 26 \`held_out_validation\`, but the W04 video was used for tuning, so its current experiment window is development-only. May 25 is a **regression reference** and Jan 20 is a **development-calibration** source; neither is automatically suitable as blind holdout. Distinct camera origin and no prior inspection must be substantiated by independent records. Original source files, trajectory truth, and model weights remain on the local machine; do not upload private footage to GitHub.

## After intake, not before

An external reviewer must establish the source's independence and allowed use. Then pin its camera and original source SHA, matching ground-truth SHA, frozen W04 model tree and policy, start/end absolute video frames, warmup, identical scoring rules, original/replacement same-window baseline and processing environment **before** opening performance results. Run the existing experimental continuous tracking path, validate with Phase 3 Build 4, collect two independent physical-object reviews, and measure true end-to-end CPU p95. Source integrity or reported FLUID precision alone cannot justify production deployment.

The frozen Chennai real-world holdout remains failed. Native SUMO remains an explicitly synthetic junction demonstration. No new site inspection is required for this *limited academic research-prototype* intake, but none of these steps substitute for field validation.
