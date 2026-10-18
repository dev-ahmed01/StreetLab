# StreetLab — private Drive-backed source portability

StreetLab's complete historical May25 original experiment is already archived in the user's private Google Drive folder under these **relative paths**:

\`\`\`text
Streetlab/
  StreetLabData/Video_1/
    20250525_video.mp4
    20250525_video_Traj.csv
  StreetLab-final/artifacts/
    w04_may25_crossdate_20261018_132633_271/
      cross_date_summary.json
      preflight.json
      batch/
        batch_report.json
        source_provenance.json
        hard_nms_ios_0.30.score.json
        hard_nms_ios_0.30.txt
        original_pre_global_merge_boxes.jsonl
    w04_may25_forensics_20261018_141138_963/
      regression_forensics.json
      review_events.csv
\`\`\`

The original video is ~505MB; the FLUID CSV ~41.5MB. These are **PRIVATE** source assets, **never commit them into GitHub**. The original frozen source SHA-256 values are recorded in the saved provenance and checked **against original file bytes**. The source in the experiment's Windows provenance is retained untouched. File relocation is not permission to reinterpret labels or change any field.

## Runtime-independent operation

The runnable Python CLI is:

\`\`\`bash
python -m streetlab_integration.w04_autonomous_diagnostics \
  --run-dir /path/to/StreetLab-final/artifacts/w04_may25_crossdate_20261018_132633_271 \
  --forensics-dir /path/to/StreetLab-final/artifacts/w04_may25_forensics_20261018_141138_963 \
  --source-video /path/to/StreetLabData/Video_1/20250525_video.mp4 \
  --source-truth /path/to/StreetLabData/Video_1/20250525_video_Traj.csv \
  --output-dir /path/to/new_unique_results
\`\`\`

The equivalent PowerShell runner accepts \`-SourceVideo\` and \`-SourceTruth\`. When both omitted, original Windows paths continue working unchanged.

All necessary input files can be materialized with a connected Google Drive/Files tool to any writable private computing environment. The code does not require a user to reupload binaries into ChatGPT, but executing against the private source still requires an **operational compute environment** and all listed files staged in it. Merely browsing Drive metadata is not an execution, and GitHub's synthetic-fixture CI cannot access private OAuth Drive content automatically.

### Safety contract and explicit limitations

- No model inference, no source-video transforms, no manual review packets, no edits to existing FLUID, original tracking files or model.
- **Original video, FLUID and every saved tracking file are cryptographically reverified**, including the unchanged \`original_pre_global_merge_boxes.jsonl\` checksum and original batch scorecard.
- Materialized paths may differ from frozen Windows paths, but immutable \`source_video_sha256\` / \`original_fluid_sha256\` must match original experiment provenance. Both relocated source paths must be provided together; a lone replacement path, symlink or mismatch fails.
- The original May25 120-frame result is historical regression evidence, not independently unseen footage. W04 policy remains \`hard_nms_ios_0.30\`; BUS lacks FLUID truth points; Chennai holdout transfer failed; SUMO conclusions remain synthetic-only; production remains blocked.
- Never claim the source-backed run has succeeded until the actual output's \`automated_object_audit.json\` exists and is consistent with its recorded input source hashes.

This file intentionally excludes the user's private Google Drive file IDs, OAuth tokens, or transferable share links; use the authenticated connector and original folder name when obtaining bytes.
