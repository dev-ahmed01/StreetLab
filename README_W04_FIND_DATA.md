# StreetLab W04 — local source and FLUID annotation discovery

This step responds to the legitimate result \`W04_HOLDOUT_NEEDS_SOURCE_AND_TRUTH\`. Instead of recording the same blocker repeatedly, inventory the **video files and FLUID trajectory CSVs already stored locally**, without rerunning the 4K model or altering original data.

## Quick Windows command

From the already updated \`C:\Users\Admin\Desktop\StreetLab-final\` checkout:

\`\`\`powershell
git pull --ff-only origin main
.\scripts\STREETLAB_W04_FIND_DATA.ps1
\`\`\`

The default dataset root is the sibling \`C:\Users\Admin\Desktop\StreetLabData\`. It automatically finds the newest existing \`artifacts\w04_policy_lock_*\` directory and verifies that the lock matches the protocol fingerprint before examining candidates. It creates a **fresh**, non-overwriting \`artifacts\w04_dataset_inventory_*.json\`.

If FLUID annotations or videos are in other existing folders, scan their *specific* parent directories together:

\`\`\`powershell
.\scripts\STREETLAB_W04_FIND_DATA.ps1 -Roots @(
    'C:\Users\Admin\Desktop\StreetLabData',
    'C:\Users\Admin\Desktop\FLUID'
)
\`\`\`

Use only directories that exist on your PC; these paths are examples except \`StreetLabData\`, which is the previously used project data location.

## What the JSON means

- \`candidate_video_count\`: discovered \`.mp4\`, \`.avi\`, \`.mov\` and \`.mkv\` files, not verified image/video decodes.
- \`candidate_fluid_label_csv_count\`: CSVs with header columns \`frame,id,cx,cy,type\` (case-insensitive). The inventory **does not** read the whole annotation file or certify correctness.
- \`video_count_with_possible_truth\`: videos with matching manifest filenames, shared video basename, or shared eight-digit date. A shared date is particularly ambiguous; **no row correspondence or camera alignment is verified**.
- Each video carries the prior study role where known: May 26 = tuned W04 development, May 25 = regression reference, Jan 20 = development calibration. None is automatically eligible as an untouched holdout.
- \`incompatible_csv_files\` reports other CSVs with nonmatching schemas, including telemetry/signal files. \`unpaired_fluid_truth_files\` records trajectory labels that were not paired by filenames.

Safety and performance: scan is bounded to four nested directory levels and 20,000 entries by default, skips \`.git\`, caches, \`node_modules\`, models and \`artifacts\`, never follows symbolic links, and fails if the cap is reached rather than silently claiming comprehensive discovery. It does **not** compute SHA-256 for every giant video, decode frames, run inference, produce accuracy metrics, access network services, or claim any candidate is independent. To inspect deeper intentionally, provide \`-MaxDepth\` with a value up to 12 or narrow your roots.

## How to proceed

After inspecting the inventory, identify a video with matching **original** annotations whose prior-use history can be independently established. Then supply those *exact* paths to:

\`\`\`powershell
.\scripts\STREETLAB_W04_HOLDOUT_INTAKE.ps1 -CandidateVideo 'C:\REAL\candidate.mp4' -CandidateTruth 'C:\REAL\candidate_Traj.csv'
\`\`\`

That intake hashes the exact selected sources but still does not prove they were unseen. A genuine research holdout further needs externally verifiable chronology/camera independence, frozen policy settings, same-window T000/reference, objective class review by two people, long-run stability and actual end-to-end CPU p95. Previously examined May 25/Jan 20 data must be labeled honestly as regression/development rather than being retroactively promoted.

**Scientific boundaries remain unchanged:** the W04 reported 88.97% F1 is a tuned **FLUID-label development** number, Chennai independent generalization remains failed, native SUMO junction demonstration is synthetic, and neither audit nor simulation confers field deployment approval.
