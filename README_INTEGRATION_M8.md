# StreetLab — Integration Phase 8 / 8: end-to-end release acceptance

**Implementation:** All eight *integration development phases* are implemented as separate, stacked, unmerged draft PRs. **Scientific and live-site acceptance remains BLOCKED** until the local Windows frozen-model run and authentic field records are supplied, verified and independently reviewed. This file is a runbook; it is **not** a production release certification.

## What Phase 8 actually delivered

1. `/release` release-readiness navigator and a **Release readiness** link on the project homepage. The browser displays only preliminary source/project states; it **does not** perform privileged or expensive release verification.
2. `streetlab_integration.release_cli`: read-only acceptance matrix covering full-media SHA-256, frozen W04 OpenVINO provenance and native observation manifest, reviewed M3 source-pixel/local-plane geometry, independently submitted M4 physical counts/holdout, recalculated M4 tripinfo quality, recomputed **all 18 M5 paired native SUMO tripinfo outputs**, and local original field-file reconciliation. It refuses to mark simulation outcomes as physical causal effects.
3. `streetlab_integration.field_acceptance`: checks raw field survey geometry, per-movement/class actual census rows, and per-trip independent holdout values against the exact stored SHA-pinned M4 inputs. Operator attestation binds the original files and original source-video digest; no sample field records are generated in production. This is **file consistency**, not proof that a human physically observed the road.
4. `scripts/STREETLAB_M8.ps1`: Windows launcher with `environment`, `web`, `worker`, `acceptance`, `audit`, and `backup` modes. Forces local-only binding and prevents an accidentally public Uvicorn listener; web and worker run in separate foreground terminals, with frozen model provenance required for the worker.
5. Full M1–M8 regression, real native SUMO synthetic integration smoke (both speed and signal interventions), and a Windows PowerShell parsing CI job. Synthetic fixtures can test correctness but cannot clear authentic release blockers.

## Acceptance states and why they are truthful

`PASS` means a local/software assertion passed; `NEEDS_DATA` requires authentic media/survey/run; `NEEDS_VERIFICATION` means local heavyweight CLI arguments were not supplied; `BLOCKED` means a prerequisite failed; `FAIL` means a contradiction or invalid artifact was found.

All software checks passing gives **`TECHNICAL_REPRODUCIBILITY_COMPLETE_OPERATOR_ATTESTED`** and still **`production_release_approved=false`**. It does NOT establish observer authentication, traffic-model accreditation, out-of-sample field behavior or a causal intervention effect. External review and authorized deployment approval remain separate organizational decisions. The CLI exits code 2 on incomplete/failed technical gates to prevent silent success in an operator automation.

## Local Windows launch (two separate terminals)

From an existing local checkout, first preserve your uncommitted work:

    cd C:\Users\Admin\Desktop\StreetLab-engine-trial
    git status --short
    git fetch origin
    git switch --track origin/codex/streetlab-integration-m8
    git pull --ff-only origin codex/streetlab-integration-m8

If that branch already exists locally, use `git switch codex/streetlab-integration-m8` instead of `--track`.

Check installed environment and installed SUMO runtime:

    .\scripts\STREETLAB_M8.ps1 -Mode environment

Terminal A: run localhost API and unified dashboard (http://127.0.0.1:8000, /reports, /release):

    .\scripts\STREETLAB_M8.ps1 -Mode web

Terminal B: point at the **actual frozen** W04 model folder and provenance file, *not a newly trained or guessed model*. This is a live foreground process which you can stop with Ctrl+C:

    .\scripts\STREETLAB_M8.ps1 -Mode worker -ModelDir 'C:\PATH\TO\W04_OPENVINO_MODEL' -FrozenProvenance 'C:\PATH\TO\W04_PROVENANCE.json'

Do not start a second worker on the same project workdir. Source video uploads and existing job state live in `.streetlab-m5/`; this is the default shared workdir. The model directory and provenance are **not present in the repository** and must be supplied from the actual Windows machine.

After an authentic project source is processed and the reviewed M3/M4/M5 workflows have completed (the M4 and M5 SUMO runs are independent CLI operations; no web handler executes SUMO), open /release and select that project's exact UUID.

## Physical field evidence required (NO synthetic substitution)

For a particular M4 revision, prepare one local folder holding exactly these four original observer-maintained records:

**`survey.json`:** JSON object with keys `project_id`, `baseline_revision`, `coordinate_system`, `center_world_m`, `arms`, `connections`, `control`. These values must match the **actual independently surveyed** geometry/lanes/control in the immutable M4 baseline `site_input.json`. Use original surveyed values—do not change them to make the model pass. The coordinate system is `LOCAL_GROUND_PLANE_METERS_NOT_GPS`.

**`census.csv`:** UTF-8 CSV with header:

    from_zone,to_zone,type_id,count

One nonnegative integer count per physical, manually observed permitted movement/class. The exact rows must reconcile to M4 `site_input.json` demand. Tracker IDs and tracking detections are **not** vehicle counts.

**`holdout.csv`:** UTF-8 CSV header:

    from_zone,to_zone,session_ref,travel_time_s

One row for *each physically measured trip*, not a single averaged number. Session ref must be the independently recorded holdout session and must differ from the field-census session. Per-movement sample number must match M4 `holdout[].samples` and arithmetic mean must agree within 0.02 seconds. **Do not create holdout records from SUMO outputs.**

**`attestation.json`:** JSON object with:

    {
      "schema_version": 1,
      "project_id": "<actual project UUID>",
      "baseline_revision": "<actual 64-digit M4 SHA>",
      "source_video_sha256": "<actual original video SHA>",
      "original_file_sha256": {
        "survey.json": "<SHA256 of real survey.json>",
        "census.csv": "<SHA256 of real census.csv>",
        "holdout.csv": "<SHA256 of real holdout.csv>"
      },
      "data_origin": "REAL_FIELD_RECORDS_OPERATOR_ATTESTED",
      "contains_synthetic_or_simulation_generated_observations": false,
      "holdout_collected_without_using_sumo_results": true,
      "real_world_intervention_effect_observed": false,
      "field_data_collector": "<specific dated collector and source record reference>",
      "independent_reviewer": "<distinct reviewer and dated source reference>",
      "survey_date_and_site_reference": "<road, time, survey location and dated evidence reference>",
      "review_record_reference": "<specific signed audit or original paper-record reference>"
    }

Placeholders are **not valid field evidence**. Do not set the boolean values falsely. The names and evidence references are plain text, not verified digital signatures; a malicious actor could falsify files and attestations. Independent reviewer and legal field-evaluation sign-off are still required outside StreetLab.

PowerShell can independently calculate an actual SHA with:

    (Get-FileHash 'C:\FIELD\census.csv' -Algorithm SHA256).Hash.ToLowerInvariant()

## Formal operator acceptance command

**This is a heavyweight local verification** and may hash multi-GB footage and parse all 18 SUMO tripinfo outputs. Choose a new unused output directory to keep every previous result intact.

    .\scripts\STREETLAB_M8.ps1 -Mode acceptance -Project '<ACTUAL_UUID>' -ModelDir 'C:\PATH\TO\W04_OPENVINO_MODEL' -FrozenProvenance 'C:\PATH\TO\W04_PROVENANCE.json' -FieldDir 'C:\PATH\TO\AUTHENTIC_FIELD_RECORDS' -VerifySourceSha -OutputDir '.\artifacts\m8_real_site_acceptance_01'

Output is `m8_acceptance.json` plus `m8_acceptance.md`, with each gate state, explanation, revision/hash evidence where available, blockers and explicit limitations. These files never get auto-uploaded to GitHub and never include the raw source video. Without actual local artifacts, you may still run this CLI without optional model/field arguments to see the **honest blocked state**. A missing source fails the release gate rather than synthesizing data.

## Final local audit and recovery

    .\scripts\STREETLAB_M8.ps1 -Mode audit
    .\scripts\STREETLAB_M8.ps1 -Mode backup

The M7 database backup contains SQLite metadata **only**. Also preserve videos under `.streetlab-m5/projects/`, immutable M2 runs, M3/M4/M5 evidence, frozen model/provenance, and original independent survey files using an approved filesystem backup. Restart the worker after restore only against the same hashed source/model. Audit the restored evidence before resuming. Keep project media private; do not commit it to GitHub.

## What CI can and cannot check

Dedicated `integration-m8.yml` tests Python/JS contract, all M1–M8 regression, actual native SUMO on a **synthetic fixture**, full M8 rescoring of all 18 native tripinfo files and PowerShell script parse on a Windows GitHub runner. It does **not** execute the user's pinned Windows W04 model on their actual footage, prove physical data or certify a public traffic deployment.

Public multitenant deployment is **out of scope**: default launcher forces loopback-only; prior M7 optional Basic mode still needs TLS/enterprise authentication, rate limiting, least privilege and privacy/legal review. No claim of production readiness until external site/independent evidence and deployment review are completed.