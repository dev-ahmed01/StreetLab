# StreetLab — Phase 9: Original Fieldwork & Real-Site Pilot Preparation

**Scope:** Post-integration pilot readiness. The M1–M8 stack remains unchanged. M9 gives field observers an actionable collection plan and blank record templates for a real site. It does NOT claim that a single real road has been surveyed, that frozen W04 inference was executed on the user's laptop, or that any simulated benefit was observed.

## Delivered

- `/fieldwork`: focused original-observation collection page. Select an M2/M3 project and see only its independently reviewed approach/exit movements. The page never invents field measurements and never uploads original field data.
- `GET /api/projects/{id}/fieldwork-plan`: verified M3 project/geometry/source SHA bindings and which physical movements need actual observation. Before source processing or reviewed M3, returns `NEEDS_REVIEWED_GEOMETRY` rather than a bogus collection plan.
- `streetlab_integration.fieldwork --init`: creates a new, non-overwriting, SHA-manifested directory containing `FIELD_COLLECTION_GUIDE.md`, `plan.json`, `survey.template.json`, `census.template.csv`, `holdout.template.csv` and `attestation.template.json` with **blank physical measurements**. These files cannot pass M8 original-record verification in their template state.
- `--hashes ORIGINAL_DIRECTORY`: computes SHA-256 of the three bounded originals so actual observers can complete their own attestation. It does NOT set the attestation's truth flags, signatures, reviewer identity, or write on the operator's behalf.
- `--check ORIGINAL_DIRECTORY`: validates authentic, independently prepared originals against exactly the current M4 baseline using existing M8 `verify_field_records`. It categorizes missing data, missing M4, stale geometry, contradictory records, or bounded file consistency. Even a reconciled set remains `FILES_RECONCILED_OPERATOR_ATTESTED_ONLY` and `release_approved=false`.
- `scripts/STREETLAB_M9.ps1`: Windows launcher for plan/kit/hashes/check, preserving original files and refusing output overwrite.
- Full M1–M9 regression + genuine SUMO synthetic contract test + M8 native result recheck + Windows PowerShell parser. No actual-site field evidence is available to GitHub CI.

## How to run on the original Windows development machine

Before switching, preserve any uncommitted work. Earlier research holdout files, `.streetlab-m5` local DB/media, the source footage, W04 model and frozen provenance are not committed to GitHub.

    cd C:\Users\Admin\Desktop\StreetLab-engine-trial
    git status --short
    git fetch origin
    git switch --track origin/codex/streetlab-integration-m9

If it already exists locally: `git switch codex/streetlab-integration-m9` and `git pull --ff-only`.

Start the same M8 app/worker on 127.0.0.1 (in separate PowerShell terminals); do not open the local M7 guard to the public Internet:

    .\scripts\STREETLAB_M8.ps1 -Mode web
    .\scripts\STREETLAB_M8.ps1 -Mode worker -ModelDir 'C:\REAL\FROZEN\W04_MODEL' -FrozenProvenance 'C:\REAL\FROZEN\source_provenance.json'

Open **http://127.0.0.1:8000/fieldwork**. Select the actual project whose video successfully processed in M2 and whose geometry was measured and checked in M3; use the project's actual UUID, not a placeholder.

Get the current M3 source-specific required observations:

    .\scripts\STREETLAB_M9.ps1 -Mode plan -Project 'REAL_PROJECT_UUID'

Generate only blank original collection paperwork into a completely new directory:

    .\scripts\STREETLAB_M9.ps1 -Mode kit -Project 'REAL_PROJECT_UUID' -OutputDir '.\artifacts\site01_fieldkit'

**IMPORTANT:** These `.template.` files have no measured numerical contents; copying them to original file names does not constitute observation. Bring these reference sheets to the field, collect real source records, and retain independent observer notes with times and permissions. Do not use tracker counts or SUMO-generated trip times.

Once original physical observations have been recorded in separate `survey.json`, `census.csv`, `holdout.csv`, and `attestation.json`, create M4 from the same actual observations at **/baseline**. Do not reverse-fit the field data to match the simulation. Keep original physical records safe in a different directory from the blank kit.

Compute the original-file SHA values to help a real collector/reviewer complete their own M8 attestation:

    .\scripts\STREETLAB_M9.ps1 -Mode hashes -Project 'REAL_PROJECT_UUID' -FieldDir 'C:\REAL\ORIGINAL_FIELD_RECORDS'

Then reconcile the originals to the independently created M4 baseline:

    .\scripts\STREETLAB_M9.ps1 -Mode check -Project 'REAL_PROJECT_UUID' -FieldDir 'C:\REAL\ORIGINAL_FIELD_RECORDS'

A passed M9 check is only consistency of original files and operator attestation. M8 formal acceptance still requires the original media SHA, frozen W04 model provenance, site geometry, actual M4 and M5 native simulations, and external review:

    .\scripts\STREETLAB_M8.ps1 -Mode acceptance -Project 'REAL_PROJECT_UUID' -ModelDir 'C:\REAL\FROZEN\W04_MODEL' -FrozenProvenance 'C:\REAL\FROZEN\source_provenance.json' -FieldDir 'C:\REAL\ORIGINAL_FIELD_RECORDS' -VerifySourceSha -OutputDir '.\artifacts\m8_actual_site_acceptance_NEW'

See `README_INTEGRATION_M8.md` for full original-file schema and the production approval boundary.

## Security, evidence and implementation rules

1. Blank forms must never contain apparent observed speeds, counts, travel times or human identities. Only real SHA-bound M3 metadata is populated.
2. GitHub/test fixtures are synthetic *software contracts*, never physical proof. Even an operator-attested file may be fraudulent; M9 cannot verify observer identities or a genuine real-world intervention.
3. M8 and M9 never sign off actual deployment. Software PASS != scientific evidence; M5 simulated deltas != observed effects.
4. Do not alter frozen `CORRESPONDENCE_LOCK_V1.json`, W04 holdout, research thresholds, source videos, historical immutable project revisions, or existing baseline/scenario outputs.
5. Treat all media and field records as private. M7 localhost guard and HTTP Basic are not multitenant RBAC or deployment-grade security.
6. All M9 CLI operations are bounded file reads or non-overwriting form creation; no native SUMO job or video model is invoked by browser/field kit.

## Follow-on work required for the final product

**Operator-dependent (cannot be completed by GitHub alone):** authenticated access to actual Windows machine, frozen W04 OpenVINO/runtime dependencies, real source video inference, field-ground surveyed geometry and lane movements, manually verified distinct physical census, separate trip-level travel-time holdout and true observer review. Then run actual SUMO in M4/M5, M8 acceptance, and a formal human review of external site validity.

**Independent engineering work:** use M9 CI failures, exact local Windows startup results, and recorded pilot acceptance blockers to harden the runtime; ensure original survey input is never auto-filled, implement reviewed UI refinements, document deployment/privacy/legal controls and conduct user acceptance testing. Do not claim a completed final product until these gates are actually cleared.

**PR scope:** Stack M9 as a draft on `codex/streetlab-integration-m8`. Prior PRs remain open/unmerged until explicit review and ordered merges.