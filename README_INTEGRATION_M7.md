# StreetLab — Integration Phase 7 / 8: Reporting, provenance, security and reliability

**Scope:** report/read-only export and local operational hardening on the M6 unified project dashboard. Previous M1–M6 receipts, W04 frozen IoS .30 model, independent field-survey requirements and Phase 2 synthetic Decision Lab remain separate and unmodified.

## Delivered capabilities

1. **Evidence report screen:** visit http://127.0.0.1:8000/reports or use Evidence report on the unified homepage. Choose a project and view native tracking provenance, geometric calibration state, baseline review state, paired simulation status, explicit scientific limitations and next action.
2. **Reproducible export:** per-project JSON and readable Markdown. A bounded ZIP includes selected small provenance receipts, stable `report.json`, readable `report.md`, and `SHA256SUMS.json` with SHA-256 for every file. Exported ZIPs for identical evidence are byte-for-byte reproducible. Independent offline verifier checks project binding, member allowlist, sizes and hashes without extracting. No raw camera footage, full tracking trajectories/rows, image overlays, trained model files, bulky SUMO output or arbitrary user paths.
3. **Local access boundary:** default `STREETLAB_SECURITY_MODE=local` only permits loopback peer IP and loopback HTTP Host (in-process TestClient exception). Site-wide security headers, no-store cache control, framing blocked. A remote HTTP request is rejected with 403 even if it knows the project UUID. This is a local single-user boundary, **not a multi-user account system**.
4. **Opt-in password protection:** set `STREETLAB_SECURITY_MODE=basic` and `STREETLAB_ACCESS_PASSWORD` (20–512 printable non-space characters), then restart. All routes and assets require HTTP Basic user `streetlab` and the password. Use **HTTPS via a trusted reverse proxy** for any non-loopback operation: HTTP Basic passwords are unsafe over unencrypted networks. Add separate network firewall, TLS termination, secure upstream proxy, rate limiting, user auth/SSO and least-privilege app execution for real production. This mode has one shared password; no per-user RBAC, device/session revocation or audit-log guarantees.
5. **Disaster-recovery metadata:** `streetlab_integration.maintenance` audits SQLite and all currently referenced immutable project receipts, writes online-consistent metadata-only SQLite backups with SHA256 receipts and verifies an existing backup. It never overwrites source video, touches W04 holdouts or silently restores older states.

## Reports and API

| Method | Route |
|---|---|
| GET | `/reports` — visual report and export interface |
| GET | `/api/projects/{id}/report` — source-linked JSON project report |
| GET | `/api/projects/{id}/report.md` — downloadable Markdown report |
| GET | `/api/projects/{id}/evidence.zip` — downloadable bounded metadata-only ZIP |

The report is generated from the currently verified source chain via M6; tampering, unavailable source or invalid original receipt aborts with HTTP 409. Cross-project UUIDs cannot be reused to retrieve a different project's report. Unlike a digitally signed and externally attested chain of custody, these checksums protect against accidental change, not a fully malicious administrator with filesystem write access.

## Local operator runbook — Windows PowerShell

Before switching branches preserve your uncommitted local work:

    cd C:\Users\Admin\Desktop\StreetLab-engine-trial
    git status --short
    git fetch origin
    git switch --track origin/codex/streetlab-integration-m7

If the M7 branch already exists locally, use `git switch codex/streetlab-integration-m7` instead, then `git pull --ff-only origin codex/streetlab-integration-m7`.

Default single-machine mode (recommended during field validation):

    $env:STREETLAB_SECURITY_MODE = 'local'
    & '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000

Browse http://127.0.0.1:8000/ and http://127.0.0.1:8000/reports. The local guard validates both peer connection and HTTP Host; **do not run the default mode behind an unauthenticated, host-rewriting reverse proxy**.

Protect all routes when the deployment intentionally needs authenticated access (configure **HTTPS at the reverse proxy**, never expose Basic directly on plain Internet HTTP):

    $env:STREETLAB_SECURITY_MODE = 'basic'
    $env:STREETLAB_ACCESS_PASSWORD = '<STRONG_RANDOM_20_PLUS_CHARACTER_SECRET>'
    & '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000

Replace the placeholder with an actual high-entropy secret stored outside source control. Browser prompts for username `streetlab` and password. Do not check the password into the GitHub repository. Protect server logs, reverse proxy logs and the underlying Windows user profile; do not include credentials in shared reports.

After exporting a project ZIP, verify it offline without extracting:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.evidence_reports 'C:\path\to\streetlab-project-evidence.zip' --project '<ACTUAL_PROJECT_UUID>'

Audit all existing project provenance receipts:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.maintenance --workdir '.streetlab-m5' --audit

Create a consistent **metadata-only** backup, with SQLite integrity check and SHA receipt:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.maintenance --workdir '.streetlab-m5' --backup

Verify a completed database backup using its name returned by --backup:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.maintenance --workdir '.streetlab-m5' --verify-backup 'streetlab-db-<ACTUAL_DATE_AND_SUFFIX>.sqlite3'

Important: SQLite backup and metadata ZIP are **not full site-data backups**. For disaster recovery additionally protect the original `.streetlab-m5/projects/` video sources, `runs/` native track results, immutable spatial/baseline/scenario directories, optional frozen OpenVINO model files and separate W04 research artifacts through an operator-approved filesystem backup. A restored SQLite file alone cannot re-create missing footage or verified source receipts.

## Operational and research boundaries

- The M7 report is not a generated claim of real travel-time or congestion savings: source-pixel tracker IDs are not physical counts; ground-plane calibration is not GPS; field census/holdouts are operator-supplied and need external validation.
- The SUMO baseline QA thresholds and nine paired synthetic/model stress conditions are modeling checks, not accredited scientific validation. No causal claim is authorized.
- Access control is a deliberately minimal single-operator boundary; reverse proxies can hide the true client address. If used remotely, configure Basic mode with HTTPS and upstream controls. Running with public binding alone is not an authentication strategy.
- ZIP members are allowlisted, SHA checked and each metadata member is capped at 2 MiB (whole export capped at 12 MiB); large artifacts are intentionally not exported. No arbitrary file paths from HTTP are accepted.
- SQLite online backups are consistent snapshots of metadata; no automatic backup scheduling, media snapshotting, offsite retention or multi-user authorization is implemented. These are explicit M8 deployment concerns.

## Verification and Phase 8

Dedicated `integration-m7.yml` tests the complete M1–M7 chain, deterministic offline ZIP validation, missing source and tamper refusal, password mode, loopback restriction, security headers, redacted audit, SQLite metadata backup and local UI API compatibility. Passing tests are software verification, not external field acceptance.

**Next: Integration Phase 8 / 8 — full real-source acceptance, operational deployment and release hardening.** Verify actual Windows worker with pinned model, independently surveyed lane geometry/OD counts and holdout, genuine SUMO baseline, repeatable scenario CLI, browser UX and accountable field study report. Do not mark the product production-ready until these gates are met.