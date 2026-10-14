# StreetLab security and release boundary

## Model of trust

StreetLab is a **local-only, single-operator** desktop/server development application. Default `STREETLAB_SECURITY_MODE=local` rejects requests from non-loopback peers and non-loopback HTTP Host headers. The browser UI, original video, field sheets and SQLite metadata must stay on an authorized Windows machine.

An optional Basic mode uses a shared secret but **is not production IAM**. Never publish the server or run it behind an unauthenticated Cloudflare tunnel or public reverse proxy. For deployment beyond localhost, obtain a security design with TLS, per-user authentication/SSO, authorization, CSRF, rate limits, audit retention, privacy rules, data residency, secure backups and incident response.

## Observed repo audit findings (October 2026)

- Source branch contained **15,153 tracked .venv files** (~1.28 GB), plus **72 generated Python cache files**. Consolidation removed these from the CURRENT tree, not historical Git blobs.
- New `.gitignore` prevents committing virtualenvs, runtime/source footage, SQLite, raw operator records and future scientific output caches.
- Repository had **no configured main branch protection** at audit time. GitHub Settings → Branches / Rulesets should require passing CI, pull-request review, forbid force pushes/deletions, and restrict who can write main. Connector permissions here do not provide a supported tool to create or update branch protection.
- Root README was stale and implied Phase 1 only. Replaced with M1–M9 local product instructions and honest acceptance boundaries.
- The large `data/` directory (~336 MB in the audited tree) contains original historical Chennai trajectory sheets and many Phase 1 / SUMO research outputs; retained intentionally to avoid breaking reproducibility. Confirm legal rights/public redistribution and whether any fields contain identifying data. New generated outputs should remain untracked.
- Previous Git revisions **still contain removed files**. A history rewrite, GitHub garbage collection and every collaborator’s reclone are a separate authorized migration. If actual credentials were ever stored, rotate them immediately; removing HEAD content is not sufficient.
- Application and model are not certified for public exposure or road safety decision making. Earlier CI tests operate with synthetic field records; no GitHub runner has the real Windows W04 model, private original video or independently authenticated survey observations.

## Security guardrails verified by automated checks

1. `scripts/repository_hygiene.py`: tracked-index guard for venv, pycache, private keys, raw video, local DB, nonresearch oversized assets and known-pattern accidentally committed tokens. This is a heuristic, NOT an exhaustive secret scan.
2. `scripts/repository_smoke.py`: starts an actual localhost Uvicorn server, loads all integrated UI pages, creates/reads a project, checks project-bound evidence gates and verifies a non-loopback HTTP Host receives 403 even for a write request. It avoids external network.
3. M7 access-guard unit tests, M8 project/evidence tamper tests and M9 original field records consistency tests run during full integration regression.
4. Native SUMO experiment/acceptance tests verify real simulation executable against **synthetic fixtures**, with no human-measured scientific claims.
5. Windows runner parser checks execute on Windows CI. These do **not** boot the actual OpenVINO model or open the original multi-GB video.

## Operator acceptance remains external

The formal M8 local CLI verifies hashes, provenance, native M4/M5 output and operator-attested field records. It always says `production_release_approved=false`. Production approval requires separate evidence and an authorized human review of physical measurements, assumptions, site permissions and model limitations.

## Safe future maintenance

- Make a new branch from the latest audited default branch.
- Never rewrite frozen W04 research, field observation raw sheets, SHA-bound M3/M4/M5 receipts or locked holdout artifacts.
- Run hygiene + full integration + native SUMO CI; request review and merge by GitHub in a documented order.
- Keep main protected and clone from latest default branch; archive any personal physical-field evidence and raw video outside GitHub.
- Dependency specifications currently contain floating lower bounds; supply-chain vulnerability scanning and deterministic worker/runtime lockfiles are recommended before operating with real outside users.
