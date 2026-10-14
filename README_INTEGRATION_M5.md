# StreetLab — Integration Phase 5 / 8: Observed-Site Scenario Experiments

M5 extends the M1 observation -> M2 native tracking -> M3 manually reviewed local-meter junction -> M4 field-measured site baseline chain. It adds immutable site-specific hypothetical changes, paired actual SUMO executions and comparison evidence. W04 frozen primary and holdout, Phase 1 persona calibration and Phase 2 synthetic demonstration are unchanged.

## Product experience

The unified dashboard now offers `/scenarios` (http://127.0.0.1:8000/scenarios). Select a project. M5 verifies the exact source-linked M4 baseline and must find `BASELINE_FIDELITY_CHECKED` with `real_site_sumo_allowed=true` before it will package a scenario. Uploading new evidence or a different baseline version never silently changes old scenarios.

Two deliberately bounded intervention families:

- `APPROACH_SPEED_LIMIT`: choose an existing M4 APPROACH and submit a lower reviewed `speed_mps` value no more than 50% below baseline. Does not change physical lane count or road routing. A speed-limit change is only a hypothetical SUMO intervention, not a prediction of driver compliance.
- `FIXED_SIGNAL_PLAN`: available only if M4 specified an actual fixed-time signal; give a duration for each original phase (within 50% of original), preserving each signal state and lane-link order. Adaptive lights, unsurveyed signals, blocked turns, new road construction and rerouting are **not fabricated**.

Every proposal includes a decision question, feasibility evidence reference, named/datable review record and explicit provenance `HYPOTHETICAL_REVIEWED`. No physical effect is claimed at proposal creation.

## Immutable and reproducible experiment

After an allowed M4 baseline, `POST /api/projects/{project_id}/scenario` accepts `{baseline_revision: '<sha256>', scenario: {intervention: {...}, assumptions: {...}, demand_multipliers: [0.9,1.0,1.1]}}`. Only the fixed-seed array [42,43,44] is used and the demand sensitivity must have one low condition in [0.8,1), central 1, and one high in (1,1.2]. These stress levels are assumptions, not measured arrival-variation confidence bounds.

The proposal is hashed and stored immutably by project; the M4 baseline revision, original source and tracking hashes, M3 spatial revision and M4 runtime receipt SHA are pinned. Each read verifies both the proposal and upstream evidence hashes.

A **standalone CLI** uses the existing verified M4 baseline `site.net.xml` and compiles a new scenario network with real netconvert; it then runs **nine paired conditions** = 3 loads × 3 seeds. Each condition executes original baseline and intervention with identical per-condition demand/seed (18 real SUMO runs total). Inputs and outputs are bounded; the overall run has a 16-minute wall-clock budget and individual subprocesses a 50-second budget. There is no remote SUMO execution in HTTP handlers.

For each case the engine reads genuine SUMO tripinfo, verifies vehicle-ID uniqueness and checks that no simulated route has more completed trips than its configured flow. The main metric is simulated mean completed-trip duration; each pair reports baseline mean, changed mean, completion, number of trips, per-movement summaries and delta = changed minus baseline in seconds. A negative delta means faster **in the simulator only**.

Comparison eligibility requires >=95% completed demand **in both sides of every pair**, at least one completed trip per declared movement in every pair, and a complete nine-pair matrix. If any condition is incomplete or missing, no aggregate mean advantage is published. Paired difference minimum and maximum over the nine experimental settings describe a **sensitivity envelope**, not a statistical confidence interval. Scenarios that look better in all nine pairs are labeled `simulated_improvement_in_every_condition` but never as proven real-world improvement.

## API

| Method | URL | Purpose |
|---|---|---|
| GET | `/scenarios` | Interactive experiment design and comparison panel |
| GET | `/api/projects/{project_id}/scenario` | Latest project scenario or explicit missing-evidence state |
| POST | `/api/projects/{project_id}/scenario` | Create reviewed immutable scenario proposal (never run SUMO here) |
| GET | `/api/projects/{project_id}/scenario/{revision}` | Exact proposal and independently verified runtime |
| GET | `/api/projects/{project_id}/scenario/{revision}/results` | Nine-pair result after successful CLI execution only |
| GET | `/api/projects/{project_id}/scenario/{revision}/files/{filename}` | SHA-verified allowlisted SUMO artifact, with no arbitrary client file path |

## Windows workflow

Preserve local changes first, then fetch M5 branch **stacked on M4**:

    cd C:\Users\Admin\Desktop\StreetLab-engine-trial
    git status --short
    git fetch origin
    git switch --track origin/codex/streetlab-integration-m5

If the branch already exists locally use `git switch codex/streetlab-integration-m5` instead.

Start the existing FastAPI server with the working project environment:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000

Open `http://127.0.0.1:8000/scenarios`. **A completed M4 baseline that passes independent field-holdout QA is required.** Choose an intervention and supply review records. M5 returns a scenario SHA. Install `sumo` and `netconvert` if absent; execute in a separate local PowerShell terminal:

    & '.\.venv-sahi-audit\Scripts\python.exe' -m streetlab_integration.scenario_experiments --workdir '.streetlab-m5' --project 'REAL_PROJECT_UUID' --revision 'REAL_SCENARIO_SHA256'

Replace uppercase placeholders with exact UI values. Use optional `--sumo-bin-dir 'C:\SUMO\bin'` when the real binaries aren't on PATH. The result is atomic/immutable and can be inspected or downloaded through `/scenarios` and the project REST endpoints. Failures do not quietly publish partial results.

## Verification and scientific boundaries

M5 GitHub CI runs all M1–M5 contract suites and installs actual SUMO/netconvert for native engine smoke. All used traffic measurements are **synthetic software fixtures**. Its native scenario test initially executes SUMO against a software fixture and creates another synthetic baseline whose 'holdout' is deliberately derived from software output. This is a toolchain acceptance procedure **not an independent holdout**, never empirical calibration. The CI assertion checks successful native paired execution and conservative reporting only.

The product's real decision gate requires authentic real-world data and an independent field holdout. A 95% completion rate and within-20% movement travel time baseline criterion is a product QA heuristic, **not scientific accreditation**. M5 makes no real-road travel-time savings, emission, congestion, speed, compliance, socioeconomic-benefit or causal claim. Use stronger model review and field evaluation for decisions with significant practical consequences.

## Remaining phases

M6: unify operational workspace navigation, project maps/observation/baseline/scenario charts and guided evidence messages. M7: accessible reports, provenance exports, security/performance hardening. M8: end-to-end real-source and field-survey acceptance, deployment packaging and reproducibility.