# StreetLab — Phase 1A Starter

This starter isolates the first scientific question:

> Can real Indian mixed-traffic trajectories be converted into stable behavioural measurements that we can later reproduce in SUMO?

It deliberately does **not** start with AI agents, computer vision, road redesign, or scenario testing.

## Dataset roles

- `ChennaiTrajectoryData2.45-3.00PM.xlsx` → **CALIBRATION / development**
- `ChennaiTrajectoryData3.00-3.15PM.xlsx` → **VALIDATION / holdout**

Never tune the behaviour model on the validation file.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy both Excel files into `data/raw/`.

Then run:

```powershell
$env:PYTHONPATH = "."
python scripts/phase1_profile.py `
  --calibration "data/raw/ChennaiTrajectoryData2.45-3.00PM.xlsx" `
  --validation "data/raw/ChennaiTrajectoryData3.00-3.15PM.xlsx"
```

## Outputs

`data/processed/` will contain:

- `calibration_clean.parquet`
- `validation_clean.parquet`
- `calibration_vehicle_features.parquet`
- `validation_vehicle_features.parquet`
- `calibration_class_profile.csv`
- `validation_class_profile.csv`
- `calibration_qa.json`
- `validation_qa.json`

## What the first script proves

1. We can map the source spreadsheet to a stable StreetLab trajectory schema.
2. We remove manually corrected/flagged points when the source provides such a flag.
3. We check timestamp sampling and obvious physical/data problems.
4. We create one feature row per vehicle without inventing personas.
5. Calibration and validation data stay separated.

## What comes next

**Sprint 1B:** validate features and identify leader/follower + lateral-neighbour interactions.

**Sprint 1C:** compare two hypotheses:

- continuous behaviour distributions;
- a small number of data-derived behaviour clusters/personas.

Personas only survive if they improve held-out reproduction.

**Sprint 1D:** map the validated distributions to SUMO `vType` / `vTypeDistribution` and sublane parameters.

**Sprint 1E:** recreate the observed Chennai traffic and score simulated vs real speed/lateral-motion distributions using the untouched validation period.

## Important

This dataset starts after the video-to-trajectory step. A later Phase 1B-video track will prove:

`raw video → detection → tracking → road coordinates → the same StreetLab trajectory schema`.

That keeps CV errors separate from traffic-model errors.
