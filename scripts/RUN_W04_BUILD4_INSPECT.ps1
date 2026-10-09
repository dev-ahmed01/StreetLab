# Run from any working directory after pulling codex/phase3-engine-shootout.
# Phase 3 research only. Does not alter W04/T000/FLUID/Geo-trax source artifacts.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo '.venv-sahi-audit\Scripts\python.exe'
$runner = Join-Path $repo 'scripts\phase3_holdout_validation.py'
$lab = Join-Path $repo 'artifacts\phase3\sahi_detector_trials'
$batch = Join-Path $lab 'W04_box_tracking_batch201_01'
$output = Join-Path $lab 'W04_continuous_build4_integrity_01.json'

foreach ($item in @($python, $runner, $batch)) {
  if (-not (Test-Path -LiteralPath $item)) { throw "Required file/folder missing: $item" }
}
if (Test-Path -LiteralPath $output) {
  throw "Refusing overwrite of immutable validation report: $output. Choose a new unique report filename."
}

Write-Host 'Build 4 inspect: SHA-check original video and FLUID, all 25 track exports and original raw-box evidence.'
& $python $runner inspect --batch-dir $batch --output $output
if ($LASTEXITCODE -ne 0) { throw "Build 4 inspection failed: exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $output)) { throw "Missing expected Build 4 verification file: $output" }
Write-Host "Build 4 immutable inspection report: $output"
Write-Host 'NOTE: This is evidence integrity only. T000 same-window comparison, true holdout and physical review are still pending.'
