# Targeted read-only investigation of HISTORICAL May-26 Geo-trax / T000 outputs.
# Run from the experimental StreetLab repo after pulling codex/phase3-engine-shootout.
# No Geo-trax or OpenVINO inference, no file rewriting, no automatic T000 certification.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo '.venv-sahi-audit\Scripts\python.exe'
$scanner = Join-Path $repo 'scripts\phase3_discover_t000_tracks.py'
$lab = Join-Path $repo 'artifacts\phase3\sahi_detector_trials'
$output = Join-Path $lab 'W04_T000_HISTORICAL_CANDIDATES_01.json'

$roots = @(
    'C:\Users\Admin\Desktop\StreetLab\artifacts\phase3\geotrax\fluid_fidrt_20250526\full',
    'C:\Users\Admin\Desktop\StreetLab\artifacts\phase3\geotrax_tuning\experiments\T000_BASELINE',
    (Join-Path $repo 'artifacts\phase3\geotrax\fluid_fidrt_20250526\full'),
    (Join-Path $repo 'artifacts\phase3\geotrax_tuning\experiments\T000_BASELINE')
) | Select-Object -Unique

foreach ($item in @($python, $scanner)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) {
        throw "Required runner is unavailable: $item"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Immutable result already exists: $output. Choose a fresh output filename before rerunning."
}

Write-Host "Checking four historical source paths (unverified leads):"
foreach ($root in $roots) {
    $exists = Test-Path -LiteralPath $root
    Write-Host ("  {0}: {1}" -f $(if ($exists) {"FOUND"} else {"MISSING"}), $root)
}

& $python $scanner --roots @($roots) --output $output
if ($LASTEXITCODE -ne 0) { throw "T000 historic inventory failed: Python exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
    throw "Expected inventory report not created: $output"
}
Write-Host "Saved candidate inventory: $output"
Write-Host "WARNING: Candidates are NOT certified T000, and sparse track rows may omit zero-detection frames."
