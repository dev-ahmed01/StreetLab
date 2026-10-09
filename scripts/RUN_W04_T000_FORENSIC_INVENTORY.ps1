# One bounded metadata/artifact recovery pass; research only.
# No Geo-trax execution, no source rewrite, no T000 certification.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv-sahi-audit\Scripts\python.exe"
$tool = Join-Path $repo "scripts\phase3_t000_forensic_inventory.py"
$output = Join-Path $repo "artifacts\phase3\sahi_detector_trials\W04_T000_FORENSIC_INVENTORY_01.json"
$desktop = "C:\Users\Admin\Desktop"

foreach ($required in @($python, $tool)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing Python or audit tool: $required"
    }
}
if (-not (Test-Path -LiteralPath $desktop -PathType Container)) {
    throw "Expected desktop is unavailable: $desktop"
}
if (Test-Path -LiteralPath $output) {
    throw "Will not overwrite an earlier forensic report: $output"
}
Write-Host "Checking historical T000 output paths and related metadata across StreetLab checkouts..."
& $python $tool --desktop $desktop --output $output
if ($LASTEXITCODE -ne 0) { throw "Historical T000 forensic scan failed, exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
    throw "Expected forensic inventory is missing: $output"
}
Write-Host "Report saved: $output"
Write-Host "No video was decoded, no original files were changed, and no baseline was promoted."
