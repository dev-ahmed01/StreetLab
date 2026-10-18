# Autonomous, no-physical-review May25 W04 research completion on saved ORIGINAL sources.
[CmdletBinding()]
param(
    [string]$RunDir = '',
    [string]$ForensicsDir = '',
    [string]$OutputDir = '',
    [string]$Python = '',
    [switch]$SkipLocalTests
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab research Python missing: $Python" }
$Artifacts = Join-Path $Repo 'artifacts'
function FindOriginal([string]$pattern,[string]$filename,[string]$errorText) {
    $results = Get-ChildItem -LiteralPath $Artifacts -Directory -Filter $pattern -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName $filename) -PathType Leaf } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $results) { throw $errorText }
    return $results.FullName
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = FindOriginal 'w04_may25_crossdate_*' 'cross_date_summary.json' 'No completed original May25 video regression on this PC.'
}
if ([string]::IsNullOrWhiteSpace($ForensicsDir)) {
    $ForensicsDir = FindOriginal 'w04_may25_forensics_*' 'regression_forensics.json' 'No source-backed May25 forensics on this PC. Run STREETLAB_W04_MAY25_FORENSICS.ps1 once.'
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Artifacts ('w04_autonomous_research_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing overwrite of existing research outputs: $OutputDir" }
Write-Host 'AUTONOMOUS MAY25 OBJECT-LEVEL RESEARCH AUDIT'
Write-Host 'No physical review or new OpenVINO inference. Original labels, metrics and W04 policy frozen.'
Write-Host "Original run: $RunDir"
Write-Host "Original forensics: $ForensicsDir"
if (-not $SkipLocalTests) {
    Write-Host 'First running original scorer/object-matcher regression tests locally.'
    & $Python -m pytest -q tests/test_w04_autonomous_diagnostics.py tests/test_w04_crossdate_forensics.py tests/test_w04_visual_review_packet.py
    if ($LASTEXITCODE -ne 0) { throw 'Pre-run regression tests failed. No real source audit executed.' }
}
& $Python -m streetlab_integration.w04_autonomous_diagnostics --run-dir $RunDir --forensics-dir $ForensicsDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Source-backed autonomous audit failed. Do not retune, modify original files or claim validation.' }
Write-Host "FINAL RESEARCH REPORT: $(Join-Path $OutputDir 'research_findings.md')"
Write-Host "Full objective evidence: $(Join-Path $OutputDir 'automated_object_audit.json')"
Write-Host "All FLUID objects: $(Join-Path $OutputDir 'all_truth_track_diagnostics.csv')"
Write-Host "Tracked model class stability: $(Join-Path $OutputDir 'predicted_track_class_stability.csv')"
Write-Host 'Historical FLUID-label regression complete; genuine unseen generalization remains unverified.'
