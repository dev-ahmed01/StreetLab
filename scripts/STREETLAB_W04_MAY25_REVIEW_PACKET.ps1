# Create SHA-verified representative original May25 review frames, no detector rerun.
[CmdletBinding()]
param(
    [string]$RunDir = '',
    [string]$ForensicsDir = '',
    [string]$OutputDir = '',
    [ValidateRange(1,30)][int]$MaxSamples = 14,
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab Python missing: $Python" }
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $recent = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_may25_crossdate_*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'cross_date_summary.json') -PathType Leaf } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $recent) { throw 'No completed original May25 cross-date run found.' }
    $RunDir = $recent.FullName
}
if ([string]::IsNullOrWhiteSpace($ForensicsDir)) {
    $recent = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_may25_forensics_*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'regression_forensics.json') -PathType Leaf } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $recent) { throw 'No completed May25 forensics found. Run STREETLAB_W04_MAY25_FORENSICS.ps1 first.' }
    $ForensicsDir = $recent.FullName
}
foreach ($f in @(
    (Join-Path $RunDir 'cross_date_summary.json'),
    (Join-Path $RunDir 'preflight.json'),
    (Join-Path $RunDir 'batch\source_provenance.json'),
    (Join-Path $RunDir 'batch\batch_report.json'),
    (Join-Path $ForensicsDir 'regression_forensics.json'),
    (Join-Path $ForensicsDir 'review_events.csv')
)) { if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { throw "Missing original forensic source: $f" } }
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo ('artifacts\w04_may25_review_packet_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing evidence overwrite: $OutputDir" }
Write-Host 'Building original May25 track-diverse review crops; no model inference, no FLUID changes.'
Write-Host "Completed cross-date run: $RunDir"
Write-Host "Verified forensics: $ForensicsDir"
& $Python -m streetlab_integration.w04_visual_review_packet --run-dir $RunDir --forensics-dir $ForensicsDir --output-dir $OutputDir --max-samples $MaxSamples
if ($LASTEXITCODE -ne 0) { throw 'Review image generation failed; the original evidence must remain unchanged.' }
Write-Host "Visual sheet: $(Join-Path $OutputDir 'review_contact_sheet.jpg')"
Write-Host "Reviewer forms: $(Join-Path $OutputDir 'reviewer_1_blank.csv') ; $(Join-Path $OutputDir 'reviewer_2_blank.csv')"
Write-Host "Source proof: $(Join-Path $OutputDir 'review_packet_manifest.json')"
Write-Host 'Upload the contact sheet for image-level diagnosis; no independent human validation is presumed.'
