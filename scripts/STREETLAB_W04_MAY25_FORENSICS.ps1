# Source-bound May25 forensic scorer and error-frame queue. No detector inference.
[CmdletBinding()]
param(
    [string]$RunDir = '',
    [string]$ProtocolFolder = '',
    [string]$OutputDir = '',
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab Python missing: $Python" }
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $latest = Get-ChildItem -LiteralPath (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_may25_crossdate_*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'cross_date_summary.json') -PathType Leaf } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw 'No completed May25 cross-date run found. Uploading just the summary is not enough for source-backed forensic rescore.' }
    $RunDir = $latest.FullName
}
if ([string]::IsNullOrWhiteSpace($ProtocolFolder)) {
    $latestLock = Get-ChildItem -LiteralPath (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_policy_lock_*' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latestLock) { throw 'No frozen W04 policy-lock folder found.' }
    $ProtocolFolder = $latestLock.FullName
}
foreach ($required in @(
    (Join-Path $RunDir 'cross_date_summary.json'),
    (Join-Path $RunDir 'preflight.json'),
    (Join-Path $RunDir 'batch\batch_report.json'),
    (Join-Path $ProtocolFolder 'holdout_protocol.json'),
    (Join-Path $ProtocolFolder 'policy_lock.json')
)) { if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing original saved evidence: $required" } }
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo ('artifacts\w04_may25_forensics_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite previous forensic evidence: $OutputDir" }
Write-Host 'Replaying May25 saved tracks against original FLUID and frozen W04 policy. No model inference.'
Write-Host "Run: $RunDir"
Write-Host "Policy lock: $ProtocolFolder"
& $Python -m streetlab_integration.w04_crossdate_forensics --run-dir $RunDir --protocol-folder $ProtocolFolder --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Forensic replay failed; original source or evidence could not be verified. No score approval.' }
Write-Host "Forensic report: $(Join-Path $OutputDir 'regression_forensics.json')"
Write-Host "Bounded review queue: $(Join-Path $OutputDir 'review_events.csv')"
Write-Host 'These are FLUID-label candidate disagreements, NOT independently adjudicated physical objects.'
