# Recalculate saved W04 development scores against original FLUID truth.
# No model inference and NO independent holdout evaluation.
[CmdletBinding()]
param(
    [string]$BatchDir = '',
    [string]$Readiness = '',
    [string]$Output = '',
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab Python missing: $Python" }
if ([string]::IsNullOrWhiteSpace($BatchDir)) { throw 'Specify -BatchDir pointing to original W04_box_tracking_batch201_01' }
if (-not (Test-Path -LiteralPath $BatchDir -PathType Container)) { throw "W04 original Build 3 directory unavailable: $BatchDir" }
if ([string]::IsNullOrWhiteSpace($Readiness)) {
    $latest = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -File -Filter 'w04_readiness_*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw 'No verified local W04 readiness JSON found; run STREETLAB_W04_READINESS.ps1 first' }
    $Readiness = $latest.FullName
}
if (-not (Test-Path -LiteralPath $Readiness -PathType Leaf)) { throw "W04 readiness JSON does not exist: $Readiness" }
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Repo ('artifacts\w04_score_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json')
}
if (Test-Path -LiteralPath $Output) { throw "Refusing to overwrite W04 evidence: $Output" }
Write-Host 'W04 development score replay from original 201-frame tracks (NOT independent heldout)'
Write-Host "Batch: $BatchDir"
Write-Host "Readiness: $Readiness"
& $Python -m streetlab_integration.w04_score_replay --batch-dir $BatchDir --readiness $Readiness --output $Output
if ($LASTEXITCODE -ne 0) { throw 'W04 development score replay failed; source or score integrity not established.' }
Write-Host "Replay evidence: $Output"
