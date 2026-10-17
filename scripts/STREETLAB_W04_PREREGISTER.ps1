# Freeze one W04 development policy and prospective holdout gates (NO unseen scoring).
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BatchDir,
    [string]$Replay = '',
    [ValidateRange(0.001,60.0)][double]$MaxCpuSecondsPerFrame = 2.0,
    [string]$OutputDir = '',
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python is missing: $Python" }
if (-not (Test-Path -LiteralPath $BatchDir -PathType Container)) { throw "Frozen W04 batch missing: $BatchDir" }
if ([string]::IsNullOrWhiteSpace($Replay)) {
    $latest = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -Filter 'w04_score_replay_*.json' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw 'No local verified W04 score replay JSON; run STREETLAB_W04_SCORE_REPLAY.ps1 first' }
    $Replay = $latest.FullName
}
if (-not (Test-Path -LiteralPath $Replay -PathType Leaf)) { throw "Replay missing: $Replay" }
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo ('artifacts\w04_policy_lock_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing overwrite of evidence: $OutputDir" }
Write-Host 'Preregistering provisional W04 hard_nms_ios_0.30 policy; NOT a blind holdout evaluation.'
Write-Host "Original Build 3: $BatchDir"
Write-Host "Verified replay: $Replay"
& $Python -m streetlab_integration.w04_preregistration --project-root $Repo --batch-dir $BatchDir --replay $Replay --output-dir $OutputDir --policy 'hard_nms_ios_0.30' --max-cpu-seconds-per-frame $MaxCpuSecondsPerFrame
if ($LASTEXITCODE -ne 0) { throw 'Prospective policy preregistration blocked: inspect original source and replay. Do not retune to force acceptance.' }
Write-Host "Policy lock: $(Join-Path $OutputDir 'policy_lock.json')"
Write-Host "Research protocol: $(Join-Path $OutputDir 'holdout_protocol.json')"
Write-Host 'Independent holdout remains pending until an untouched external source and baseline are verified.'
