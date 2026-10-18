# Verify prospective holdout source readiness without running video inference.
[CmdletBinding()]
param(
    [string]$ProtocolFolder = '',
    [string]$CandidateVideo = '',
    [string]$CandidateTruth = '',
    [string]$BaselineTracks = '',
    [ValidateSet('none','original_t000','new_replacement')][string]$BaselineKind = 'none',
    [string]$Output = '',
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if ([string]::IsNullOrWhiteSpace($ProtocolFolder)) {
    $latest = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_policy_lock_*' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw 'No W04 policy lock folder; run STREETLAB_W04_PREREGISTER.ps1 first' }
    $ProtocolFolder = $latest.FullName
}
$protocol = Join-Path $ProtocolFolder 'holdout_protocol.json'
$lock = Join-Path $ProtocolFolder 'policy_lock.json'
foreach ($file in @($protocol, $lock)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing original lock evidence: $file" }
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Repo ('artifacts\w04_holdout_intake_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json')
}
if (Test-Path -LiteralPath $Output) { throw "Output already exists: $Output" }
$ArgsList = @('-m', 'streetlab_integration.w04_holdout_intake', '--protocol', $protocol, '--policy-lock', $lock, '--output', $Output, '--baseline-kind', $BaselineKind)
if (-not [string]::IsNullOrWhiteSpace($CandidateVideo)) { $ArgsList += @('--video', $CandidateVideo) }
if (-not [string]::IsNullOrWhiteSpace($CandidateTruth)) { $ArgsList += @('--truth', $CandidateTruth) }
if (-not [string]::IsNullOrWhiteSpace($BaselineTracks)) { $ArgsList += @('--baseline', $BaselineTracks) }
Write-Host 'Read-only W04 candidate intake: no blind pass or production approval will be issued.'
& $Python @ArgsList
if ($LASTEXITCODE -ne 0) { throw 'Source intake contract failed; verify the policy lock/protocol source hashes.' }
Write-Host "Source intake receipt: $Output"
