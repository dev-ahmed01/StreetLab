# Real May-25 cross-date regression with the May-26 frozen W04 model/policy.
# It is not an independent blind holdout. Runs 150 continuous frames on CPU.
[CmdletBinding()]
param(
    [string]$ProtocolFolder = '',
    [string]$Video = 'C:\Users\Admin\Desktop\StreetLabData\Video_1\20250525_video.mp4',
    [string]$Truth = 'C:\Users\Admin\Desktop\StreetLabData\Video_1\20250525_video_Traj.csv',
    [string]$ModelDir = 'C:\Users\Admin\Desktop\StreetLab-engine-trial\artifacts\phase3\sahi_detector_trials\W04_openvino_export640_01\checkpoint_openvino_model',
    [string]$OutputDir = '',
    [string]$Python = '',
    [switch]$PreflightOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab Python missing: $Python" }
foreach ($file in @($Video, $Truth)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing original May-25 source: $file" }
}
if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) { throw "Frozen W04 OpenVINO export missing: $ModelDir" }
if ([string]::IsNullOrWhiteSpace($ProtocolFolder)) {
    $latest = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_policy_lock_*' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw 'Missing original W04 policy lock; run STREETLAB_W04_PREREGISTER.ps1 first.' }
    $ProtocolFolder = $latest.FullName
}
$protocol = Join-Path $ProtocolFolder 'holdout_protocol.json'
$lock = Join-Path $ProtocolFolder 'policy_lock.json'
foreach ($file in @($protocol, $lock)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing locked W04 evidence: $file" }
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo ('artifacts\w04_may25_crossdate_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite existing study output: $OutputDir" }
$ArgList = @('-m', 'streetlab_integration.w04_crossdate_regression', '--protocol', $protocol,
             '--policy-lock', $lock, '--video', $Video, '--truth', $Truth,
             '--model-dir', $ModelDir, '--output-dir', $OutputDir)
Write-Host 'May-25 historical REGRESSION reference. Model and W04 policy remain frozen.'
Write-Host 'Window: source 60-179; tracker warmup source 30-59; FLUID truth 61-180.'
if (-not $PreflightOnly) {
    Write-Host 'Executing actual SAHI/OpenVINO + ByteTrack on 150 consecutive frames. CPU detector may be expensive.'
    $ArgList += '--run-inference'
} else {
    Write-Host 'PreflightOnly selected: no detector inference will occur.'
}
& $Python @ArgList
if ($LASTEXITCODE -ne 0) {
    Write-Host "Any completed preflight remains in: $(Join-Path $OutputDir 'preflight.json')"
    throw 'Cross-date experiment failed; frozen source must not be modified to force success.'
}
Write-Host "Cross-date run folder: $OutputDir"
Write-Host 'The result cannot establish independent holdout performance or production readiness.'
