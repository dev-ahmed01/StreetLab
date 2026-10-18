# Enumerate possible local video + FLUID trajectory pairs without opening video frames.
[CmdletBinding()]
param(
    [string[]]$Roots = @(),
    [string]$ProtocolFolder = '',
    [ValidateRange(0,12)][int]$MaxDepth = 4,
    [ValidateRange(1,200000)][int]$MaxEntries = 20000,
    [string]$Output = '',
    [string]$Python = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab Python missing: $Python" }
if ($Roots.Count -eq 0) {
    $defaultRoot = Join-Path (Split-Path -Parent $Repo) 'StreetLabData'
    if (-not (Test-Path -LiteralPath $defaultRoot -PathType Container)) {
        throw "Dataset root not found: $defaultRoot. Pass -Roots @('C:\path\to\dataset','D:\path\to\labels')"
    }
    $Roots = @($defaultRoot)
}
if ([string]::IsNullOrWhiteSpace($ProtocolFolder)) {
    $latest = Get-ChildItem -Path (Join-Path $Repo 'artifacts') -Directory -Filter 'w04_policy_lock_*' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw 'No W04 policy lock folder; run STREETLAB_W04_PREREGISTER.ps1 first.' }
    $ProtocolFolder = $latest.FullName
}
$protocol = Join-Path $ProtocolFolder 'holdout_protocol.json'
$lock = Join-Path $ProtocolFolder 'policy_lock.json'
$manifest = Join-Path $Repo 'data\benchmarks\fluid_fidrt\manifest.json'
foreach ($file in @($protocol, $lock, $manifest)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing frozen research evidence: $file" }
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Repo ('artifacts\w04_dataset_inventory_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json')
}
if (Test-Path -LiteralPath $Output) { throw "Refusing to overwrite inventory receipt: $Output" }
$ArgList = @('-m', 'streetlab_integration.w04_dataset_inventory', '--protocol', $protocol,
             '--policy-lock', $lock, '--manifest', $manifest, '--max-depth', [string]$MaxDepth,
             '--max-entries', [string]$MaxEntries, '--output', $Output)
foreach ($dir in $Roots) {
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { throw "Dataset folder missing: $dir" }
    $ArgList += @('--root', $dir)
}
Write-Host 'Local dataset inventory only: candidate file matches are NOT independently verified holdouts.'
& $Python @ArgList
if ($LASTEXITCODE -ne 0) { throw 'W04 local dataset inventory failed; no valid inventory report created.' }
Write-Host "Candidate inventory: $Output"
