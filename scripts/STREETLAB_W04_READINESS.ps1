# Read-only local W04 source/model readiness audit. Does not run inference.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Video,
    [Parameter(Mandatory=$true)][string]$ModelDir,
    [Parameter(Mandatory=$true)][string]$FrozenProvenance,
    [string]$Truth = "",
    [string]$Output = "",
    [string]$Python = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Repo ".venv-sahi-audit\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "StreetLab virtualenv Python missing: $Python"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Repo ("artifacts\w04_readiness_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + ".json")
}
if (Test-Path -LiteralPath $Output) {
    throw "Output must be new to preserve previous evidence: $Output"
}
$ArgsList = @("-m", "streetlab_integration.w04_readiness",
    "--video", $Video, "--model-dir", $ModelDir,
    "--provenance", $FrozenProvenance, "--output", $Output)
if (-not [string]::IsNullOrWhiteSpace($Truth)) {
    $ArgsList += @("--truth", $Truth)
}
Write-Host "Read-only W04 audit output: $Output"
& $Python @ArgsList
if ($LASTEXITCODE -ne 0) {
    Write-Host "W04 source/model integrity NOT VERIFIED; review blockers above and the JSON report."
    exit $LASTEXITCODE
}
Write-Host "W04 source/model matched provided frozen provenance. This is NOT independent holdout validation."
