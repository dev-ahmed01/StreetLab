# Run a research submission demo using published/pre-existing datasets.
# This does NOT synthesize actual field measurements or clear the M8 site gate.
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$OutputDir = "",
    [switch]$EvidenceOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Repo ".venv-sahi-audit\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python not found: set -Python to the verified StreetLab environment"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo ("artifacts\research_submission_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Output folder already exists; choose a new folder to preserve old evidence"
}
$ArgsList = @("-m", "streetlab_integration.research_submission", "--project-root", $Repo, "--output-dir", $OutputDir)
if (-not $EvidenceOnly) {
    if (-not (Get-Command sumo -ErrorAction SilentlyContinue)) {
        throw "SUMO is not on PATH. Open the PowerShell terminal in which 'sumo --version' succeeds."
    }
    if (-not (Get-Command netconvert -ErrorAction SilentlyContinue)) {
        throw "netconvert is not on PATH"
    }
    $ArgsList += "--run-sumo"
}
& $Python @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Research submission evidence: $OutputDir"
