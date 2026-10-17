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
Write-Host "Research run output folder: $OutputDir"
& $Python @ArgsList
$runExitCode = $LASTEXITCODE
if ($runExitCode -ne 0) {
    $failure = Join-Path $OutputDir "FAILED_RESEARCH_RUN.txt"
    $nativeLog = Join-Path $OutputDir "native_sumo.log"
    if (Test-Path -LiteralPath $nativeLog -PathType Leaf) {
        Write-Host "Native SUMO failed. Showing the final 40 log lines:"
        Get-Content -LiteralPath $nativeLog -Tail 40
        Write-Host "Full native log: $nativeLog"
    }
    if (Test-Path -LiteralPath $failure -PathType Leaf) {
        Write-Host "Failure record: $failure"
    }
    exit $runExitCode
}
Write-Host "Research submission evidence: $OutputDir"
