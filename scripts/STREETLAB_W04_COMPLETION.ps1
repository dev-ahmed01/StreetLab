# Final six-phase StreetLab research prototype receipt. No model reruns, no human labels.
[CmdletBinding()]
param(
    [string]$Python = '',
    [string]$OutputDir = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Repo '.venv-sahi-audit\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "StreetLab Python missing: $Python" }
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Repo ('artifacts\w04_completion_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing prior receipt overwrite: $OutputDir" }
& $Python -m streetlab_integration.w04_completion_gate --project-root $Repo --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Six-phase research acceptance report failed; never infer missing evidence.' }
Write-Host "Six-phase research report: $(Join-Path $OutputDir 'completion_verdict.md')"
Write-Host "Machine-readable evidence: $(Join-Path $OutputDir 'completion_verdict.json')"
Write-Host 'Research prototype gate is distinct from independent validation or production.'
