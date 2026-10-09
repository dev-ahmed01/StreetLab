# StreetLab M8 Windows single-operator launcher.
# Open TWO terminals: -Mode web and -Mode worker. Never hosts public routes.
# A release acceptance check is read-only and does not fabricate field data.
[CmdletBinding()]
param(
    [ValidateSet("environment","web","worker","acceptance","audit","backup")]
    [string]$Mode = "environment",
    [string]$Workdir = ".streetlab-m5",
    [string]$Python = "",
    [string]$ModelDir = "",
    [string]$FrozenProvenance = "",
    [string]$Project = "",
    [string]$FieldDir = "",
    [switch]$VerifySourceSha,
    [string]$OutputDir = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Repo ".venv-sahi-audit\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python missing. Supply -Python pointing to the verified virtualenv python.exe"
}
if ($Workdir -notmatch '^[a-zA-Z0-9._\\/:\- ]+$') {
    throw "Workdir contains unsupported characters"
}
if (-not [System.IO.Path]::IsPathRooted($Workdir)) {
    $Workdir = Join-Path $Repo $Workdir
}
$env:STREETLAB_SECURITY_MODE = "local"
function Require-File([string]$Name,[string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or -not (Test-Path -LiteralPath $Value -PathType Leaf)) {
        throw "$Name must be an existing local file"
    }
}
switch ($Mode) {
    "environment" {
        & $Python -m streetlab_integration.release_cli --environment
    }
    "web" {
        Write-Host "StreetLab local web only: http://127.0.0.1:8000"
        Write-Host "Start the separate frozen worker in another PowerShell terminal."
        & $Python -m uvicorn streetlab_phase2.api:app --host 127.0.0.1 --port 8000
    }
    "worker" {
        if ([string]::IsNullOrWhiteSpace($ModelDir) -or -not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
            throw "ModelDir must be the frozen W04 OpenVINO model directory"
        }
        Require-File "FrozenProvenance" $FrozenProvenance
        & $Python -m streetlab_integration.worker --workdir $Workdir --model-dir $ModelDir --frozen-provenance $FrozenProvenance
    }
    "acceptance" {
        if ([string]::IsNullOrWhiteSpace($Project)) {
            throw "Project must be the actual existing project UUID"
        }
        $ArgsList = @("-m","streetlab_integration.release_cli","--workdir",$Workdir,"--project",$Project)
        if (-not [string]::IsNullOrWhiteSpace($ModelDir)) {
            if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) { throw "ModelDir not found" }
            $ArgsList += @("--model-dir",$ModelDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($FrozenProvenance)) {
            Require-File "FrozenProvenance" $FrozenProvenance
            $ArgsList += @("--frozen-provenance",$FrozenProvenance)
        }
        if (-not [string]::IsNullOrWhiteSpace($FieldDir)) {
            if (-not (Test-Path -LiteralPath $FieldDir -PathType Container)) { throw "FieldDir not found" }
            $ArgsList += @("--field-dir",$FieldDir)
        }
        if ($VerifySourceSha) { $ArgsList += "--verify-source-sha" }
        if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { $ArgsList += @("--output-dir",$OutputDir) }
        & $Python @ArgsList
    }
    "audit" {
        & $Python -m streetlab_integration.maintenance --workdir $Workdir --audit
    }
    "backup" {
        & $Python -m streetlab_integration.maintenance --workdir $Workdir --backup
    }
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
