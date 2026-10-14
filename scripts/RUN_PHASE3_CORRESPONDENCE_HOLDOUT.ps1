[CmdletBinding()]
param(
    [ValidateSet('Lock','Prepare','Evaluate')][string]$Mode = 'Lock',
    [string]$DevelopmentAudit = '.\W04_PHYSICAL_CORRESPONDENCE_AUDIT.zip',
    [string]$Video = '',
    [string]$PrimaryTracks = '',
    [string]$ShadowTracks = '',
    [int]$First = -1,
    [int]$Last = -1,
    [string]$Reviewer01 = '',
    [string]$Reviewer02 = ''
)

# One frozen evaluation protocol, separate source-video review, one scoring run.
# NEVER edit W04 source/FLUID, primary tracker rows, or production geometry.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo '.venv-sahi-audit\Scripts\python.exe'
$runner = Join-Path $repo 'scripts\phase3_correspondence_holdout.py'
$folder = Join-Path $repo 'artifacts\phase3\independent_holdout'
$lock = Join-Path $folder 'CORRESPONDENCE_LOCK_V1.json'
$packet = Join-Path $folder 'PAIR_REVIEW_PACKET_01'
$score = Join-Path $folder 'PAIR_DIAGNOSTICS_01'
foreach ($file in @($python,$runner)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing required holdout executable: $file"
    }
}
$oldPythonPath = $env:PYTHONPATH
try {
    if ($oldPythonPath) {
        $env:PYTHONPATH = $repo + [System.IO.Path]::PathSeparator + $oldPythonPath
    } else {
        $env:PYTHONPATH = $repo
    }
    if ($Mode -eq 'Lock') {
        if (-not (Test-Path -LiteralPath $DevelopmentAudit -PathType Leaf)) {
            throw "Copy the prior immutable W04 development audit ZIP here: $DevelopmentAudit"
        }
        if (Test-Path -LiteralPath $lock) {
            throw 'Frozen lock already exists. Do not regenerate after examining holdout.'
        }
        & $python $runner lock --development-audit $DevelopmentAudit --out $lock
    } elseif ($Mode -eq 'Prepare') {
        if (-not (Test-Path -LiteralPath $lock -PathType Leaf)) {
            throw 'Preregister and archive the frozen lock BEFORE viewing any unseen-video labels.'
        }
        if (-not $Video -or -not $PrimaryTracks -or -not $ShadowTracks -or
            $First -lt 0 -or $Last -lt $First) {
            throw 'Prepare requires -Video, -PrimaryTracks, -ShadowTracks, -First and -Last.'
        }
        foreach ($path in @($Video,$PrimaryTracks,$ShadowTracks)) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Missing heldout original input: $path"
            }
        }
        & $python $runner prepare --lock $lock --video $Video --primary $PrimaryTracks --shadow $ShadowTracks --first $First --last $Last --out $packet
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Reviewer one ZIP: $(Join-Path $packet 'REVIEWER_R01_ONLY.zip')"
            Write-Host "Reviewer two ZIP: $(Join-Path $packet 'REVIEWER_R02_ONLY.zip')"
            Write-Host 'Distribute these separately. Keep INTERNAL_manifest.json private.'
        }
    } else {
        if (-not $Reviewer01 -or -not $Reviewer02) {
            throw 'Evaluate requires independent -Reviewer01 and -Reviewer02 CSV files.'
        }
        & $python $runner evaluate --lock $lock --packet-dir $packet --reviewer-01 $Reviewer01 --reviewer-02 $Reviewer02 --out $score
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Research diagnostic only: $score"
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Holdout $Mode aborted (Python exit code $LASTEXITCODE). No production changes."
    }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
