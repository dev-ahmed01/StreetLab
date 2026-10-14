[CmdletBinding()]
param(
    [ValidateSet('Prepare','Consensus')]
    [string]$Mode = 'Prepare',
    [string]$Reviewer01 = '',
    [string]$Reviewer02 = ''
)

# Research-only W04 source-video reviewer workflow: never overwrites primary tracks.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv-sahi-audit\Scripts\python.exe'
$batch = Join-Path $root 'artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01'
$replay = Join-Path $root 'artifacts\phase3\sahi_detector_trials\W04_unified_hybrid_replay_01'
$sourceVideo = 'C:\Users\Admin\Desktop\StreetLabData\Video_2\20250526_video.mp4'
$reviewDir = Join-Path $root 'artifacts\phase3\sahi_detector_trials\W04_VISUAL_TWO_REVIEWER_KIT_01'
$consensusDir = Join-Path $root 'artifacts\phase3\sahi_detector_trials\W04_VISUAL_CONSENSUS_01'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

$oldPythonPath = $env:PYTHONPATH
try {
    if ($oldPythonPath) {
        $env:PYTHONPATH = $root + [System.IO.Path]::PathSeparator + $oldPythonPath
    } else {
        $env:PYTHONPATH = $root
    }
    if ($Mode -eq 'Prepare') {
        $generator = Join-Path $root 'scripts\phase3_w04_visual_review_pack.py'
        foreach ($required in @($generator, $sourceVideo,
            (Join-Path $batch 'batch_report.json'),
            (Join-Path $batch 'source_provenance.json'),
            (Join-Path $replay 'unified_hybrid_report.json'))) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "Cannot prepare visual packet: missing exact original input $required"
            }
        }
        if (Test-Path -LiteralPath $reviewDir) {
            throw "Will not overwrite existing visual review evidence: $reviewDir"
        }
        Write-Host 'Checking source SHA and all original W04 tracking evidence.'
        Write-Host 'Generating 2 BLINDED reviewer ZIPs; no model/tracker inference.'
        & $python $generator --batch-dir $batch --replay-dir $replay --video $sourceVideo --output-dir $reviewDir
        if ($LASTEXITCODE -ne 0) {
            throw "Visual review packet generation failed (Python exit $LASTEXITCODE)"
        }
        Write-Host "Reviewer 1: $(Join-Path $reviewDir 'REVIEWER_R01_ONLY.zip')"
        Write-Host "Reviewer 2: $(Join-Path $reviewDir 'REVIEWER_R02_ONLY.zip')"
        Write-Host 'Send separately. Do not share INTERNAL_case_manifest.json with reviewers.'
        Write-Host 'Reviewers open their index.html offline and export independent CSVs.'
    } else {
        if (-not $Reviewer01 -or -not $Reviewer02) {
            throw 'Consensus needs -Reviewer01 and -Reviewer02 CSV file paths.'
        }
        $consensus = Join-Path $root 'scripts\phase3_w04_review_consensus.py'
        foreach ($required in @($consensus,
            (Join-Path $reviewDir 'INTERNAL_case_manifest.json'),
            $Reviewer01, $Reviewer02)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "Missing required independent review evidence: $required"
            }
        }
        if (Test-Path -LiteralPath $consensusDir) {
            throw "Will not overwrite completed reviewer consensus: $consensusDir"
        }
        & $python $consensus --packet-dir $reviewDir --reviewer-01 $Reviewer01 --reviewer-02 $Reviewer02 --output-dir $consensusDir
        if ($LASTEXITCODE -ne 0) {
            throw "Review agreement/adjudication failed (Python exit $LASTEXITCODE)"
        }
        Write-Host "Human review consensus report: $consensusDir"
        Write-Host 'Consensus is evidence only: original FLUID and tracks remain unchanged.'
    }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
