[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Video,
    [Parameter(Mandatory=$true)][string]$ModelDir,
    [string]$FrozenW04SourceProvenance = '',
    [int]$FirstEvalFrame = 30,
    [int]$LastEvalFrame = 329,
    [int]$WarmupFrames = 30
)

# One new-video OpenVINO inference pass -> two independent ByteTrack feeds
# -> one blinded pair review pack. Original primary, FLUID, W04 unchanged.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv-sahi-audit\Scripts\python.exe'
$capture = Join-Path $root 'scripts\phase3_unseen_dual_lane.py'
$review = Join-Path $root 'scripts\phase3_correspondence_holdout.py'
$folder = Join-Path $root 'artifacts\phase3\independent_holdout'
$lock = Join-Path $folder 'CORRESPONDENCE_LOCK_V1.json'
$output = Join-Path $folder 'DUAL_LANE_SOURCE_01'
$packets = Join-Path $folder 'PAIR_REVIEW_PACKET_01'
if (-not $FrozenW04SourceProvenance) {
    $FrozenW04SourceProvenance = Join-Path $root 'artifacts\phase3\sahi_detector_trials\W04_box_tracking_batch201_01\source_provenance.json'
}
foreach ($path in @($python,$capture,$review,$lock,$FrozenW04SourceProvenance,$Video)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required source or PRE-FROZEN lock missing: $path"
    }
}
if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
    throw "Original frozen OpenVINO model directory missing: $ModelDir"
}
if ((Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $packets)) {
    throw 'Immutable dual-lane and/or reviewer result already exists; never overwrite it.'
}
$oldPythonPath = $env:PYTHONPATH
try {
    if ($oldPythonPath) {
        $env:PYTHONPATH = $root + [System.IO.Path]::PathSeparator + $oldPythonPath
    } else {
        $env:PYTHONPATH = $root
    }
    Write-Host 'New-video experiment: same OpenVINO model; one inference pass; 2 true tracker instances.'
    Write-Host 'OpenVINO CPU source frames and model SHA will be verified. No FLUID scoring or promotion.'
    & $python $capture --video $Video --model-dir $ModelDir `
        --frozen-w04-source-provenance $FrozenW04SourceProvenance `
        --first-eval-frame $FirstEvalFrame --last-eval-frame $LastEvalFrame `
        --warmup-frames $WarmupFrames --output-dir $output
    if ($LASTEXITCODE -ne 0) {
        throw 'Holdout original-model two-lane tracking aborted; review packet NOT created.'
    }
    $primary = Join-Path $output 'primary_ios030.txt'
    $shadow = Join-Path $output 'shadow_rare_iou050.txt'
    & $python $review prepare --lock $lock --video $Video `
        --primary $primary --shadow $shadow --first $FirstEvalFrame `
        --last $LastEvalFrame --out $packets
    if ($LASTEXITCODE -ne 0) {
        throw 'Holdout blind review preparation aborted; source track exports remain immutable.'
    }
    Write-Host "Independent R01 packet: $(Join-Path $packets 'REVIEWER_R01_ONLY.zip')"
    Write-Host "Independent R02 packet: $(Join-Path $packets 'REVIEWER_R02_ONLY.zip')"
    Write-Host 'Keep INTERNAL_manifest.json separate; reviewer labels and tracker class are blinded.'
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
