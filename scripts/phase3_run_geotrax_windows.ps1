[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StudyId,

    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [Parameter(Mandatory = $true)]
    [string]$FluidTracks,

    [ValidateSet("Smoke", "Full")]
    [string]$Mode = "Smoke",

    [int]$SmokeFrames = 600,

    [double]$MaxPixelDistance = 50.0,

    [string]$GeoTraxVenv = ".venv-geotrax",

    [string]$OutputRoot = "artifacts\phase3\geotrax",

    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$VideoFull = (Resolve-Path $VideoPath).Path
$FluidFull = (Resolve-Path $FluidTracks).Path

if (-not $SkipSetup) {
    $SetupScript = Join-Path $PSScriptRoot "phase3_setup_geotrax_windows.ps1"
    & $SetupScript -VenvPath $GeoTraxVenv
}

$GeoTraxExe = Join-Path (Join-Path $RepoRoot $GeoTraxVenv) "Scripts\geotrax.exe"
if (-not (Test-Path $GeoTraxExe)) {
    throw "Geo-trax is not installed. Run scripts\phase3_setup_geotrax_windows.ps1 first."
}

$StreetLabPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $StreetLabPython)) {
    $StreetLabPython = (Get-Command python.exe -ErrorAction Stop).Source
}

& $StreetLabPython -c "import numpy, scipy; import streetlab_phase3" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "StreetLab Python environment is missing project dependencies. Activate/install the StreetLab .venv first."
}

$ModeName = $Mode.ToLowerInvariant()
$OutputDir = Join-Path $RepoRoot (Join-Path $OutputRoot (Join-Path $StudyId $ModeName))
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$GeoArgs = @(
    "batch",
    $VideoFull,
    "--no-geo",
    "--output-folder",
    $OutputDir,
    "-c",
    "default"
)

if ($Mode -eq "Smoke") {
    if ($SmokeFrames -lt 1) {
        throw "SmokeFrames must be positive."
    }
    $GeoArgs += @("--cut-frame-right", "$SmokeFrames")
}

Write-Host ""
Write-Host "StreetLab P3B raw-video extraction"
Write-Host "Study: $StudyId"
Write-Host "Mode: $Mode"
Write-Host "Video: $VideoFull"
Write-Host "Output: $OutputDir"
if ($Mode -eq "Smoke") {
    Write-Host "Frame limit: $SmokeFrames"
}
Write-Host ""

& $GeoTraxExe @GeoArgs
if ($LASTEXITCODE -ne 0) {
    throw "Geo-trax extraction failed with exit code $LASTEXITCODE"
}

$Stem = [System.IO.Path]::GetFileNameWithoutExtension($VideoFull)
$ExpectedTrack = Join-Path $OutputDir "$Stem.txt"
if (Test-Path $ExpectedTrack) {
    $TrackFile = $ExpectedTrack
}
else {
    $Candidate = Get-ChildItem -Path $OutputDir -Recurse -Filter "$Stem.txt" | Select-Object -First 1
    if (-not $Candidate) {
        throw "Geo-trax completed but no '$Stem.txt' track file was found under $OutputDir"
    }
    $TrackFile = $Candidate.FullName
}

$Scorecard = Join-Path $OutputDir "pixel_scorecard.json"
$CompareScript = Join-Path $RepoRoot "scripts\phase3_compare_geotrax_fluid.py"

Write-Host ""
Write-Host "Comparing Geo-trax extraction against FLUID pixel ground truth..."

$CompareArgs = @(
    $CompareScript,
    "--geotrax-tracks", $TrackFile,
    "--fluid-tracks", $FluidFull,
    "--output", $Scorecard,
    "--max-pixel-distance", "$MaxPixelDistance"
)

& $StreetLabPython @CompareArgs
if ($LASTEXITCODE -ne 0) {
    throw "StreetLab FLUID comparison failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "P3B run complete."
Write-Host "Geo-trax tracks: $TrackFile"
Write-Host "Scorecard: $Scorecard"
Write-Host ""
if ($Mode -eq "Smoke") {
    Write-Host "If the smoke scorecard is sensible, rerun the exact command with -Mode Full."
}
