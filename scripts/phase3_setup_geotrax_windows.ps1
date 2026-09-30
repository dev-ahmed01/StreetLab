[CmdletBinding()]
param(
    [string]$VenvPath = ".venv-geotrax",
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvFull = Join-Path $RepoRoot $VenvPath
$GeoTraxVersion = "1.5.1"

Write-Host "StreetLab P3B - Geo-trax setup"
Write-Host "Repo: $RepoRoot"
Write-Host "External venv: $VenvFull"

if (-not (Test-Path $VenvFull)) {
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        Write-Host "Creating Python $PythonVersion environment..."
        & py "-$PythonVersion" -m venv $VenvFull
    }
    else {
        $Python = Get-Command python.exe -ErrorAction Stop
        $Actual = & $Python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($Actual -notin @("3.11", "3.12", "3.13")) {
            throw "Geo-trax needs Python 3.11-3.13. Found Python $Actual."
        }
        & $Python.Source -m venv $VenvFull
    }
}

$PythonExe = Join-Path $VenvFull "Scripts\python.exe"
$GeoTraxExe = Join-Path $VenvFull "Scripts\geotrax.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Geo-trax Python environment was not created correctly: $PythonExe"
}

Write-Host "Updating pip..."
& $PythonExe -m pip install --upgrade pip

Write-Host "Installing pinned Geo-trax $GeoTraxVersion..."
& $PythonExe -m pip install "geo-trax==1.5.1"

$Installed = & $PythonExe -c "import geotrax; print(geotrax.__version__)"
if ($Installed.Trim() -ne $GeoTraxVersion) {
    throw "Expected Geo-trax $GeoTraxVersion but found $Installed"
}

if (-not (Test-Path $GeoTraxExe)) {
    throw "Geo-trax executable not found: $GeoTraxExe"
}

Write-Host ""
Write-Host "Geo-trax $Installed is ready."
Write-Host "Executable: $GeoTraxExe"
Write-Host "The detector model downloads automatically on first extraction."
