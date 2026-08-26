[CmdletBinding()]
param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentRoot = Join-Path $ProjectRoot ".tools\rig-calibration-app-build"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-rig-calibration-app.txt"
$Spec = Join-Path $ProjectRoot "packaging\rig_calibration_app.spec"
$DistributionRoot = Join-Path $ProjectRoot "artifacts\rig-calibration-app\windows"
$WorkRoot = Join-Path $ProjectRoot ".tools\rig-calibration-app-pyinstaller"
$PyInstallerConfigRoot = Join-Path $ProjectRoot ".tools\rig-calibration-app-pyinstaller-config"

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    Write-Host "Creating isolated build environment at $EnvironmentRoot"
    & $PythonCommand -m venv $EnvironmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Cannot create the isolated build environment" }
}

Write-Host "Installing build dependencies only inside the isolated environment"
& $EnvironmentPython -m pip install --disable-pip-version-check -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Cannot install rig-calibration app dependencies" }

Write-Host "Building one-folder Windows application"
$PreviousPyInstallerConfig = $env:PYINSTALLER_CONFIG_DIR
try {
    $env:PYINSTALLER_CONFIG_DIR = $PyInstallerConfigRoot
    & $EnvironmentPython -m PyInstaller `
        --noconfirm `
        --distpath $DistributionRoot `
        --workpath $WorkRoot `
        $Spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
}
finally {
    $env:PYINSTALLER_CONFIG_DIR = $PreviousPyInstallerConfig
}

$Executable = Join-Path $DistributionRoot "AriaTraceRigCalibration\AriaTraceRigCalibration.exe"
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Build completed without the expected executable: $Executable"
}

Write-Host "Built $Executable"
