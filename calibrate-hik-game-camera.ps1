param(
    [string]$GameId = "genshin-impact",
    [string]$CameraId,
    [string]$PhoneSerial,
    [string]$RigOutput,
    [string]$MinimapOutput,
    [string]$ProfilesRoot,
    [switch]$NoDemo
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $RigOutput) {
    $RigOutput = Join-Path $root "artifacts\hik-calibration-$timestamp"
}
if (-not $MinimapOutput) {
    $MinimapOutput = Join-Path $root "artifacts\game-minimap-calibration-$timestamp"
}
if (-not $ProfilesRoot) {
    $ProfilesRoot = Join-Path $root "profiles"
}

function Invoke-AriaStage {
    param(
        [string]$Label,
        [string]$Command,
        [string[]]$StageArguments
    )
    Write-Host ""
    Write-Host $Label -ForegroundColor Cyan
    & $Command @StageArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

try {
    $rigArguments = @("--headless", "--save", "--output", $RigOutput)
    if ($CameraId) { $rigArguments += @("--camera-id", $CameraId) }
    if ($PhoneSerial) { $rigArguments += @("--phone-serial", $PhoneSerial) }
    Invoke-AriaStage `
        "[1-2/6] Wake phone and calibrate the HIK rig" `
        (Join-Path $root "calibrate-hik-rig.bat") `
        $rigArguments

    $rigCalibration = Join-Path $RigOutput "hik_camera_calibration.json"
    if (-not (Test-Path -LiteralPath $rigCalibration -PathType Leaf)) {
        throw "Rig calibration did not produce $rigCalibration"
    }

    $minimapArguments = @(
        "--game-id", $GameId,
        "--rig-calibration", $RigOutput,
        "--calibration-output", $MinimapOutput,
        "--profiles-root", $ProfilesRoot
    )
    if ($CameraId) { $minimapArguments += @("--camera-id", $CameraId) }
    if ($PhoneSerial) { $minimapArguments += @("--phone-serial", $PhoneSerial) }
    Invoke-AriaStage `
        "[3-5/6] Launch game, capture the zigzag, and calibrate the mini-map" `
        (Join-Path $root "calibrate-game-minimap.bat") `
        $minimapArguments

    $summaryPath = Join-Path $MinimapOutput "calibration_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Mini-map calibration did not produce $summaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $rigGameProfile = [string]$summary.rig_game_profile
    if (-not $rigGameProfile -or -not (Test-Path -LiteralPath $rigGameProfile -PathType Leaf)) {
        throw "Mini-map calibration did not publish a usable rig-game profile"
    }

    Write-Host ""
    Write-Host "Complete calibration succeeded." -ForegroundColor Green
    Write-Host "Rig:      $rigCalibration"
    Write-Host "Mini-map: $summaryPath"
    Write-Host "Profile:  $rigGameProfile"

    if (-not $NoDemo) {
        Invoke-AriaStage `
            "[6/6] Run the calibrated HIK adapter demo (dual stream)" `
            (Join-Path $root "demo-hik-camera.bat") `
            @($RigOutput, "--minimap-calibration", $rigGameProfile, "--mode", "dual")
    }
    exit 0
}
catch {
    Write-Host ""
    Write-Host "Complete calibration stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Completed stage artifacts were retained for diagnosis."
    exit 1
}
