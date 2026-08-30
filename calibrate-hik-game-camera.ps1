param(
    [string]$GameId = "genshin-impact",
    [string]$CameraId,
    [string]$PhoneSerial,
    [string]$DiagnosticRigCalibrationOverride,
    [string]$RigOutput,
    [string]$MinimapOutput
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
$captureRoot = Join-Path $root "sessions\calibration"
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
    $precheckOutput = "$RigOutput-precheck"
    $precheckArguments = @(
        "--output", $precheckOutput
    )
    if ($DiagnosticRigCalibrationOverride) {
        $precheckArguments += @(
            "--diagnostic-calibration-override", $DiagnosticRigCalibrationOverride
        )
    }
    if ($CameraId) { $precheckArguments += @("--camera-id", $CameraId) }
    if ($PhoneSerial) { $precheckArguments += @("--phone-serial", $PhoneSerial) }
    Invoke-AriaStage `
        "[0/5] Check whether the saved HIK rig calibration is still valid" `
        (Join-Path $root "precheck-hik-rig.bat") `
        $precheckArguments

    $precheckPath = Join-Path $precheckOutput "precheck.json"
    if (-not (Test-Path -LiteralPath $precheckPath -PathType Leaf)) {
        throw "Rig precheck did not produce $precheckPath"
    }
    $precheck = Get-Content -LiteralPath $precheckPath -Raw | ConvertFrom-Json
    if ($precheck.reusable -and $precheck.camera_adapter_is_calibrated) {
        $effectiveRigCalibration = [string]$precheck.calibration
        New-Item -ItemType Directory -Path $RigOutput | Out-Null
        [ordered]@{
            schema_version = "1.0"
            status = "reused"
            calibration = $effectiveRigCalibration
            precheck = $precheckPath
            comparison = $precheck.comparison
        } | ConvertTo-Json -Depth 12 | Set-Content `
            -LiteralPath (Join-Path $RigOutput "reused_calibration.json") `
            -Encoding UTF8
        Write-Host "Saved rig calibration is unchanged; full rig calibration skipped." -ForegroundColor Green
    }
    else {
        Write-Host "Rig reuse was not proven ($($precheck.status)); running full calibration." -ForegroundColor Yellow
        $rigArguments = @("--headless", "--save", "--output", $RigOutput)
        if ($CameraId) { $rigArguments += @("--camera-id", $CameraId) }
        if ($PhoneSerial) { $rigArguments += @("--phone-serial", $PhoneSerial) }
        Invoke-AriaStage `
            "[1-2/5] Wake phone and calibrate the HIK rig" `
            (Join-Path $root "calibrate-hik-rig.bat") `
            $rigArguments
        $effectiveRigCalibration = Join-Path $RigOutput "hik_camera_calibration.json"
        if (-not (Test-Path -LiteralPath $effectiveRigCalibration -PathType Leaf)) {
            throw "Rig calibration did not produce $effectiveRigCalibration"
        }
    }

    New-Item -ItemType Directory -Force -Path $captureRoot | Out-Null
    $existingSessions = @{}
    Get-ChildItem -LiteralPath $captureRoot -Directory | ForEach-Object {
        $existingSessions[$_.FullName] = $true
    }
    $captureArguments = @(
        "--game-id", $GameId,
        "--output-root", $captureRoot
    )
    if ($DiagnosticRigCalibrationOverride) {
        $captureArguments += @(
            "--diagnostic-rig-calibration-override", $effectiveRigCalibration
        )
    }
    if ($CameraId) { $captureArguments += @("--camera-id", $CameraId) }
    if ($PhoneSerial) { $captureArguments += @("--phone-serial", $PhoneSerial) }
    Invoke-AriaStage `
        "[3-4/5] Launch game, run the zigzag, and retain dual-source audit evidence" `
        (Join-Path $root "capture-game-minimap-zigzag.bat") `
        $captureArguments

    $freshSessions = @(
        Get-ChildItem -LiteralPath $captureRoot -Directory |
        Where-Object { -not $existingSessions.ContainsKey($_.FullName) } |
        Sort-Object LastWriteTimeUtc -Descending
    )
    if ($freshSessions.Count -ne 1) {
        throw "Capture produced $($freshSessions.Count) new sessions; expected exactly one"
    }
    $sessionPath = $freshSessions[0].FullName

    $minimapArguments = @(
        $sessionPath,
        "--output", $MinimapOutput
    )
    Invoke-AriaStage `
        "[5/5] Discover the Android mini-map and project it through session registration" `
        (Join-Path $root "calibrate-game-minimap.bat") `
        $minimapArguments

    $summaryPath = Join-Path $MinimapOutput "localization_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Mini-map calibration did not produce $summaryPath"
    }
    Write-Host ""
    Write-Host "Fresh POC localization succeeded." -ForegroundColor Green
    Write-Host "Rig:      $effectiveRigCalibration"
    Write-Host "Session:  $sessionPath"
    Write-Host "Mini-map: $summaryPath"
    Write-Host "Profile:  not published by this POC-only localization stage"
    exit 0
}
catch {
    Write-Host ""
    Write-Host "Complete calibration stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Completed stage artifacts were retained for diagnosis."
    exit 1
}
