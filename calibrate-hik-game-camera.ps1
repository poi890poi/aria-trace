param(
    [string]$GameId = "genshin-impact",
    [string]$CameraId,
    [string]$PhoneSerial,
    [string]$ProfileRoot,
    [string]$RigOutput
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $RigOutput) {
    $RigOutput = Join-Path $root "artifacts\hik-calibration-$timestamp"
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
    $rigArguments = @(
        "--reuse-if-unchanged",
        "--headless",
        "--save",
        "--output", $RigOutput
    )
    if ($CameraId) { $rigArguments += @("--camera-id", $CameraId) }
    if ($PhoneSerial) { $rigArguments += @("--phone-serial", $PhoneSerial) }
    if ($ProfileRoot) { $rigArguments += @("--profile-root", $ProfileRoot) }
    Invoke-AriaStage `
        "[1-2/4] Reuse or calibrate the HIK rig through the profile registry" `
        (Join-Path $root "calibrate-hik-rig.bat") `
        $rigArguments

    New-Item -ItemType Directory -Force -Path $captureRoot | Out-Null
    $existingSessions = @{}
    Get-ChildItem -LiteralPath $captureRoot -Directory | ForEach-Object {
        $existingSessions[$_.FullName] = $true
    }
    $captureArguments = @(
        "--game-id", $GameId,
        "--output-root", $captureRoot
    )
    if ($CameraId) { $captureArguments += @("--camera-id", $CameraId) }
    if ($PhoneSerial) { $captureArguments += @("--phone-serial", $PhoneSerial) }
    if ($ProfileRoot) { $captureArguments += @("--profile-root", $ProfileRoot) }
    Invoke-AriaStage `
        "[3-4/4] Launch game and retain dual-source source data" `
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

    Write-Host ""
    Write-Host "Rig calibration and source capture succeeded." -ForegroundColor Green
    Write-Host "Rig:      active profile registry revision"
    Write-Host "Session:  $sessionPath"
    Write-Host "Mini-map: not run; the caller must select and feed image frames"
    exit 0
}
catch {
    Write-Host ""
    Write-Host "Complete calibration stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Completed stage artifacts were retained for diagnosis."
    exit 1
}
