[CmdletBinding()]
param(
    [string]$Destination,
    [string]$Installer
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Destination) {
    $Destination = Join-Path $ProjectRoot ".tools\python-3.12.10"
}
if (-not $Installer) {
    $Installer = Join-Path $ProjectRoot ".tools\downloads\python-3.12.10-amd64.exe"
}
$Python = Join-Path $Destination "python.exe"

if (Test-Path -LiteralPath $Python) {
    $Version = & $Python -c "import platform; print(platform.python_version())"
    if ($Version -eq "3.12.10") {
        Write-Host "Python 3.12.10 is already installed at $Python"
        exit 0
    }
    throw "The destination contains Python $Version, not 3.12.10: $Destination"
}

$InstallerDirectory = Split-Path -Parent $Installer
New-Item -ItemType Directory -Force -Path $InstallerDirectory | Out-Null
if (-not (Test-Path -LiteralPath $Installer)) {
    Write-Host "Downloading the official CPython 3.12.10 Windows x64 installer"
    Invoke-WebRequest `
        -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" `
        -OutFile $Installer
}

Write-Host "Installing a private Python 3.12.10 build toolchain at $Destination"
$Process = Start-Process `
    -FilePath $Installer `
    -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "Include_launcher=0",
        "Include_test=0",
        "PrependPath=0",
        "Shortcuts=0",
        "TargetDir=$Destination"
    ) `
    -Wait `
    -PassThru `
    -WindowStyle Hidden
if ($Process.ExitCode -ne 0) {
    throw "CPython 3.12.10 installer failed with exit code $($Process.ExitCode)"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "CPython installation completed without $Python"
}
$InstalledVersion = & $Python -c "import platform; print(platform.python_version())"
if ($InstalledVersion -ne "3.12.10") {
    throw "Expected Python 3.12.10, found $InstalledVersion"
}
Write-Host "Installed Python 3.12.10 at $Python"
