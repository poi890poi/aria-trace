[CmdletBinding()]
param(
    [string]$PythonCommand,
    [string]$OutputDirectory,
    [switch]$SkipDependencyInstall,
    [switch]$SkipApplicationBuild,
    [switch]$SkipPhoneTargetBuild,
    [switch]$NoArchive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolchainRoot = Join-Path $ProjectRoot ".tools\standalone-release-py31210"
$BuildRoot = Join-Path $ProjectRoot ".tools\standalone-release-pyinstaller"
$StageRoot = Join-Path $ProjectRoot ".tools\standalone-release-stage"
$DefaultPython = Join-Path $ProjectRoot ".tools\python-3.12.10\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-standalone-release.txt"

if (-not $PythonCommand) { $PythonCommand = $DefaultPython }
if (-not (Test-Path -LiteralPath $PythonCommand -PathType Leaf)) {
    throw "Python 3.12.10 was not found at $PythonCommand. Run setup-python-3.12.10.bat first."
}
$PythonVersion = & $PythonCommand -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -ne 0 -or $PythonVersion -ne "3.12.10") {
    throw "Standalone releases must be built with CPython 3.12.10; found $PythonVersion"
}

$EnvironmentPython = Join-Path $ToolchainRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    Write-Host "Creating the isolated Python 3.12.10 release environment"
    & $PythonCommand -m venv $ToolchainRoot
    if ($LASTEXITCODE -ne 0) { throw "Cannot create the release environment" }
}
$EnvironmentVersion = & $EnvironmentPython -c "import platform; print(platform.python_version())"
if ($EnvironmentVersion -ne "3.12.10") {
    throw "Release environment uses Python $EnvironmentVersion; remove $ToolchainRoot and rebuild"
}
if (-not $SkipDependencyInstall) {
    Write-Host "Installing pinned release dependencies in the isolated environment"
    & $EnvironmentPython -m pip install --disable-pip-version-check -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Cannot install release dependencies" }
}
& $EnvironmentPython -c "import cv2,numpy,yaml,zxingcpp,PyInstaller"
if ($LASTEXITCODE -ne 0) {
    throw "The release environment is incomplete; rerun without -SkipDependencyInstall"
}

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $ProjectRoot "artifacts\standalone-release\IRIS-Windows-x64"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$ExpectedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "artifacts"))
if (-not $OutputDirectory.StartsWith($ExpectedOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must remain below $ExpectedOutputRoot"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    Remove-Item -LiteralPath $OutputDirectory -Recurse -Force
}
if (-not $SkipApplicationBuild) {
    if (Test-Path -LiteralPath $StageRoot) {
        Remove-Item -LiteralPath $StageRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null
}

$Applications = @(
    [ordered]@{ Name = "iris-rig-calibration"; Entry = "packaging\windows\entrypoints\rig_calibration.py" },
    [ordered]@{ Name = "iris-zigzag-acquisition"; Entry = "packaging\windows\entrypoints\zigzag_acquisition.py" },
    [ordered]@{ Name = "iris-minimap-calibration"; Entry = "packaging\windows\entrypoints\minimap_calibration.py" },
    [ordered]@{ Name = "iris-game-calibration"; Entry = "packaging\windows\entrypoints\game_calibration.py" },
    [ordered]@{ Name = "iris-game-color-calibration"; Entry = "packaging\windows\entrypoints\game_color_calibration.py" }
)
if ($SkipApplicationBuild) {
    foreach ($Application in $Applications) {
        $StagedExecutable = Join-Path $StageRoot (
            "apps\{0}\{0}.exe" -f $Application.Name
        )
        if (-not (Test-Path -LiteralPath $StagedExecutable -PathType Leaf)) {
            throw "Cannot skip application build; staged executable is missing: $StagedExecutable"
        }
    }
}
else {
    $PyInstallerConfigRoot = Join-Path $BuildRoot "config"
    $PreviousPyInstallerConfig = $env:PYINSTALLER_CONFIG_DIR
    try {
        $env:PYINSTALLER_CONFIG_DIR = $PyInstallerConfigRoot
        foreach ($Application in $Applications) {
            $Name = $Application.Name
            $Entry = Join-Path $ProjectRoot $Application.Entry
            $Work = Join-Path $BuildRoot $Name
            Write-Host "Building $Name with Python 3.12.10"
            & $EnvironmentPython -m PyInstaller `
                --noconfirm `
                --clean `
                --onedir `
                --console `
                --noupx `
                --name $Name `
                --paths $ProjectRoot `
                --distpath (Join-Path $StageRoot "apps") `
                --workpath $Work `
                --specpath $Work `
                $Entry
            if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed for $Name" }
        }
    }
    finally {
        $env:PYINSTALLER_CONFIG_DIR = $PreviousPyInstallerConfig
    }
}

$ReleaseRoot = $OutputDirectory
New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $StageRoot "apps") -Destination $ReleaseRoot -Recurse

$PhoneTargetApk = Join-Path $ProjectRoot "artifacts\android-phone-target\iris-phone-target.apk"
if (-not $SkipPhoneTargetBuild) {
    & (Join-Path $ProjectRoot "android\phone-target\build-phone-target.ps1") -Output $PhoneTargetApk
    if ($LASTEXITCODE -ne 0) { throw "Native phone target build failed" }
}
if (-not (Test-Path -LiteralPath $PhoneTargetApk -PathType Leaf)) {
    throw "Native phone target APK is missing: $PhoneTargetApk"
}
$PhoneTargetRelease = Join-Path $ReleaseRoot "phone-target"
New-Item -ItemType Directory -Force -Path $PhoneTargetRelease | Out-Null
Copy-Item -LiteralPath $PhoneTargetApk -Destination (Join-Path $PhoneTargetRelease "iris-phone-target.apk")

$PythonSource = Join-Path $ReleaseRoot "python"
New-Item -ItemType Directory -Force -Path $PythonSource | Out-Null
$IrisExcludedSourcePaths = @(
    "aria_trace\adapters\windows.py",
    "aria_trace\apps\workbench",
    "aria_trace\apps\record.py",
    "aria_trace\apps\review.py",
    "aria_trace\apps\session_inspector.py",
    "aria_trace\evidence\poc_catalog.py",
    "aria_trace\evidence\tracking.py",
    "aria_trace\services\localization",
    "aria_trace\services\mapping",
    "aria_trace\services\tracking",
    "aria_trace\workflows\input_verification.py",
    "aria_trace\workflows\portal_initialization.py",
    "aria_trace\workflows\route.py",
    "aria_trace\workflows\teleport.py"
)
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "aria_trace") -Recurse -File -Filter "*.py" | ForEach-Object {
    $Relative = $_.FullName.Substring($ProjectRoot.Length + 1)
    $Excluded = $IrisExcludedSourcePaths | Where-Object {
        $Relative.Equals($_, [System.StringComparison]::OrdinalIgnoreCase) -or
        $Relative.StartsWith($_ + "\", [System.StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $Excluded) {
        $Destination = Join-Path $PythonSource $Relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $Destination
    }
}
New-Item -ItemType Directory -Force -Path (Join-Path $PythonSource "android") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "android\phone-target") -Destination (Join-Path $PythonSource "android\phone-target") -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRoot "hikcam.py") -Destination $PythonSource
Copy-Item -LiteralPath (Join-Path $ProjectRoot "iris_tools.py") -Destination $PythonSource
Copy-Item -LiteralPath (Join-Path $ProjectRoot "requirements-hik-camera-adapter.txt") -Destination $PythonSource
Copy-Item -LiteralPath (Join-Path $ProjectRoot "requirements-python-tools.txt") -Destination $PythonSource
$BuildSourceFiles = @(
    "build-standalone-release.bat",
    "build-standalone-release.ps1",
    "setup-python-3.12.10.bat",
    "setup-python-3.12.10.ps1",
    "requirements-standalone-release.txt"
)
foreach ($Name in $BuildSourceFiles) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination $PythonSource
}
New-Item -ItemType Directory -Force -Path (Join-Path $PythonSource "packaging") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\windows") -Destination (Join-Path $PythonSource "packaging") -Recurse
New-Item -ItemType Directory -Force -Path (Join-Path $PythonSource "docs") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\standalone-release") -Destination (Join-Path $PythonSource "docs") -Recurse

Copy-Item -Path (Join-Path $ProjectRoot "packaging\windows\release-files\*") -Destination $ReleaseRoot -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\standalone-release") -Destination (Join-Path $ReleaseRoot "docs") -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRoot "IRIS_README.md") -Destination (Join-Path $ReleaseRoot "README.md")
New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot "artifacts"),(Join-Path $ReleaseRoot "sessions"),(Join-Path $ReleaseRoot "profiles") | Out-Null

$EmbeddedVersions = & $EnvironmentPython -c "import cv2,numpy,yaml,zxingcpp,PyInstaller; print('|'.join([numpy.__version__,cv2.__version__,yaml.__version__,getattr(zxingcpp,'__version__','3.1.1'),PyInstaller.__version__]))"
$VersionParts = $EmbeddedVersions -split "\|"
$Manifest = @(
    "# Human-readable standalone release identity and external runtime contract.",
    "schema_version: '1.0'",
    "product: iris-invariant-rig-system",
    "platform: windows-x64",
    "python_build_version: '3.12.10'",
    "build_time_utc: '$([DateTime]::UtcNow.ToString("o"))'",
    "embedded_python_packages:",
    "  numpy: '$($VersionParts[0])'",
    "  opencv: '$($VersionParts[1])'",
    "  pyyaml: '$($VersionParts[2])'",
    "  zxing_cpp: '$($VersionParts[3])'",
    "  pyinstaller: '$($VersionParts[4])'",
    "executables:",
    "  - apps/iris-rig-calibration/iris-rig-calibration.exe",
    "  - apps/iris-zigzag-acquisition/iris-zigzag-acquisition.exe",
    "  - apps/iris-minimap-calibration/iris-minimap-calibration.exe",
    "  - apps/iris-game-calibration/iris-game-calibration.exe",
    "  - apps/iris-game-color-calibration/iris-game-color-calibration.exe",
    "camera_adapter_import: python/hikcam.py",
    "native_phone_target: phone-target/iris-phone-target.apk",
    "python_tools_import: python/iris_tools.py",
    "external_environment:",
    "  hik_mvs: required",
    "  adb: required_on_path_or_pass_explicit_path",
    "  scrcpy_server: required_for_android_capture_scrcpy_only",
    "  ffmpeg: required_for_android_capture_scrcpy_only",
    "bundled_scope: python_runtime_dependencies_source_and_native_phone_target"
)
Set-Content -LiteralPath (Join-Path $ReleaseRoot "release-manifest.yaml") -Value $Manifest -Encoding UTF8

Write-Host "Running offline command-surface smoke tests"
foreach ($Application in $Applications) {
    $Executable = Join-Path $ReleaseRoot ("apps\{0}\{0}.exe" -f $Application.Name)
    & $Executable --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$Executable --help failed" }
}

if (-not $NoArchive) {
    $Archive = "$OutputDirectory.zip"
    if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $Archive -CompressionLevel Optimal
    $HashStream = [System.IO.File]::OpenRead($Archive)
    try {
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            $ArchiveHash = ([System.BitConverter]::ToString($Hasher.ComputeHash($HashStream))).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $Hasher.Dispose()
        }
    }
    finally {
        $HashStream.Dispose()
    }
    Set-Content -LiteralPath "$Archive.sha256" -Value "$ArchiveHash *$([IO.Path]::GetFileName($Archive))" -Encoding ASCII
    Write-Host "Built archive: $Archive"
    Write-Host "SHA-256: $ArchiveHash"
}
Write-Host "Built standalone release: $OutputDirectory"
