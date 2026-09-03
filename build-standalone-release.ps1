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
$ThirdPartyCache = Join-Path $ProjectRoot ".tools\third-party-release-cache"
$DefaultPython = Join-Path $ProjectRoot ".tools\python-3.12.10\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-standalone-release.txt"

$ScrcpyVersion = "4.1"
$ScrcpyArchiveName = "scrcpy-win64-v4.1.zip"
$ScrcpyArchiveUrl = "https://github.com/Genymobile/scrcpy/releases/download/v4.1/scrcpy-win64-v4.1.zip"
$ScrcpyArchiveSha256 = "5b12172b3264b2889f4583ee64752ce832e29bc8b1089dca81093459697165db"
$ScrcpySourceName = "scrcpy-v4.1-source.tar.gz"
$ScrcpySourceUrl = "https://github.com/Genymobile/scrcpy/archive/refs/tags/v4.1.tar.gz"
$ScrcpySourceSha256 = "537b2ade623cb94b6edddfa5c61bf0b0af21484aa8365ea2531b686ea573249a"

$FfmpegVersion = "n9.0.1-6-g9d4ca21220-20260823"
$FfmpegArchiveName = "ffmpeg-n9.0.1-6-g9d4ca21220-win64-lgpl-9.0.zip"
$FfmpegArchiveUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-23-13-03/$FfmpegArchiveName"
$FfmpegArchiveSha256 = "96ee3965c8f8ba3210e59374c8b1c58f7c9552ea877d930f3fb63fac94fefcec"
$FfmpegSourceName = "FFmpeg-9d4ca21220-source.tar.gz"
$FfmpegSourceUrl = "https://github.com/FFmpeg/FFmpeg/archive/9d4ca21220.tar.gz"
$FfmpegSourceSha256 = "693b19b88c741aa9e02566a40c9c090ca64f808aefdd7fcbff8e5d8cc42db04f"
$FfmpegBuildSourceName = "FFmpeg-Builds-48576f197ad1c2afb2e0b8efe204919a1afbff54-source.tar.gz"
$FfmpegBuildSourceUrl = "https://github.com/BtbN/FFmpeg-Builds/archive/48576f197ad1c2afb2e0b8efe204919a1afbff54.tar.gz"
$FfmpegBuildSourceSha256 = "04b2fcde9a02e2d42c8cb69fe43b912f127746dbc859118b0f81cd124971f8a4"
$PhoneTargetSignerSha256 = "b7ce8a1a7953520d646ccfc21ae37a21af7a3dcca9893fa6a78f3af30943d286"

function Get-PortableSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $HashStream = [System.IO.File]::OpenRead($Path)
    try {
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString(
                $Hasher.ComputeHash($HashStream)
            )).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $Hasher.Dispose()
        }
    }
    finally {
        $HashStream.Dispose()
    }
}

function Get-PinnedThirdPartyFile {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Sha256
    )
    New-Item -ItemType Directory -Force -Path $ThirdPartyCache | Out-Null
    $Destination = Join-Path $ThirdPartyCache $Name
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        Write-Host "Downloading pinned third-party archive $Name"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
    }
    $Actual = Get-PortableSha256 $Destination
    if ($Actual -ne $Sha256) {
        throw "Third-party archive checksum mismatch for $Name; expected $Sha256, got $Actual"
    }
    return $Destination
}

function Write-Sha256Sidecar {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Hash = Get-PortableSha256 $Path
    Set-Content -LiteralPath "$Path.sha256" -Value "$Hash *$([IO.Path]::GetFileName($Path))" -Encoding ASCII
    return $Hash
}

function Get-PortableTreeSha256 {
    param([Parameter(Mandatory = $true)][string]$Root)
    $AbsoluteRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $Records = Get-ChildItem -LiteralPath $AbsoluteRoot -Recurse -File -Force |
        Sort-Object FullName |
        ForEach-Object {
            $Relative = $_.FullName.Substring($AbsoluteRoot.Length + 1).Replace('\', '/')
            "$Relative`t$(Get-PortableSha256 $_.FullName)"
        }
    $Payload = [System.Text.Encoding]::UTF8.GetBytes(($Records -join "`n"))
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString(
            $Hasher.ComputeHash($Payload)
        )).Replace("-", "").ToLowerInvariant()
    }
    finally { $Hasher.Dispose() }
}

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
    [ordered]@{ Name = "iris-game-color-calibration"; Entry = "packaging\windows\entrypoints\game_color_calibration.py" },
    [ordered]@{ Name = "iris-camera-adapter-demo"; Entry = "packaging\windows\entrypoints\camera_adapter_demo.py" }
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

$ScrcpyArchive = Get-PinnedThirdPartyFile $ScrcpyArchiveName $ScrcpyArchiveUrl $ScrcpyArchiveSha256
$FfmpegArchive = Get-PinnedThirdPartyFile $FfmpegArchiveName $FfmpegArchiveUrl $FfmpegArchiveSha256
$ScrcpySource = Get-PinnedThirdPartyFile $ScrcpySourceName $ScrcpySourceUrl $ScrcpySourceSha256
$FfmpegSource = Get-PinnedThirdPartyFile $FfmpegSourceName $FfmpegSourceUrl $FfmpegSourceSha256
$FfmpegBuildSource = Get-PinnedThirdPartyFile $FfmpegBuildSourceName $FfmpegBuildSourceUrl $FfmpegBuildSourceSha256

$ThirdPartyRoot = Join-Path $ReleaseRoot "third_party"
$ThirdPartyExtract = Join-Path $BuildRoot "third-party-extract"
if (Test-Path -LiteralPath $ThirdPartyExtract) {
    Remove-Item -LiteralPath $ThirdPartyExtract -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ThirdPartyExtract,$ThirdPartyRoot | Out-Null

$ScrcpyExtract = Join-Path $ThirdPartyExtract "scrcpy"
Expand-Archive -LiteralPath $ScrcpyArchive -DestinationPath $ScrcpyExtract
$ScrcpyPackageRoot = Get-ChildItem -LiteralPath $ScrcpyExtract -Directory | Select-Object -First 1
if (-not $ScrcpyPackageRoot) { throw "Official scrcpy archive has no package root" }
$ScrcpyReleaseRoot = Join-Path $ThirdPartyRoot "scrcpy"
New-Item -ItemType Directory -Force -Path $ScrcpyReleaseRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $ScrcpyPackageRoot.FullName "scrcpy-server") -Destination $ScrcpyReleaseRoot
Copy-Item -LiteralPath (Join-Path $ScrcpyPackageRoot.FullName "LICENSE.txt") -Destination (Join-Path $ScrcpyReleaseRoot "LICENSE.txt")

$FfmpegExtract = Join-Path $ThirdPartyExtract "ffmpeg"
Expand-Archive -LiteralPath $FfmpegArchive -DestinationPath $FfmpegExtract
$FfmpegPackageRoot = Get-ChildItem -LiteralPath $FfmpegExtract -Directory | Select-Object -First 1
if (-not $FfmpegPackageRoot) { throw "Pinned FFmpeg archive has no package root" }
$FfmpegReleaseRoot = Join-Path $ThirdPartyRoot "ffmpeg"
New-Item -ItemType Directory -Force -Path (Join-Path $FfmpegReleaseRoot "bin") | Out-Null
Copy-Item -LiteralPath (Join-Path $FfmpegPackageRoot.FullName "bin\ffmpeg.exe") -Destination (Join-Path $FfmpegReleaseRoot "bin\ffmpeg.exe")
Copy-Item -LiteralPath (Join-Path $FfmpegPackageRoot.FullName "LICENSE.txt") -Destination (Join-Path $FfmpegReleaseRoot "LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\windows\release-files\THIRD-PARTY-NOTICES.md") -Destination $ThirdPartyRoot

$PhoneTargetApk = Join-Path $ProjectRoot "artifacts\android-phone-target\iris-phone-target.apk"
if (-not $SkipPhoneTargetBuild) {
    & (Join-Path $ProjectRoot "android\phone-target\build-phone-target.ps1") `
        -Output $PhoneTargetApk `
        -ExpectedSignerSha256 $PhoneTargetSignerSha256
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
$RigRuntimeSource = Join-Path $ProjectRoot "rig_runtime"
if (-not (Test-Path -LiteralPath $RigRuntimeSource -PathType Container)) {
    throw "Neutral IRIS runtime package is missing: $RigRuntimeSource"
}
Get-ChildItem -LiteralPath $RigRuntimeSource -Recurse -File -Filter "*.py" | ForEach-Object {
    $Relative = $_.FullName.Substring($ProjectRoot.Length + 1)
    $Destination = Join-Path $PythonSource $Relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $Destination
}
New-Item -ItemType Directory -Force -Path (Join-Path $PythonSource "android") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "android\phone-target") -Destination (Join-Path $PythonSource "android\phone-target") -Recurse
$PythonPhoneTarget = Join-Path $PythonSource "android\phone-target"
Get-ChildItem -LiteralPath $PythonPhoneTarget -Filter "*.apk" -File | Remove-Item -Force
Copy-Item -LiteralPath $PhoneTargetApk -Destination (Join-Path $PythonPhoneTarget "iris-phone-target.apk")
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

$GitMetadata = Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -Force | Where-Object {
    $_.Name -eq ".git" -or $_.Name -eq ".gitmodules"
}
if ($GitMetadata) {
    throw "Release export contains forbidden Git metadata: $($GitMetadata.FullName -join ', ')"
}
$PythonSourceTreeSha256 = Get-PortableTreeSha256 $PythonSource

$EmbeddedVersions = & $EnvironmentPython -c "import cv2,numpy,yaml,zxingcpp,PyInstaller; print('|'.join([numpy.__version__,cv2.__version__,yaml.__version__,getattr(zxingcpp,'__version__','3.1.1'),PyInstaller.__version__]))"
$VersionParts = $EmbeddedVersions -split "\|"
$SourceCommit = "unavailable"
$GitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($GitCommand) {
    $CandidateCommit = & $GitCommand.Source -C $ProjectRoot rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $CandidateCommit) {
        $SourceCommit = [string]$CandidateCommit
    }
}
$Manifest = @(
    "# Human-readable standalone release identity and external runtime contract.",
    "schema_version: '1.3'",
    "product: iris-invariant-rig-system",
    "platform: windows-x64",
    "source_commit: '$SourceCommit'",
    "exported_python_source_tree_sha256: '$PythonSourceTreeSha256'",
    "git_metadata_included: false",
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
    "  - apps/iris-camera-adapter-demo/iris-camera-adapter-demo.exe",
    "camera_adapter_import: python/hikcam.py",
    "native_phone_target: phone-target/iris-phone-target.apk",
    "python_native_phone_target: python/android/phone-target/iris-phone-target.apk",
    "native_phone_target_signer_sha256: '$PhoneTargetSignerSha256'",
    "python_tools_import: python/iris_tools.py",
    "bundled_tools:",
    "  scrcpy_server:",
    "    version: '$ScrcpyVersion'",
    "    path: third_party/scrcpy/scrcpy-server",
    "    license: Apache-2.0",
    "    upstream_archive_sha256: '$ScrcpyArchiveSha256'",
    "  ffmpeg:",
    "    version: '$FfmpegVersion'",
    "    path: third_party/ffmpeg/bin/ffmpeg.exe",
    "    build_variant: win64-lgpl-static",
    "    license: LGPL-2.1-or-later",
    "    upstream_archive_sha256: '$FfmpegArchiveSha256'",
    "external_environment:",
    "  hik_mvs: required",
    "  adb: required_on_path_or_pass_explicit_path",
    "  scrcpy_server: bundled_for_android_capture_scrcpy_only",
    "  ffmpeg: bundled_for_android_capture_scrcpy_only",
    "third_party_corresponding_source: ../IRIS-Third-Party-Source.zip",
    "bundled_scope: python_runtime_dependencies_source_native_phone_target_scrcpy_server_and_ffmpeg"
)
Set-Content -LiteralPath (Join-Path $ReleaseRoot "release-manifest.yaml") -Value $Manifest -Encoding UTF8

$ReleasePhoneTarget = Join-Path $ReleaseRoot "phone-target\iris-phone-target.apk"
$PythonReleasePhoneTarget = Join-Path $ReleaseRoot "python\android\phone-target\iris-phone-target.apk"
if (-not (Test-Path -LiteralPath $ReleasePhoneTarget -PathType Leaf)) {
    throw "Release phone-target APK is missing: $ReleasePhoneTarget"
}
if (-not (Test-Path -LiteralPath $PythonReleasePhoneTarget -PathType Leaf)) {
    throw "Pure-Python phone-target APK is missing: $PythonReleasePhoneTarget"
}

Write-Host "Running offline command-surface smoke tests"
foreach ($Application in $Applications) {
    $Executable = Join-Path $ReleaseRoot ("apps\{0}\{0}.exe" -f $Application.Name)
    & $Executable --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$Executable --help failed" }
}

if (-not $NoArchive) {
    $Archive = "$OutputDirectory.zip"
    $PendingArchive = "$Archive.pending-$PID.zip"
    if (Test-Path -LiteralPath $PendingArchive) { Remove-Item -LiteralPath $PendingArchive -Force }
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $PendingArchive -CompressionLevel Optimal
    Move-Item -LiteralPath $PendingArchive -Destination $Archive -Force
    $ArchiveHash = Write-Sha256Sidecar $Archive

    $SourceStage = Join-Path $BuildRoot "third-party-source"
    if (Test-Path -LiteralPath $SourceStage) { Remove-Item -LiteralPath $SourceStage -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $SourceStage | Out-Null
    Copy-Item -LiteralPath $ScrcpySource,$FfmpegSource,$FfmpegBuildSource -Destination $SourceStage
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\windows\release-files\THIRD-PARTY-NOTICES.md") -Destination $SourceStage
    $SourceReadme = @(
        "IRIS bundled third-party corresponding source",
        "",
        "This archive accompanies the separately aggregated scrcpy server and FFmpeg executable in IRIS-Windows-x64.zip.",
        "",
        "scrcpy v$ScrcpyVersion source: $ScrcpySourceName",
        "scrcpy source SHA-256: $ScrcpySourceSha256",
        "FFmpeg $FfmpegVersion source: $FfmpegSourceName",
        "FFmpeg source SHA-256: $FfmpegSourceSha256",
        "BtbN build recipe commit 48576f197ad1c2afb2e0b8efe204919a1afbff54: $FfmpegBuildSourceName",
        "BtbN build source SHA-256: $FfmpegBuildSourceSha256",
        "",
        "The exact FFmpeg configure line is available from: third_party/ffmpeg/bin/ffmpeg.exe -version",
        "The BtbN build recipes pin and fetch the LGPL-compatible dependency sources used for the build."
    )
    Set-Content -LiteralPath (Join-Path $SourceStage "README.txt") -Value $SourceReadme -Encoding UTF8
    $SourceArchive = Join-Path (Split-Path -Parent $OutputDirectory) "IRIS-Third-Party-Source.zip"
    $PendingSourceArchive = "$SourceArchive.pending-$PID.zip"
    if (Test-Path -LiteralPath $PendingSourceArchive) { Remove-Item -LiteralPath $PendingSourceArchive -Force }
    Compress-Archive -Path (Join-Path $SourceStage "*") -DestinationPath $PendingSourceArchive -CompressionLevel Optimal
    Move-Item -LiteralPath $PendingSourceArchive -Destination $SourceArchive -Force
    $SourceArchiveHash = Write-Sha256Sidecar $SourceArchive
    Write-Host "Built archive: $Archive"
    Write-Host "SHA-256: $ArchiveHash"
    Write-Host "Built corresponding-source archive: $SourceArchive"
    Write-Host "Source SHA-256: $SourceArchiveHash"
}
Write-Host "Built standalone release: $OutputDirectory"
