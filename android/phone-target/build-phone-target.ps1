[CmdletBinding()]
param(
    [string]$AndroidSdk = $env:ANDROID_SDK_ROOT,
    [string]$Output = "",
    [string]$BuildDirectory = ""
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repository = Split-Path -Parent (Split-Path -Parent $Project)
if (-not $AndroidSdk) { $AndroidSdk = $env:ANDROID_HOME }
if (-not $AndroidSdk -and (Test-Path -LiteralPath "E:\Android\Sdk")) {
    $AndroidSdk = "E:\Android\Sdk"
}
if (-not $AndroidSdk) { throw "Set ANDROID_SDK_ROOT or pass -AndroidSdk" }
$Platform = Get-ChildItem -LiteralPath (Join-Path $AndroidSdk "platforms") -Directory |
    Sort-Object { [int]($_.Name -replace '[^0-9]', '') } -Descending |
    Select-Object -First 1
$BuildTools = Get-ChildItem -LiteralPath (Join-Path $AndroidSdk "build-tools") -Directory |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if (-not $Platform -or -not $BuildTools) { throw "Android platform/build-tools are missing" }
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $Repository ".tools\phone-target-build" }
if (-not $Output) { $Output = Join-Path $Repository "artifacts\android-phone-target\aria-phone-target.apk" }
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
$Output = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $BuildDirectory) { Remove-Item -LiteralPath $BuildDirectory -Recurse -Force }
New-Item -ItemType Directory -Force -Path $BuildDirectory,(Split-Path -Parent $Output) | Out-Null
$Classes = Join-Path $BuildDirectory "classes"
$Resources = Join-Path $BuildDirectory "resources"
$Dex = Join-Path $BuildDirectory "dex"
New-Item -ItemType Directory -Force -Path $Classes,$Resources,$Dex | Out-Null
$AndroidJar = Join-Path $Platform.FullName "android.jar"
$Aapt2 = Join-Path $BuildTools.FullName "aapt2.exe"
$Aapt = Join-Path $BuildTools.FullName "aapt.exe"
$D8 = Join-Path $BuildTools.FullName "d8.bat"
$ZipAlign = Join-Path $BuildTools.FullName "zipalign.exe"
$Signer = Join-Path $BuildTools.FullName "apksigner.bat"

& $Aapt2 compile --dir (Join-Path $Project "res") -o $Resources
if ($LASTEXITCODE -ne 0) { throw "aapt2 resource compilation failed" }
$CompiledResources = Get-ChildItem -LiteralPath $Resources -File | Select-Object -ExpandProperty FullName
$Unsigned = Join-Path $BuildDirectory "unsigned.apk"
& $Aapt2 link -o $Unsigned -I $AndroidJar --manifest (Join-Path $Project "AndroidManifest.xml") $CompiledResources
if ($LASTEXITCODE -ne 0) { throw "aapt2 link failed" }
& javac -encoding UTF-8 --release 8 -classpath $AndroidJar -d $Classes `
    (Join-Path $Project "src\io\ariatrace\phonetarget\PhoneTargetActivity.java")
if ($LASTEXITCODE -ne 0) { throw "javac failed" }
$ClassJar = Join-Path $BuildDirectory "classes.jar"
& jar cf $ClassJar -C $Classes .
& $D8 --lib $AndroidJar --output $Dex $ClassJar
if ($LASTEXITCODE -ne 0) { throw "d8 failed" }
Push-Location $BuildDirectory
try { & $Aapt add $Unsigned "dex/classes.dex" | Out-Null } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "Cannot add classes.dex" }
$Aligned = Join-Path $BuildDirectory "aligned.apk"
& $ZipAlign -f 4 $Unsigned $Aligned
if ($LASTEXITCODE -ne 0) { throw "zipalign failed" }
$KeyStore = Join-Path $Repository ".tools\phone-target-debug.keystore"
if (-not (Test-Path -LiteralPath $KeyStore)) {
    & keytool -genkeypair -keystore $KeyStore -storepass android -keypass android `
        -alias androiddebugkey -dname "CN=AriaTrace Local Target,O=AriaTrace,C=TW" `
        -keyalg RSA -keysize 2048 -validity 10000 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Cannot create local signing key" }
}
Copy-Item -LiteralPath $Aligned -Destination $Output -Force
& $Signer sign --ks $KeyStore --ks-pass pass:android --key-pass pass:android --out $Output $Aligned
if ($LASTEXITCODE -ne 0) { throw "APK signing failed" }
& $Signer verify --verbose $Output | Out-Null
if ($LASTEXITCODE -ne 0) { throw "APK verification failed" }
Write-Host "Built native phone target: $Output"
