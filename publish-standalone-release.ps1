[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Title,
    [switch]$Prerelease,
    [switch]$SkipBuild,
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$ReleaseRoot = Join-Path $ProjectRoot "artifacts\standalone-release"
$PackageRoot = Join-Path $ReleaseRoot "IRIS-Windows-x64"
$PackageArchive = "$PackageRoot.zip"
$SourceArchive = Join-Path $ReleaseRoot "IRIS-Third-Party-Source.zip"
$ManifestPath = Join-Path $PackageRoot "release-manifest.yaml"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    Write-Host ""
    Write-Host "=== $Description ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-PortableSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString(
                $Hasher.ComputeHash($Stream)
            )).Replace("-", "").ToLowerInvariant()
        }
        finally { $Hasher.Dispose() }
    }
    finally { $Stream.Dispose() }
}

function Assert-ArchiveHash {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Sidecar = "$Path.sha256"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Release asset is missing: $Path"
    }
    if (-not (Test-Path -LiteralPath $Sidecar -PathType Leaf)) {
        throw "Release checksum is missing: $Sidecar"
    }
    $Expected = ((Get-Content -LiteralPath $Sidecar -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
    $Actual = Get-PortableSha256 $Path
    if ($Expected -ne $Actual) {
        throw "Release checksum mismatch for $Path; expected $Expected, got $Actual"
    }
    Write-Host "Verified $([IO.Path]::GetFileName($Path)): $Actual"
}

if (-not $Title) { $Title = "IRIS $Tag" }

Invoke-Checked "Preflight source identity" {
    git -C $ProjectRoot diff --quiet --exit-code
    if ($LASTEXITCODE -ne 0) { throw "Tracked working-tree changes must be committed before publishing" }
    git -C $ProjectRoot diff --cached --quiet --exit-code
    if ($LASTEXITCODE -ne 0) { throw "Staged changes must be committed before publishing" }
    git -C $ProjectRoot rev-parse --verify '@{upstream}' 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "The current branch has no upstream" }
    $Ahead = [int](git -C $ProjectRoot rev-list --count '@{upstream}..HEAD')
    $Behind = [int](git -C $ProjectRoot rev-list --count 'HEAD..@{upstream}')
    if ($Ahead -ne 0 -or $Behind -ne 0) {
        throw "Source is not synchronized with its upstream (ahead=$Ahead, behind=$Behind); push or pull first"
    }
}

if (-not $SkipBuild) {
    Invoke-Checked "Build release from the pushed commit" {
        $Arguments = @()
        if ($SkipDependencyInstall) { $Arguments += "-SkipDependencyInstall" }
        & (Join-Path $ProjectRoot "build-standalone-release.bat") @Arguments
    }
}

Invoke-Checked "Validate package identity and checksums" {
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Release manifest is missing: $ManifestPath"
    }
    $Head = [string](git -C $ProjectRoot rev-parse HEAD)
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw
    if ($Manifest -notmatch "(?m)^source_commit:\s*'?$([regex]::Escape($Head))'?$" ) {
        throw "Package source_commit does not match HEAD $Head; rebuild before publishing"
    }
    Assert-ArchiveHash $PackageArchive
    Assert-ArchiveHash $SourceArchive
}

$Gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $Gh) {
    throw "GitHub CLI (gh) is required only for the upload stage. Install it, run 'gh auth login', then rerun this command with -SkipBuild."
}
Invoke-Checked "Verify GitHub authentication" { & $Gh.Source auth status }

$Head = [string](git -C $ProjectRoot rev-parse HEAD)
$ExistingTag = git -C $ProjectRoot rev-list -n 1 $Tag 2>$null
if ($LASTEXITCODE -eq 0 -and $ExistingTag -and ([string]$ExistingTag).Trim() -ne $Head.Trim()) {
    throw "Tag $Tag already points to $ExistingTag, not current HEAD $Head"
}
if (-not $ExistingTag) {
    Invoke-Checked "Create release tag $Tag" { git -C $ProjectRoot tag -a $Tag -m $Title $Head }
}
Invoke-Checked "Push release tag $Tag" { git -C $ProjectRoot push origin "refs/tags/$Tag" }

$Assets = @(
    $PackageArchive,
    "$PackageArchive.sha256",
    $SourceArchive,
    "$SourceArchive.sha256"
)
$ExistingRelease = $false
& $Gh.Source release view $Tag *> $null
if ($LASTEXITCODE -eq 0) { $ExistingRelease = $true }

if ($ExistingRelease) {
    Invoke-Checked "Resume release upload $Tag" {
        & $Gh.Source release upload $Tag @Assets --clobber
    }
}
else {
    Invoke-Checked "Create GitHub Release $Tag" {
        $Arguments = @("release", "create", $Tag) + $Assets + @("--title", $Title, "--generate-notes")
        if ($Prerelease) { $Arguments += "--prerelease" }
        & $Gh.Source @Arguments
    }
}

Write-Host ""
Write-Host "Release publication completed: $Tag" -ForegroundColor Green
