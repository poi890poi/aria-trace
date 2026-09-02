[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Title,
    [switch]$Prerelease,
    [switch]$SkipBuild,
    [switch]$SkipDependencyInstall,
    [switch]$ValidateOnly
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
    $CommitLine = Get-Content -LiteralPath $ManifestPath | Where-Object {
        $_ -match "^source_commit\s*:"
    } | Select-Object -First 1
    if (-not $CommitLine) {
        throw "Package manifest has no source_commit field"
    }
    $ManifestCommit = (($CommitLine -split ":", 2)[1]).Trim().Trim("'", '"')
    if ($ManifestCommit -ne $Head.Trim()) {
        throw "Package source_commit $ManifestCommit does not match HEAD $Head; rebuild before publishing"
    }
    $TreeLine = Get-Content -LiteralPath $ManifestPath | Where-Object {
        $_ -match "^exported_python_source_tree_sha256\s*:"
    } | Select-Object -First 1
    if (-not $TreeLine) {
        throw "Package manifest has no exported Python source-tree hash"
    }
    $ExpectedTreeHash = (($TreeLine -split ":", 2)[1]).Trim().Trim("'", '"')
    $ActualTreeHash = Get-PortableTreeSha256 (Join-Path $PackageRoot "python")
    if ($ExpectedTreeHash -ne $ActualTreeHash) {
        throw "Exported Python source-tree hash mismatch; rebuild the package"
    }
    $GitMetadata = Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force | Where-Object {
        $_.Name -eq ".git" -or $_.Name -eq ".gitmodules"
    }
    if ($GitMetadata) {
        throw "Release package contains forbidden Git metadata: $($GitMetadata.FullName -join ', ')"
    }
    Assert-ArchiveHash $PackageArchive
    Assert-ArchiveHash $SourceArchive
}

if ($ValidateOnly) {
    Write-Host ""
    Write-Host "Release validation completed; no tag or release was changed." -ForegroundColor Green
    exit 0
}

$Gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $Gh) {
    throw "GitHub CLI (gh) is required only for the upload stage. Install it, run 'gh auth login', then rerun this command with -SkipBuild."
}
Invoke-Checked "Verify GitHub authentication" { & $Gh.Source auth status }

$Head = [string](git -C $ProjectRoot rev-parse HEAD)
$PreviousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
try {
    git -C $ProjectRoot show-ref --verify --quiet "refs/tags/$Tag"
    $TagExists = $LASTEXITCODE -eq 0
}
finally {
    $ErrorActionPreference = $PreviousErrorAction
}
$ExistingTag = if ($TagExists) { git -C $ProjectRoot rev-list -n 1 $Tag } else { $null }
if ($TagExists -and $ExistingTag -and ([string]$ExistingTag).Trim() -ne $Head.Trim()) {
    throw "Tag $Tag already points to $ExistingTag, not current HEAD $Head"
}
if (-not $TagExists) {
    Invoke-Checked "Create release tag $Tag" { git -C $ProjectRoot tag -a $Tag -m $Title $Head }
}
Invoke-Checked "Push release tag $Tag" { git -C $ProjectRoot push origin "refs/tags/$Tag" }

$Assets = @(
    $PackageArchive,
    "$PackageArchive.sha256",
    $SourceArchive,
    "$SourceArchive.sha256"
)
$PreviousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
try {
    & $Gh.Source release view $Tag --json tagName | Out-Null
    $ExistingRelease = $LASTEXITCODE -eq 0
}
finally {
    $ErrorActionPreference = $PreviousErrorAction
}

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
