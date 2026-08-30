[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,

    [string]$RepoRoot = $PSScriptRoot,

    [string]$SourceRoot = 'E:\workspace\aria-trace',

    [string]$ExpectedSha256,

    [switch]$Merge
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath([string]$Value) {
    return [System.IO.Path]::GetFullPath($Value).TrimEnd('\', '/')
}

function Get-Sha256([string]$Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $bytes = $algorithm.ComputeHash($stream)
        return ([System.BitConverter]::ToString($bytes)).Replace('-', '')
    }
    finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$repositoryPath = Resolve-FullPath $RepoRoot

if (-not (Test-Path -LiteralPath (Join-Path $repositoryPath '.git'))) {
    throw "Restore target is not a Git clone: $repositoryPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $repositoryPath 'acquisition\workbench.py'))) {
    throw "Restore target is not an AriaTrace clone: $repositoryPath"
}

if ($ExpectedSha256) {
    $actualHash = Get-Sha256 $archivePath
    if ($actualHash -ne $ExpectedSha256) {
        throw "Backup SHA-256 mismatch: expected $ExpectedSha256, got $actualHash"
    }
}

$allowedRoots = @(
    'sessions/workbench/',
    'artifacts/workbench/',
    'profiles/phone_game/',
    'profiles/rig/',
    'profiles/rig_game/'
)

$entries = @(tar.exe -tf $archivePath)
if ($LASTEXITCODE -ne 0) {
    throw "Could not list backup archive: $archivePath"
}
if ($entries.Count -eq 0) {
    throw 'Backup archive is empty'
}

foreach ($entry in $entries) {
    $normalized = ([string]$entry).Replace('\', '/').TrimStart('./')
    if (-not $normalized -or $normalized.StartsWith('/') -or $normalized -match '^[A-Za-z]:') {
        throw "Unsafe absolute backup entry: $entry"
    }
    $segments = $normalized.Split('/')
    if ($segments -contains '..') {
        throw "Unsafe parent traversal in backup entry: $entry"
    }
    $allowed = $false
    foreach ($root in $allowedRoots) {
        $rootWithoutSlash = $root.TrimEnd('/')
        if ($normalized -eq $rootWithoutSlash -or $normalized.StartsWith($root)) {
            $allowed = $true
            break
        }
    }
    if (-not $allowed) {
        throw "Unexpected backup entry outside Workbench data roots: $entry"
    }
}

$targetRoots = @(
    'sessions\workbench',
    'artifacts\workbench',
    'profiles\phone_game',
    'profiles\rig',
    'profiles\rig_game'
)
if (-not $Merge) {
    foreach ($relative in $targetRoots) {
        $target = Join-Path $repositoryPath $relative
        if (Test-Path -LiteralPath $target) {
            $existing = Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($null -ne $existing) {
                throw "Restore target already contains data: $target. Use -Merge only after reviewing it."
            }
        }
    }
}

if (-not $PSCmdlet.ShouldProcess($repositoryPath, "Restore Workbench backup $archivePath")) {
    return
}

tar.exe -xf $archivePath -C $repositoryPath
if ($LASTEXITCODE -ne 0) {
    throw "Backup extraction failed with exit code $LASTEXITCODE"
}

$oldWindows = Resolve-FullPath $SourceRoot
$newWindows = $repositoryPath
$oldJson = $oldWindows.Replace('\', '\\')
$newJson = $newWindows.Replace('\', '\\')
$oldSlash = $oldWindows.Replace('\', '/')
$newSlash = $newWindows.Replace('\', '/')
$textExtensions = @('.json', '.jsonl', '.yaml', '.yml', '.txt', '.md')
$rebasedFiles = 0

foreach ($relative in $targetRoots) {
    $target = Join-Path $repositoryPath $relative
    if (-not (Test-Path -LiteralPath $target)) {
        continue
    }
    foreach ($file in Get-ChildItem -LiteralPath $target -Recurse -File) {
        if ($textExtensions -notcontains $file.Extension.ToLowerInvariant()) {
            continue
        }
        $original = [System.IO.File]::ReadAllText($file.FullName)
        $updated = $original.Replace($oldJson, $newJson)
        $updated = $updated.Replace($oldSlash, $newSlash)
        $updated = $updated.Replace($oldWindows, $newWindows)
        if ($updated -ne $original) {
            [System.IO.File]::WriteAllText(
                $file.FullName,
                $updated,
                [System.Text.UTF8Encoding]::new($false)
            )
            $rebasedFiles += 1
        }
    }
}

[PSCustomObject]@{
    archive = $archivePath
    repository = $repositoryPath
    entries = $entries.Count
    rebased_text_files = $rebasedFiles
    source_root = $oldWindows
} | Format-List
