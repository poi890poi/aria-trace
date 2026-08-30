# Workbench backup and restore

Workbench working data spans two required roots:

```text
sessions/workbench/
artifacts/workbench/
```

A portable hardware-aware backup additionally includes:

```text
profiles/phone_game/
profiles/rig/
profiles/rig_game/
```

Stop recording and analysis before creating or restoring a backup. The Workbench
server may remain open if `/api/state` reports no active recording or analysis.

## Restore into a fresh clone

Clone the repository, then run the restore script from any PowerShell directory:

```powershell
$archive = 'E:\path\to\workbench-portable-data.zip'
$clone = 'E:\workspace\aria-trace-fresh'

& "$clone\restore-workbench-backup.ps1" `
  -Archive $archive `
  -RepoRoot $clone `
  -SourceRoot 'E:\workspace\aria-trace' `
  -ExpectedSha256 '<SHA-256 printed when the backup was created>'
```

`SourceRoot` is the repository location where the backup was created, not the new
clone location. The script extracts only the five allowed data roots and rebases
the original absolute repository path in JSON, JSONL, YAML, and text evidence.

The script rejects absolute archive entries, parent traversal, unexpected roots,
checksum mismatch, a non-AriaTrace target, and non-empty target data by default.
Use `-WhatIf` to validate an archive without extracting it.

Use `-Merge` only after reviewing an existing clone's data. It permits extraction
into non-empty target roots and may replace files with the same relative path.

## Archive scope

The sessions/artifacts-only archive is sufficient to review recordings and most
existing analysis. It does not reproduce the machine-specific camera/phone adapter
selection. Use the portable archive when the restored clone must reuse the same
rig, phone, and rig-game profiles.

External programs and drivers are deliberately excluded: Python environments,
ADB, scrcpy, FFmpeg, HIK MVS runtime/driver, and device permissions must be
installed separately on the restored machine.
