# Profile retention and evidence sizing

## Outcome and impact

This is a new maintenance feature with intentional data deletion on explicit
apply. Previously, indexed revision history and calibration evidence accumulated
without a retention entry point. `python -m iris_tools profiles purge` now
reports a plan; `--apply` executes it. The source release exposes the same command
through `python-tools.bat profiles purge --apply`, without network or Git access.
No live user storage was purged during development.

The policy belongs to the filesystem profile registry. Defaults are:

| Data | Retention |
| --- | --- |
| `game_model`, `phone_game`, `phone_game_color`, `rig_game_color` | Latest 10 per identity |
| `rig`, `rig_game`, `rig_game_orientation` | Latest 3 per identity |
| Recognized calibration output bundles and optional copied review files | Oldest first above 384 MB |

Identity is the existing registry key: kind, owner, game, panel signature, and
game display signature. Variants have separate histories. There is no standalone
phone revision kind; phone configuration remains unchanged. Camera-specific color
uses the larger count because displacement does not invalidate color.

Active revisions, revisions named by portable activation mirrors, and transitive
dependencies of every retained revision survive beyond the count. Stale portable
mirrors can retain an extra revision until a later activation refreshes them.
This avoids breaking an offline copy that rebuilds its index from those mirrors.

## Nine-game estimate

Assume one phone and one rig. These are local sample sizes, not measurements of
all nine games; image dimensions and content affect compression.

| Local sample | Decimal MB |
| --- | ---: |
| Three minimap output sets, `artifacts/game-minimap-calibration-20260829-*` | 15.09–15.12 each |
| Color output in `artifacts/game-color-workflow-smoke-20260831/result` | 17.32 |
| Of that color output: review images | 10.22 |
| Of that color output: generated standalone adapter | 7.10 |
| Typical indexed rig revision in the local profile root | About 4 |

Nine sets of minimap plus color outputs are approximately
`9 × (15.1 + 17.32) = 292 MB`. A **384 MB** evidence allowance leaves about
92 MB for failures and duplicate review files. The earlier 24.35 MB color-folder
total also included shared rig/profile data, so it overestimated per-game output.
Three typical rig revisions add about 12 MB outside the evidence allowance.
Portable models, masks, required color references, and metadata add further
runtime storage; the samples do not establish a reliable total for nine games
with ten revisions each. 384 MB is an evidence budget, not a total disk cap.

A 256 MB budget is also usable if older evidence may be discarded sooner; it
does not fit one complete sampled output set for every game. The quota is global,
so neither size guarantees evidence for every game. Copied optional review files
share this quota rather than bypassing it inside retained profile directories.

## Deletion and recovery boundaries

- Revision retention acts on indexed immutable revision directories only.
- Standalone evidence cleanup is restricted to recognized bundles beneath
  `<profile-root>/calibrations`. Unknown files count toward that directory's
  size but are not deleted. Custom outputs and recordings are not managed.
- Only runtime entries explicitly named `review_evidence_*` are optional.
  Models, masks, references, calibration files, and other runtime entries survive.
  An `evidence-retention.json` ledger explains removed reviews without rewriting
  the immutable calibration manifest or changing its identity. Registry loading
  and portable export use this effective manifest view.
- Explicit retained runtime references protect source bundles. Audit provenance
  alone does not pin a duplicate bundle. Evidence changed within 24 hours is
  protected; `--evidence-min-age-hours` can change this grace period.
- Paths must resolve strictly inside the managed root. Links and Windows reparse
  points are refused. Whole trees are checked before moving or recursive removal.
- Removed revisions first move into journaled quarantine outside the revision
  hierarchy while SQLite holds the publication lock. A pre-commit failure rolls
  back the index and restores moved revisions. Ordinary registry opening restores
  interrupted moves but does not run retention or delete committed quarantine.
- Committed quarantine is cleaned on explicit apply, with existing Windows access
  retries. Locked files produce a partial result and can be retried. Quarantine
  cannot be rediscovered as a portable revision after a committed deletion.
- Evidence cleanup takes the publication lock and verifies the retained revision
  set has not changed between phases. New publication defers that evidence plan.
  Run cleanup while calibration, imports, exports, and consumers of historical
  revisions are idle: file generation/readers do not all take a registry lock.

JSON reports distinguish preview, complete, and partial, including planned and
actual removals, protection reasons, evidence bytes, over-budget bytes, and
pending quarantine bytes. Apply returns exit code 1 on partial cleanup. Protected
data can exceed either limit; working profiles take precedence over a hard cap.
Ordinary registry initialization/recovery still occurs when opening the CLI;
preview itself removes no data.

## Automatic cleanup after successful calibration

Accepted rig and mini-map publication, and accepted orientation/color calibration,
request cleanup after activation. The shared calibration boundary coalesces those
requests by profile root until the outer workflow exits. Multi-session game runs
finish every component and summary first; the rig CLI also finishes readiness
evidence and standalone adapter export before cleanup. Standalone publication of
a calibrated profile uses the same boundary. Raw registry writes, imports,
composition, adapter resolution and frame processing do not trigger retention.

The automatic path applies the existing default policy: 10 portable revisions,
3 displacement-dependent revisions, 384 MB evidence, and a 24-hour grace period.
Active revisions, portable activation pointers, runtime data and retained
dependencies remain protected. CLI overrides still apply to manual invocations.

Partial runs with accepted, activated components trigger one cleanup; candidate
or entirely failed runs with no accepted publication do not. Cleanup exceptions
and partial-cleanup warnings are logged separately, preserving both successful
calibration results and any original calibration exception. A subsequent
successful calibration or manual apply retries deferred work. No daily timer is
installed. Evidence from failed runs becomes eligible at the next successful
calibration after its grace period.

Impact: storage maintenance now happens automatically and may irreversibly remove
eligible old history. Risks are premature cleanup inside multi-step workflows and
cleanup failures masking calibration success; deferred boundaries, the existing
retention protections/journal and separate error reporting address those risks.
No retention rule, stored schema, calibration acceptance or runtime adapter
behavior changes. Automatic tests use temporary registries only; this change does
not purge the development workspace's profiles as part of verification.

## Verification

Tests use temporary roots only. Coverage includes independent profile identities,
active and transitive dependency protection, color classification, oldest-first
evidence cleanup, protected runtime references, optional-review export, registry
rebuild, stale portable mirrors, concurrent publication between purge phases,
interrupted moves, locked cleanup and retry, path escape and simulated Windows
reparse refusal, and CLI preview/apply behavior. Existing registry, manager,
portable export, Windows write retry, game calibration, configuration, adapter
resolution, CLI, and release-export tests also run.

No phone/camera hardware experiment or packaged executable build is required to
establish these filesystem behaviors; neither was run for this feature. Reparse
refusal and crash/permission cases use controlled fault injection, not an actual
power failure or the user's restricted environment.
