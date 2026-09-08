# Successful cursor calibration versus stale deployed geometry

## Reproduced cause

The deployment itself is inaccessible. The reported contradiction is reproduced
locally through real publication, registry selection, and the public camera
adapter, with only camera acquisition replaced by a deterministic frame source.

Cursor calibration writes its successful evidence and calls
`publish_minimap_profiles(..., activate=True, compose_rig=False)`. This updates
the portable `phone_game` revision independently of the rig. Automatic adapter
resolution previously selected the existing `rig_game` revision, then followed
its immutable `phone_game` dependency. It checked for a superseded rig but did
not check for an updated portable game source. Reopening the adapter therefore
still loaded the old pivot. The previous answer that reopening alone would load
the new calibration was incomplete.

`compose_rig=False` was added by `63682a5d` on 2026-09-02; the dependency-following
resolver already existed. Independent portable publication is valid, but lacked
the corresponding derived-profile refresh. The cursor estimator fix exposed
this selection gap rather than causing it. An additional same-identity case
showed that rig recomposition prioritized the exact established old revision
over its newly active replacement.

## Correction and boundaries

Automatic resolution now selects current usable portable geometry, refreshes
the derived rig-game snapshot when needed, and uses the existing software
geometry checks before activating the new derived graph. This shared path is
used by the camera adapter, demo, profile resolution CLI, and adapter export.
It also handles a first portable game calibration on an existing rig. Existing
deployed profile roots can recover without refitting already successful cursor
measurements. An unchanged source causes no new publication.

Selection retains source-phone preference and caller-specified display facts.
Inactive candidates do not replace active results. Explicit rig/game revision
overrides remain explicit. A changed rig still follows the existing rig-readiness
workflow. Invalid new geometry reports a failure rather than silently displaying
an old center as current. Optional color is preserved and stays non-gating.
An optimistic active-graph check prevents refresh from overwriting concurrent
activation. Refresh requires writable profile storage and occurs at resolution,
not per frame. An already-open adapter retains its chosen snapshot.

The demo prints its effective profile root and the full rig, phone-game, and
rig-game revision IDs. A refresh is reported explicitly. These distinguish a
stale composition from a different configured profile root or an explicit pin.
Generated standalone adapters intentionally embed a fixed snapshot; regenerate
them after calibration updates. Updating only profile files cannot change an
older executable or a previously exported embedded adapter.

## Demo parity

The demo formerly defaulted to BGR whereas the public adapter defaulted to RGB.
It now takes its color default from the same `ADAPTER_DEFAULTS` object as the
adapter. All other default processing options already shared that policy. The
GUI converts a display copy to BGR for OpenCV, then draws markers and status
panels. Adapter frame arrays remain unchanged. Native/explicit diagnostic camera
paths remain BGR. Explicit GUI operations such as launching a game or pressing O
can change state; they are not default processing.

## Evidence

- Before the fix, successful partial cursor publication changed (75, 30) to
  (73, 32), but a newly opened adapter still selected the older phone revision.
- After the fix, the adapter selects the new revision, reports pivot (73, 32),
  and preserves the independently fitted boundary center (75, 30).
- Default demo construction and default public adapter construction return
  exactly equal frame arrays, resolved processing plans, and cursor geometry for
  the same deterministic input. Reopening unchanged data adds no revision.
- GUI presentation checks RGB-to-BGR conversion at the display boundary and
  proves the input RGB pixels are unchanged.
- Regression coverage includes candidates, explicit old revision selection,
  first-game composition, same-identity source updates during rig republication,
  other-phone protection, existing rig readiness, color fallback, crop transforms,
  portable profiles, retention, and standalone source export.

The selected integration suite passed 168 tests. Tests use temporary profile
roots and a simulated camera. No deployment-device test or packaged executable
build was performed. Diagnostics and tests do not establish which exact binary,
profile root, or optional override the inaccessible deployment is using.
