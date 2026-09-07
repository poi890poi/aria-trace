# IRIS demo annotation and recalibration reliability audit

Inspected 2026-09-07 at source HEAD `4f98e91`, then implemented the fixes below
in the working tree. The numbered findings preserve the original pre-fix
evidence; their line references refer to that version. Unrelated workspace
changes were preserved. No physical camera or phone was operated.

## Implementation status

- Rig publication now stages and validates dependents before activating the
  complete configuration in one SQLite transaction. A failed composition,
  coverage check, camera read, or database update preserves prior selections.
  Concurrent profile changes reject an outdated publication attempt.
- Recomposing an established game retains its exact phone-game source revision;
  a newer profile from another phone cannot replace it. A new full-only open
  loads available game geometry and rejects stale dependencies consistently
  with mini-map and dual modes.
- Both fresh and reuse headless paths validate the composed adapter in all
  three modes. Fresh readiness is recorded in `game_readiness.json`; reuse
  records it in `reused_calibration.json`. Software-only publication identifies
  its limited evidence separately from camera frame validation.
- Color policy was subsequently corrected per the user's explicit requirement:
  color never gates rig publication. Displacement retains the previous fit for
  the same camera/phone/game; missing or unusable fits fall back to rig-locked
  output, including explicit game-matched requests. See
  [the color P0 correction](iris-color-nonblocking-2026-09-07.md). The historical
  color findings below describe the earlier, superseded strict contract.
- The demo shows separate boundary, cursor, and axis reasons, including
  provider errors, disabled toggles, missing calibration, and coordinate-space
  mismatches. A bad optional orientation vector no longer hides a valid circle.
  Status and FPS panels sit outside the camera image, avoiding occlusion.
- Canonical phone crops with tagged geometry are no longer independently
  rotated by conflicting surface hints. Legacy untagged geometry is bound to
  its recorded phone raster during composition.
- A complete portable active-profile snapshot prevents partial per-profile
  pointer updates from reconstructing a mixed configuration. A healthy live
  database remains authoritative over stale or damaged mirrors.

Regression coverage is in `tests/test_iris_recalibration_contract.py`, including
the actual headless entry point, registry, exported adapter path, public camera
facade, frame delivery and geometry mapping. Only physical acquisition and
optical measurement are substituted. Known sensor pixels independently check
geometry placement before and after a synthetic rig shift. Database failure,
concurrent modification, copied-profile recovery, and GUI explanations are
covered explicitly.

Validation: **269 tests passed** across recalibration-contract, stream GUI,
game-camera, registry, profile manager, rig reuse, automatic profiles, adapter
export, portable profiles, game-color workflow, configuration values,
architecture boundaries, rig calibration, and system configuration suites.
The run log is `artifacts/iris-recalibration-tests.log`. `git diff --check`
passed for the changed implementation and tests. The synthetic demo preview
at `artifacts/iris-annotation-preview.png` was visually inspected: annotation
pixels remain visible between the external telemetry header and status footer.
No physical headless run or packaged-executable validation was performed.

One pre-existing reuse test initially wrote a synthetic `CAM-1` revision to the
workspace profile root because it lacked an explicit temporary root. That exact
revision and its pointers were removed after verifying its temporary provenance;
the test now uses an isolated root. Existing real selections were preserved.

## Required behavior

Once a phone/game profile has supplied working boundary, cursor, and game-axis
annotations, reopening with the same profile revisions and output settings must
preserve those capabilities. The annotations are saved calibration geometry, not
per-frame game-content detections.

A successful fresh headless rig calibration must also preserve that working
phone/game profile and compose it against the new rig. Success must mean that the
previously available capabilities remain usable, not merely that a rig file was
saved. If changed coverage or display geometry makes that impossible, report the
specific incompatibility and preserve the previous active configuration. An
unchanged-rig reuse result must check the dependent game configuration as well.

## Findings

### 1. Recalibration can replace the working phone's geometry with another phone's

**High impact; reproduced through the real registry, composition, camera geometry,
and GUI renderer with simulated camera frames.**

`profile_management.py:587` iterates all active phone-game profiles, newest first
(`profile_registry.py:1197`). Compatibility checks platform and panel raster, not
the target phone. The deduplication key at `profile_management.py:607` includes
platform, game ID, and game-display signature, but excludes source phone identity.
The first matching variant wins. `_rig_phone_game_context` then assigns it to the
new rig's phone. The orientation composition loop has the same selection pattern.

Reproduction:

1. Publish a working PHONE-1 game profile with boundary, axes, and cursor geometry.
2. Verify that the adapter's full stream in dual mode draws annotations.
3. Publish a newer PHONE-2 profile for the same game and raster, with a usable crop
   but no boundary/cursor geometry.
4. Publish a fresh PHONE-1 rig calibration.
5. Publication returns normally, but the new PHONE-1 rig-game profile depends on
   PHONE-2. The stream still opens; its annotation disappears.

The original PHONE-1 profile remains in the registry but is no longer the selected
source of the composed geometry. This directly violates the requested invariant.

Correction boundary: preserve the established working phone-game revision;
prefer exact phone and layout identity when choosing a source. Portability must
not silently override that source merely because a different phone is newer.

### 2. Activation precedes dependent publication; reuse can bypass the broken state

**High impact; partial-publication failure reproduced, reuse path inspected.**

`publish_rig_calibration` activates the new rig at `profile_management.py:284`,
then reconciles its dependents at line 293. An exception during reconciliation
leaves the new rig active. In the reproduction, the next dual adapter resolution
rejects the old rig-game revision as stale. The failed publication itself raises
an error; it does not falsely return success.

However, the reuse path at `apps/hik_rig_calibration.py:366` checks the physical rig,
writes a reuse receipt, exports a full-only adapter, and returns at line 412. It
does not reconcile or check game-dependent readiness. It can therefore report an
unchanged reusable rig without repairing or detecting the broken game dependency.

Correction boundary: stage the rig and its dependent profiles, validate the whole
selected configuration, and activate it together. Reuse must also verify the
capabilities expected from previously working phone/game profiles.

### 3. The GUI discards both error details and legitimate unavailable reasons

**Confirmed; three silent paths reproduced.**

At `apps/hik_stream.py:248`, `_runtime_geometry` converts `RuntimeError`,
`TypeError`, and `ValueError` into `{}`. A missing getter or non-mapping response
also becomes `{}`. `overlay_stream_geometry` ignores the adapter's `reason` and
silently rejects missing or mismatched image-size metadata.

A provider exception, an explicit unrectified-space reason, and missing frame
metadata each produced an unchanged image with no stdout/stderr explanation.
The GUI's telemetry shows FPS/read time/age, not annotation readiness. Toggle
feedback and ROI fallback messages are console-only.

Correction boundary: retain a status and reason separately for boundary, cursor,
and axes, visible in the GUI. Distinguish disabled, absent from profile,
unsupported output space, incompatible metadata, and provider error. Report a
change once in the log rather than silently swallowing it or logging every frame.

### 4. Invalid optional axes can suppress an otherwise valid boundary

**Confirmed with a real profiled reader and simulated frames.**

`game_camera.py:640` validates the whole spatial circle, including its optional
orientation frame. A zero-length axis raises `ValueError`; the GUI swallows it.
The camera still opened and delivered frames with masking disabled, but no
boundary was drawn even when the axes toggle was off. Meanwhile orientation
composition (`profile_management.py:470`) deliberately accepts invalid axis
evidence via a documented fallback. The runtime and publication paths therefore
disagree about what that fallback permits.

Correction boundary: validate annotation capabilities before publication and
keep failure of optional axes separate from usable circle geometry. A fallback
must retain its reason through to the GUI.

### 5. Full mode deliberately omits the game geometry profile

**Confirmed existing policy and a user-visible capability gap, not a random fault.**

The demo defaults to `full`. `AdapterRequest.requires_minimap_profile` at
`profile_registry.py:299` is true only for minimap/dual. In full mode,
`resolve_adapter` does not load `rig_game`, even when a game ID and working game
profile exist. `compat.py:535` then constructs a rig-only reader, whose facade
geometry getters return empty mappings.

The same game can thus show annotations in the full window of dual mode but show
none in full-only mode. Existing test
`test_adapter_mode_controls_dependencies_not_profile_identity` explicitly asserts
this profile-selection behavior. The adapter exported after rig calibration is
also configured as full-only.

Correction boundary: distinguish requesting a full output raster from requesting
a rig-only capability set. A full game stream should be able to use the selected
game's saved geometry without requiring a second output window.

### 6. Successful publication does not establish downstream annotation readiness

**Confirmed by inspection and the successful cross-phone reproduction above.**

Recomposition copies optional geometry fields without a required-capability
check (`profile_management.py:313`). Nonmatching panel variants are skipped with
`continue`; no per-game skipped/incompatible result is returned. The CLI prints
counts of recomposed profiles and returns success, without validating a complete
game output or comparing capabilities with the previously working configuration.
Calibration of a visible portion of the phone also does not prove that the
selected game's complete mini-map is visible; that is checked later at adapter
open (`game_camera.py:741`). The demo can then fall back to full output, with only
a console message (`hik_stream.py:896`, `:926`).

Related success semantics: `--headless` without `--save` can finish without saving;
`--no-profile` intentionally avoids publication. Those are valid explicitly
selected operations, but their completion must not be presented as game readiness.

Secondary trust issue: `_space_matches_frame` checks only width/height. Two
different coordinate spaces with the same dimensions pass that guard. This is
not a demonstrated cause of the user's missing annotation, but it means visible
annotation alone does not establish spatial correctness.

## Conditions where annotations are legitimately unavailable

| Condition | Expected missing component | Required distinction |
|---|---|---|
| Native MVS or a rig-only reader | All game geometry | No game calibration loaded |
| Full-only reader without a game profile | All game geometry | Full mode now loads game geometry when a compatible profile exists |
| `--no-rectify` | Runtime boundary, cursor, and axes overlays | Current renderer does not support projective ROI geometry; canonical geometry may still exist |
| No calibrated boundary in the selected profile | Boundary and attached game axes | Absent evidence; unacceptable silent regression from a previously complete profile |
| Static-only cursor calibration without a verified rotation center | Cursor center/envelope | Static size can exist without a calibrated rotating center |
| Center exists but rotating-envelope diameter does not | Cursor ellipse only | The center cross can still be drawn |
| Legacy boundary without an orientation frame | Game-axis arrows only | The boundary should remain available |
| `G`, `B`, `C`, or `A` disabled a component | The disabled component | User setting; defaults reset to enabled for a new GUI run |
| Geometry queried before a frame in that stream has been acquired | Runtime-space overlay until metadata exists | Normal API startup state, not normal during the demo's successful read/draw loop |
| Required mini-map lies outside the camera's view after moving the rig | Mini-map output/annotation cannot be promised | Explicit coverage incompatibility; do not claim the previous game capability is ready |

Provider exceptions, stale dependencies, metadata mismatches after a successful
frame read, or selecting another phone's profile are failures requiring an
explanation, not normal reasons to silently remove annotation. Ordinary changes
in scene content, brightness, motion blur, or cursor visibility are not per-frame
gates for these saved-geometry overlays. Such changes can affect image-based
orientation initialization when explicitly requested, but do not erase saved
geometry.

## Verification and limits

- 98 existing tests passed across `test_hik_stream_cli`, `test_hik_game_camera`,
  `test_profile_registry`, `test_profile_manager`, and `test_hik_rig_reuse`, using
  the project's pinned Python test environment and unittest.
- The isolated reproduction is
  [iris-annotation-audit-20260907.py](../artifacts/iris-annotation-audit-20260907.py).
  Its recorded findings are
  [iris-annotation-audit-20260907.json](../artifacts/iris-annotation-audit-20260907.json).
- Four opens of the unchanged complete profile all provided annotations in the
  simulated reader. The demonstrated loss followed profile recomposition.
- Exceptions were injected only for the activation-failure experiment; the
  cross-phone selection and missing annotations exercised actual implementation.
- No physical headless calibration or packaged-executable run was performed.
  The specific user's intermittent incident is not yet correlated with its launch
  arguments, resolved revisions, and output metadata. These findings establish
  reproducible flaws, not an unsupported claim about which one caused that run.

The appropriate acceptance check is: working phone/game profile -> fresh rig
calibration -> compose the same source revision -> reopen the adapter in each
supported output mode -> verify retained annotation capabilities and coordinate
contracts. A failure must preserve the last working active configuration and
name the affected capability. Pixel positions may change with the new rig;
availability and correct space conversion must not regress silently.

## Follow-up: review of the actual headless command boundary

The additional check runs the real `apps.hik_rig_calibration.main`, registry,
publication, standalone export, public `HikCamera` facade, ROI validation, and
geometry getters. The optical calibration session is replaced by a synthetic
saved calibration; the physical reuse measurement is simulated separately.
Camera frames come from the existing test adapter. This checks the software
contract after measurement, not the physical accuracy of the measurement itself.

| Scenario | Headless result | Subsequent adapter result |
|---|---|---|
| Compatible new rig, one working phone/game profile | Exit 0; publishes and exports | Full and mini-map streams in dual mode open; boundary, cursor, axes, and matching image-size metadata remain available; exact phone-game revision retained |
| New rig whose sensor view cuts off the required game mini-map | Exit 0; reports one rig-game and one orientation profile recomposed; exports adapter | Public facade rejects open with `MinimapRoiUnavailableError` |
| Interrupted dependent publication, followed by physically reusable rig result | Exit 0; reports unchanged saved calibration and exports adapter | Same stale `rig_game` dependency error before and after reuse |

These extend findings 2 and 6 from code inspection to executable command-path
reproductions. The coverage case deliberately supplies an insufficient saved
rig; it does not establish that a particular real optical calibration accepted
that same fixture. The missing downstream check is directly demonstrated.

Evidence:

- [Headless contract reproduction](../artifacts/iris-headless-contract-review-20260907.py)
- [Recorded command logs and adapter results](../artifacts/iris-headless-contract-review-20260907.json)

### Color-dependent applications have an additional compatibility gap

At `profile_registry.py:1370`, a stale rig-game color profile enables fallback
to rig-locked color. That fallback applies even to an explicit
`color_policy="game_matched"` request (lines 1400 and 1417). The headless CLI
prints that synchronized game calibration is needed to refresh the fit
(`apps/hik_rig_calibration.py:521`). The existing
`test_new_rig_recomposes_active_phone_game_without_game_images` explicitly asserts
the downgrade to `rig_locked`.

This is documented existing behavior, not evidence that the calibration math
randomly failed. It does mean the stronger promise of "the previous game-phone
configuration still works without another game calibration" is not currently
met for consumers that require game-matched colors. Whether a particular game
algorithm fails under that downgrade is unmeasured. A stale optical color fit
must not simply be reused as if it were valid: restoration or validation of the
required color capability belongs in the new-rig completion path, or that
capability must be reported as unavailable. A console warning alone does not
satisfy an explicit game-matched output request.

### Existing test coverage does not establish the complete contract

- `tests/test_hik_rig_reuse.py:325` mocks `publish_rig_calibration` in its
  headless saved-output test, and mocks the standalone exporter. It verifies
  command wiring and configured paths, not the resulting adapter.
- The reuse test mocks the physical result and the exporter, then checks that
  the optical calibration session was skipped. It does not open a game adapter.
- `tests/test_profile_manager.py:308` verifies recomposed profile fields,
  activation, and resolver output, but does not read a frame through the public
  facade after recalibration.
- The new compatible-path check verifies software capabilities and metadata
  dimensions. It does not prove optical pixel alignment, color fidelity, or
  packaged/physical operation.

The completion contract therefore needs checks for the retained source profile,
required coverage, adapter startup, frame-space geometry, and any explicitly
required color calibration before committing active selections. Both fresh and
reuse paths need this check. Retain the previous configuration for recovery;
after physical rig movement, do not imply its old transform is still valid for
the moved hardware merely because publication of the new configuration failed.

The reproductions above describe the pre-fix review. Runtime fixes and their
verification are recorded in the implementation status at the start of this report.
