# P0: game color must not block headless calibration

The color-readiness gate introduced in `622417f` was too strict. It interpreted
any new rig revision as invalidating game color, so displacement alone could
raise `ProfileResolutionError` before activating otherwise usable geometry.
It also rejected explicit `game_matched` requests when no matching fit existed.

The user's corrected contract is authoritative: color is optional, ordinary
uncorrected game color is always an acceptable fallback, and displacement does
not invalidate an existing color fit.

The fix removes color from the geometry-readiness gate. Readiness probes do not
apply optional game color. Color selection accepts a usable existing fit for
the same camera, phone, and game independently of its measured rig revision.
The original immutable revision and its measurement dependency are retained;
the system does not fabricate a fresh color measurement or rewrite provenance.

Missing, unreadable, or malformed optional color profiles fall back to the
rig's locked imaging without a game gamma/CCM correction, including explicit
`game_matched` requests and missing manual color selections. A color file lost
between resolution and camera open follows the same fallback. If SDK color
setup fails, the public facade closes the affected handle and opens a fresh
reader without game color, avoiding partially applied conversion state.

Rebuilding now follows a functional rule: publish whenever the resulting
streams and required geometry work. Optional orientation evidence can fall
back to calibration-display orientation; malformed optional axes retain their
usable boundary and produce readiness notices. Legacy orientation records with
missing source data no longer stop unrelated usable profiles from rebuilding.

Geometry dependencies remain pinned and validated. A real
camera acquisition failure can still stop calibration; color availability
cannot. Changes in lighting, white balance or exposure can affect an old fit's
accuracy, but that is an optional quality concern, not an activation gate.
Displacement alone never demands recalibrating color.

Regression coverage includes fresh headless success with an old color fit,
retained exact color revision and applied coefficients, missing/malformed/lost
color data, missing game identity, missing manual color revision, and an SDK
color failure followed by successful frame delivery without correction.
Tests also retain hard failures for mini-map overflow and incompatible phone
dimensions while accepting absent orientation and invalid optional axes.
Physical color fidelity and packaged-executable operation were not tested.

Validation: **287 tests passed**, including the actual headless entry point,
public adapter, all three output modes, retained color coefficients, SDK
fallback, registry/rebuild behavior, export, and the Windows retry regressions.
Log: `artifacts/iris-color-p0-tests.log`. Changed-file diff checks also passed.
