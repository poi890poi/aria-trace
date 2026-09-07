# Direct game-color calibration is optional

The previous export fix did not remove the blocker in
`calibrate_game_color_session`. That function compared the captured rig JSON
hash with the current active rig and raised on any difference. It also raised
for unavailable color inputs, relying on the game-calibration orchestrator to
catch them. Direct callers and the standalone color CLI had no such recovery.

The function now fits synchronized data using the capture's own coordinate
conversion. Current rig displacement is irrelevant to that recorded pair. The
source rig is found among all revisions, including inactive ones, solely to
record measurement provenance. A new fit keeps that source dependency and the
captured calibration hash; it is never relabeled as measured on the current rig.
If that revision is not registered, the source calibration hash/path still
records provenance without inventing a revision dependency.

Missing or ineligible color evidence, failed decoding, failed color-profile
resolution, and unhelpful fits return `status: optional_fallback`. The result
states its reason and whether it retains a usable fit for the same camera,
phone, platform, and game or uses uncorrected color. Rig revision changes do not
invalidate retained fits. Fallback does not publish or activate replacement
profiles and does not claim a fresh color measurement. Direct CLI fallback exits
successfully and prints the reason instead of assuming fresh fit metrics exist.
The game-calibration report forwards the same reason and fallback decision.

Spatial correspondence checks still prevent fitting color to different image
regions. A rejected optional fit does not block the game calibration. Real
registry/output persistence errors remain errors; the function does not report
a successful write when persistence fails. Missing current adapter geometry can
leave an explicit unavailable export result without invalidating a measured fit.

The reproduction failed seven cases at the active-rig hash gate. Regression
coverage uses the real public function, session/registry files, publication,
and CLI; image decoding and numeric color fitting are substituted. It verifies
fresh fitting after displacement with the original rig dependency; missing
frames, fit/decoder/resolution failures with and without previous color; exact
retention of active revisions and coefficients; rejection of an unhelpful fit;
and a direct CLI fallback with no fresh metrics. The spatial-mismatch rejection
test remains in place. Physical color fidelity and packaged execution were not
tested.

Logs: `artifacts/iris-direct-color-before.log` and
`artifacts/iris-direct-color-final-tests.log`.
The combined suite passed **298 tests** in 142.031 seconds. Changed-file
whitespace checks passed.
