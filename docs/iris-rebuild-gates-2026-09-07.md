# Functional profile rebuilding: remaining optional gates

This follow-up fixes violations of the functional rebuild contract established
in `8307a8f`: usable streams and geometry determine readiness. Optional metadata
must not prevent a previously working game/phone configuration from rebuilding.

The audit traced headless publication, dependent-profile composition, atomic
activation, and subsequent public-camera startup. New regressions produced six
errors before the fix. Valid source geometry and the independently defined
screen-to-sensor mapping were unchanged; optional orientation strings, game-model
settings, or missing optional/unrelated manifests caused the errors.

The owning boundaries now behave as follows:

- Orientation composition falls back from invalid game-orientation hints and
  surface turns to remaining usable evidence. Valid canonical mini-map axes
  retain priority. Invalid game-model settings cannot veto geometry rebuilding.
  Fallback reasons are recorded in the orientation profile and readiness report.
- Required established geometry sources are loaded first. The registry index
  excludes unrelated camera/phone rig-game records before reading their files.
  An unreadable optional portable candidate is skipped with a warning.
- Optional game-model files can be missing during composition or camera reopen.
  Automatic selection uses the existing default behavior. Explicit manual model
  selections still report an unavailable requested revision.

This changes error recovery, not the stored geometry or its coordinate spaces.
No schema migration is required. Fallback orientation can be an assumption when
measured axes are absent; the recorded reason makes that limitation inspectable.
Missing required geometry files, incompatible dimensions, mini-map overflow,
and failed frame delivery still prevent activation of the candidate. Persistence
errors and conflicting concurrent activation remain errors rather than being
treated as successful fallbacks.

Verification exercises the real headless CLI, registry, composition, readiness,
and public camera facade with substituted optical measurement and acquisition.
The new positive cases reopen full, mini-map, and dual output and check annotation
coordinates against independent synthetic sensor pixels. Negative cases retain
overflow, changed phone dimensions, missing required source, failed acquisition,
and transactional rollback. Physical-device and packaged-executable validation
were not run. Automatic-distortion and unrelated tracking changes are separate.

Validation: **292 tests passed** in 42.855 seconds. Test logs:
`artifacts/iris-rebuild-gates-before.log` (reproduction),
`artifacts/iris-rebuild-gates-tests.log` (combined regression suite).
