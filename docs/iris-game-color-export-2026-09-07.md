# Optional color at the generated-adapter boundary

Reported symptom: color availability still blocks game calibration. The exact
incident traceback was requested but was not available during this review.
The game-calibration orchestrator already records color-fit exceptions as
`optional_failed_non_gating`. Two remaining export-path defects were reproduced
independently; they are not assumed to explain every possible calibration error.

The generated adapter raised `ValueError` for an explicit `game_matched` request
when its embedded files had no color profile. This happened before the shared
camera facade could provide its established color fallback. Export also read an
optional color file without recovery after registry selection; losing that file
between selection and embedding raised `FileNotFoundError` despite intact rig
and mini-map geometry.

Generated adapters now use the shared camera fallback and record the reason.
Export skips an unreadable optional color file, selects rig-locked imaging,
clears the missing embedded color revision from its receipt, and emits a warning.
Required calibration and mini-map files retain their existing error behavior.
Existing generated Python adapters contain their own constructor code and must
be regenerated to acquire this change; updating the package cannot remove a
raise statement already embedded in an older generated file.

Regression evidence covers a generated adapter delivering both streams with
correct synthetic annotation pixels despite an explicit `game_matched` request,
and a color file lost after real registry resolution followed by successful
export and camera startup. A separate orchestration regression injects
`ProfileResolutionError` from optional color fitting and verifies that both the
returned and persisted game summary remain complete with the geometry result.

The two export cases failed before the fix. Logs:
`artifacts/iris-game-color-export-before.log` and
`artifacts/iris-game-color-nonblocking-tests.log`.
The combined regression suite passed **293 tests** in 41.514 seconds, and
changed-file whitespace checks passed.
Physical camera and packaged-executable validation were not run.
