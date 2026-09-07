# Final ChArUco target transition: cause and compatibility review

Type: bug fix in final rig verification, plus correction of misleading status
wording. The user reported an initial `[OK]` atlas message followed by no final
ChArUco detections, with a white/gray patch in the failed raw camera frame.
Their evidence is on another machine and was not available for inspection.

## Introduction history

| Date/time (+08:00) | Commit | What the source establishes |
|---|---|---|
| 2026-08-29 01:43:25 | `874368b` | Added `verify_final_imaging()` in `acquisition/rig_calibration/hik/workflow.py`. It requested ChArUco, waited for the phone paint acknowledgement, then immediately used a fixed camera-frame budget. The `Final manual imaging...` error and `complete squares` layout wording were present from introduction. |
| 2026-08-29 02:04:08 | `18ca839` | Added camera-backlog draining for Data Matrix capture. Final ChArUco verification was left unchanged. |
| 2026-08-31 00:45:09 | `e6a8c28` | Moved the code into the calibration service. Blame at the current location points here, but inspecting its predecessor identifies the earlier feature commit as the origin. |
| 2026-09-01 02:51:18 | `90daf68` | Added stage-owned final-image retention and its regression. That test deliberately mocks detections on uniform white/gray frames to test retention; it does not test target acquisition. |
| 2026-09-01 21:50:38 | `1b1afc1` | Added console status styling that classifies the substring ` complete` as success. This turned a generated atlas layout description into an apparent `[OK]` detection result. |

The current owner is `rig_runtime/workflows/hik_rig_calibration.py`, method
`HikRigCalibrationSession.verify_final_imaging`. The final acquisition loop
survived subsequent moves. The September 2 headless benchmark reduction runs
later and left this verification unchanged. The September 4 defaults refactor
retained the 12-frame default. The September 7 automatic distortion change
`951df90` changes earlier distortion/geometry work, not this final target
transition. The reporting fix `54a5822` exposes the exception but does not alter
which frames final verification receives. These facts establish the origin of
the defect, not which timing/device change made it appear on the user's rig.

## Mechanism and missed boundary

Exposure verification presents a white patch; white-balance measurement keeps
that target visible. Final verification then requests the atlas. A phone paint
acknowledgement says the phone processed the new presentation; it does not say
the camera's queued exposures now contain that presentation. Counting queued
white/gray frames against the entire verification budget can therefore report
zero detections before an otherwise readable atlas arrives.

The final check was introduced to verify ChArUco under the resulting locked
camera settings. Its acquisition path omitted the display-to-camera transition
boundary. The later Data Matrix backlog correction was not applied consistently
to this path. Existing final-frame retention tests exercised a different
contract and mocked away this defect. Passing those tests was insufficient
evidence for target-transition correctness.

`manual` meant the camera's auto controllers were disabled after software
selection of exposure/gain/WB. It never meant a human calibration step. The new
wording says `automatically selected, locked`. The atlas message explicitly
describes a generated layout and uses `whole squares`, avoiding the false
success classification without changing styling of other messages.

## Narrow correction and working behavior

Final verification now reads until the actual ChArUco detector sees the atlas,
bounded by the existing operation timeout. The first observable atlas frame
begins the unchanged configured verification sample count. Leading frames that
do not contain readable ChArUco are counted separately as acquisition discards;
failures after acquisition still count against verification quality. No fixed
sleep or unconditional frame discard is added. A blocking camera read remains
subject to the adapter's own read timeout; an in-flight read can finish after
the acquisition deadline.

The detector, geometry, reprojection calculation, best-frame selection, camera
settings, successful profile policy, and headless stage order are unchanged.
A target that never becomes observable still fails. Hardware read exceptions
propagate immediately. This does not loosen detection or geometry acceptance.
An atlas that is unreadable under the locked settings is not misrepresented as
a confirmed queue problem.

Acquisition status, acknowledged revision, discarded-frame count, elapsed time,
and last detector error are saved with final verification and failure JSON.
The timeout message distinguishes the phone acknowledgement from camera
observation and lists a previous patch or unreadable atlas as possibilities.

## Verification

The delayed-target regression uses a real generated atlas and real ChArUco
detector, valid phone acknowledgement telemetry, 16 preceding white/gray
frames, and the normal 12-frame verification budget. The old path fails before
the atlas arrives; the corrected path discards those 16 frames and verifies
12 atlas frames. A fake clock bounds the timeout test without device sleeps.

A direct comparison against `54a5822` also exercised identical image inputs:

| Sequence | Baseline | Corrected |
|---|---|---|
| Two immediately visible atlas frames | 2 reads, 36/36 corners, detection rate 1.0 | Exactly identical existing metrics, read count, and saved atlas |
| Atlas followed by a blank frame | 2 reads, 36/0 corners, detection rate 0.5 | Exactly identical existing metrics, read count, and saved atlas |
| 12 blank frames followed by two atlas frames | Failed after 2 reads | 12 acquisition discards, then two successful verification frames |

The comparison retained identical p50/p95 reprojection errors for the working
sequences. Additional tests reject a permanently blank acknowledged target,
preserve its raw failure image and acquisition metadata, propagate camera
disconnection, and retain the stage-owned final frame.

Run the focused suite:

All 114 focused tests pass on the bundled Python 3.12 environment.

```powershell
./.tools/standalone-release-py31210/Scripts/python.exe -B -m unittest tests.test_hik_final_target_transition tests.test_hik_calibration_failure_reporting tests.test_hik_rig_calibration tests.test_rig_presentation
```

Local evidence: `artifacts/hik-final-transition-before.log`,
`artifacts/hik-final-transition-tests.log`, and
`artifacts/hik-final-transition-comparison.jsonl`. The comparison script is
`artifacts/compare_final_target_transition.py` (run as a module from the root).
Physical-phone and packaged-executable validation remain unrun. The user's
reported image is consistent with the reproduced mechanism, but cannot prove
the exact device-side cause without that run's timing and image sequence.
