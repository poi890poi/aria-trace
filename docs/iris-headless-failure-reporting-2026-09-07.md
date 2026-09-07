# Headless rig calibration failure reporting

Type: bug fix in the shared HIK calibration session's failure handler.

Previously, the progress stream reported only `Failure evidence saved for
review: <directory>`. Console styling classified that line as an error, but
the exception explanation was available only in the propagated exception
(usually stderr) and the saved `failure.json`, not in the progress stream.
An exception from the diagnostic writer could also replace the original
calibration exception.

The handler now reports the failed stage, exception type, and original message
before attempting to save evidence. For example:

```text
Rig calibration failed during geometry: RuntimeError: Only 3 ChArUco corners detected; need at least 6
Failure evidence saved for review: <directory>
```

If there are no images, the original failure is still reported. If evidence
writing fails, a secondary warning gives that error and the original calibration
exception is re-raised unchanged. Empty exception messages still expose the
exception type. The failure JSON retains `error` and `error_type` and adds
`failed_stage`; failures outside a timed stage leave that field null.

This changes failure reporting only. Calibration algorithms, quality gates,
saved profiles, exit/exception behavior for the original failure, image evidence
formats, and successful headless stage order remain unchanged. Failed runs
still do not create a successful calibration bundle. Opening devices remains
outside the existing evidence handler; this fix addresses the reported
failure-evidence path, not device-opening behavior.

Before the patch, four controlled failure regressions produced three assertion
failures and one replaced-exception error. After the patch, 109 tests passed:

```powershell
./.tools/standalone-release-py31210/Scripts/python.exe -B -m unittest tests.test_hik_calibration_failure_reporting tests.test_hik_rig_calibration tests.test_rig_presentation
```

The regression executes the real session failure handler and evidence writer
with a synthetic target image, and checks explanation ordering, stage metadata,
no-image behavior, an evidence permission error, original exception identity,
cleanup, and absence of a successful bundle. The existing suite covers the
successful headless stage sequence and console presentation. Logs are in
`artifacts/hik-failure-reporting-before.log` and
`artifacts/hik-failure-reporting-tests.log`.

Physical-camera and packaged execution were not run. These checks demonstrate
the reporting defect and its correction; the cause of the user's particular
calibration failure requires its evidence directory's `failure.json`.
