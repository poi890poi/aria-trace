# Recorder and HUD contention investigation — 2026-09-06

Type: bug fix. Scope: Workbench capture-status synchronization. First-input start,
selected take duration, frame/input formats, cancellation/discard policy, capture
exclusion, and foreground-only HUD visibility are unchanged.

## Timeline and retained successful recordings

Times below are Asia/Taipei (UTC+08:00), using commit author timestamps and recording
manifest timestamps, not file modification times.

| Change | Time | Evidence after the change |
| --- | --- | --- |
| `8338301`: first-input recording, including the per-input callback acquiring `self._lock` | Aug 24 22:32:46 | Many retained successful recordings, through run 19 |
| `1f52915`: retain HUD overlays across transient refresh misses | Aug 30 21:29:02 | Runs 17, 18, 19 |
| `8bba4d4`: extract shared recording runtime | Sep 3 12:46:51 | Runs 17, 18, 19 |
| `00deb72`: add a post-capture route-repeatability label | Sep 4 22:58:18 | Run 19 |
| `a4f9e34`: isolate live-tracker publication from catalog polling | Sep 5 09:27:54 | No later retained successful Workbench recording found |

Recorded sessions under `sessions/workbench/recordings-genshin-impact-pc`:

| Session | Start | Manifest duration | Frames | Inputs | Status |
| --- | --- | ---: | ---: | ---: | --- |
| run_17 | Sep 4 22:44:38 | 181.61 s | 5,414 | 11,286 | complete |
| run_18 | Sep 4 22:50:38 | 181.42 s | 5,410 | 7,693 | complete |
| run_19 | Sep 5 08:30:36 | 90.10 s | 2,703 | 7,824 | complete |

Run 19 also has `automatic:capture_end` and `capture_complete` annotations.
These are successful retained captures, but their manifests do not identify the
running code revision or prove that the HUD was visible. A recording after a
commit timestamp is not proof that its process had loaded that commit.

## Failed take and live reproduction

Read-only `/api/state` inspection of Workbench instance `23ea6f93ac9c` (PID 47064,
started Sep 6 10:21:34) reported the latest attempted run 20:

- selected duration: 240 seconds;
- recording started on `pc_raw_keyboard`;
- manifest duration: 350.993 seconds;
- 5,422 frames and 11,340 persisted input events;
- input health: healthy, with physical keyboard/mouse device handles;
- last error: `Take was canceled; the partial capture was discarded`.

The failed directory had already been removed. `SessionWriter.close()` computes
manifest duration after encoder close, so 350.993 seconds includes finalization;
it does not establish how long acquisition continued beyond the deadline.

Three HUD requests without an induced concurrent state poll succeeded in
0.161, 0.022, and 0.002 seconds. Four paired requests started a full `/api/state`
poll, then requested `/api/hud` 100 ms later with the HUD's real 500 ms timeout.
All four HUD requests timed out (0.502–0.513 s); full state polls took
1.275, 1.316, 1.410, and 1.342 seconds. Raw local response evidence is in
`.tmp/recorder-investigation/live-polling.json` (ignored, not a release artifact).

## Demonstrated mechanism and fix

`descriptor()` holds the Workbench lock across filesystem catalog scans. Both
`hud_descriptor()` and the recording callbacks acquire that lock. The callback
runs synchronously for each input, including the shutdown queue drain. Thus
catalog polling can prevent HUD refresh, delay the recorder's next deadline
check, and block draining/finalizing a take that has already reached its timeout.
The Stop UI posts to the cancellation endpoint, explaining why stopping during
the stall discards otherwise healthy recorded data.

The September 5 tracker change left this existing recorder contention in place;
its diff does not establish a newly introduced recorder defect. The evidence
supports a latent shared-lock defect exposed by catalog workload. It does not
identify the exact artifact or commit that first made this user's workload slow.

Follow-up read-only timing of the individual catalog methods against the current
session/artifact directories further limits attribution. In three repetitions,
`_runs()` (existing recording summaries) took 0.764–0.800 s, `_map_stitches()` took
0.091–0.271 s, and `_live_tracking_runs()` took only 0.025–0.047 s. The initial
`analysis_candidates()` scan took 0.695 s before its session cache warmed;
subsequent calls took 0.016–0.022 s. Thus existing recording summaries were the
largest measured recurring cost, not tracking history. Recording does not execute
the tracker; the shared status response loads all these catalogs regardless of
which UI task is selected. These are current-directory component measurements,
not instrumentation of the already-running process or a historical benchmark.
Raw measurements: `.tmp/recorder-investigation/catalog-timings.json`.

The fix introduces a short-held capture-status lock for recorder callbacks and
HUD reads, with consistent locking for configuration, active-take, and completion
status publication. Catalog consistency retains its existing lock. Catalog
publication may still delay the final browser update, but no longer blocks each
recorded input or the recorder's writer-close path.

## Verification

`test_recording_timeout_and_hud_do_not_wait_for_catalog_polling` uses the real
Workbench recording path and encoder, with a fake desktop and physical-input
packets. It compares the same 200 ms take with and without a held catalog lock.
HUD reads must complete within 400 ms and the recorder must close its writer
within two seconds while the catalog lock is still held. Both variants retain
complete sessions containing frames and inputs after the catalog lock releases.

- Before the fix, the contention variant failed; the no-contention control passed.
- After the fix, both variants passed.
- Negative control: aliasing the new capture-status lock back to the original
  Workbench lock reproduced both failures (`HUD=False`, `writer_saved=False`).
- `python -m unittest tests.test_workbench tests.test_acquisition
  tests.test_workbench_ui tests.test_live_tracker -q`: 112 tests passed.

No live game recording or visible Windows HUD check was performed after the fix.
The existing Workbench process has not been restarted and still runs its old
loaded code. Restart and an in-game timed capture are the remaining device gate.
