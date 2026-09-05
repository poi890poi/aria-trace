# Whole-scene shift benchmark

This research package compares fast scene-yaw mechanisms at the recorded
30 FPS cadence. Production does not import it. The high-cost 960 px,
800-corner, forward/backward, essential-gated KLT method is a cross-method
reference, not external truth. Timestamp-aligned raw mouse input independently
checks direction and response lag; it is not angular ground truth.

The candidate ladder changes one cost at a time:

1. spatial downsampling: 960, 640, 480, 320, then 240 px width;
2. feature subsampling at 480 px: 800, 400, 200, then 100 corners;
3. forward/backward and essential validation independently disabled;
4. gray and gradient phase correlation at 480, 320, and 240 px;
5. frame stride two, whose held frames count against fresh coverage.

Run a bounded development screen first, then rerun supported candidates on
complete development and holdout sessions:

```powershell
$python = '.tools\standalone-release-py31210\Scripts\python.exe'
& $python -m benchmarks.scene_shift.run_benchmark `
  --session run17=sessions\workbench\recordings-genshin-impact-pc\run_17 `
  --output artifacts\poc\scene-shift-screen `
  --maximum-frames 600
```

`report.json` contains parameters, source and implementation hashes, aggregate
metrics, and per-session results. Candidate JSONL files preserve chronological
per-frame telemetry. `REPORT.txt` is the narrow-screen review summary.

Fresh coverage means a current-frame measurement was produced; a reused or
subsampled state is not fresh. The 30 FPS core criterion is at least 95% of all
recorded frame transitions producing fresh evidence within 33.3 ms. Accuracy
metrics are reported only as disagreement from the accurate KLT reference. A
candidate is retained when its worst-session disagreement is at most 0.10
degree P95 and 0.50 degree worst per processed interval; these are engineering
screening limits, not claims of physical ground-truth accuracy.
