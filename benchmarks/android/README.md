# Genshin Airtest comparison

For benchmark setup and commands, use the [benchmark supplement](../../docs/android-genshin-benchmark-usage.md)
and [benchmark requirements](../../docs/android-genshin-benchmark-requirements.md).
For ordinary framework use, start with the [framework guide](../../docs/android-usage.md).
The benchmark documents distinguish complete cold-start E2E time from warm tasks
and component timings. The original protocol
below is historical; later studies and their timing boundaries are linked in the
follow-up sections.

For warm script overhead without startup/app loading, see the
[steady-state latency experiments](../../docs/android-steady-latency-2026-11-07.md).
The command, counted-input, capture and preprocessing probes compare one
mechanism at a time and keep raw evidence under ignored `artifacts/`.

This is benchmark tooling, separate from the framework runtime. It compares the
existing TestKit OCR profile with an independently written Airtest template script.
The target is the same unlocked waypoint at the player's current position. Each
run opens the map, verifies the selected waypoint panel, sends one Teleport command,
observes a loading screen, and confirms two consecutive gameplay observations.
It measures a same-location teleport, not navigation to a different destination.

## Protocol fixed before measured trials

- Samsung SM-G781B, Android 13, serial RFCR91GWXLX; Windows host, Python 3.12.
- USB ADB, game network over existing Wi-Fi. Keep power connection, graphics,
  brightness and display resolution unchanged. Record charging status and brightness.
- Force-stop the game without clearing data before each trial. Both implementations
  must handle the start prompt. Installed helper binaries and OS disk caches are warm.
- One development preflight per implementation before collecting holdout trials.
  Complete any fixes before freezing inputs; record failures rather than dropping them.
- Eight trials in order Airtest, TestKit, TestKit, Airtest, TestKit, Airtest,
  Airtest, TestKit. Four runs per implementation, opposite order within pairs.
- At least 60 seconds screen-off rest before each trial. Seek battery <=34.5 C
  and skin <=37.0 C, for at most 300 seconds. Record if this gate is not reached.
- Independent host sampler reads `dumpsys battery` and `dumpsys thermalservice`
  every two seconds. Save baseline, end, peak and temperature change for battery,
  AP and SKIN sensors. Report cached HAL readings as device telemetry, not external
  surface measurements. No energy-in-joules claim follows from temperature.
- Measure wall-clock process time (imports, device setup, task, teardown), task time,
  and map-to-arrival separately. Helper installation and rest are outside timing.
- Freeze code, assets and settings before holdout; save hashes and package versions.
  Save independent ADB screenshots after each trial for outcome review. Inspect
  loading and arrival evidence; do not count unverified actions as success.
- Report all failures, per-trial values, medians and ranges. Four trials per tool on
  one phone do not establish statistical significance or general Android performance.
- TestKit uses automatic JPEG capture, local OCR and existing adaptive waits.
  Airtest 1.4.3 uses JAVACAP, MAXTOUCH and bounded template searches. Its default
  MINICAP setup failed during preflight; the working fallback is explicit in trials.
  Javacap's natural portrait pixels are rotated clockwise to match this phone's
  landscape-left layout. Other rotations/resolutions require new configuration.
- The pipelines differ in recognition, polling, capture and input. Any timing gap
  belongs to these scripts and settings, not an isolated framework effect.

## Run

Create an isolated Python environment and install `airtest==1.4.3`. Use a second
environment with this repository's package and `[ocr]` extra for TestKit. Host
automation uses Python; no PowerShell or shell script is required.

`prepare_assets.py` crops inspected setup screenshots from the ignored
`artifacts/airtest-benchmark/` directory. It writes provenance in `assets/manifest.json`.
Templates are private generated evidence and are not committed. To use another
device, capture its entry screen, gameplay HUD, map, and selected waypoint and
update the crop/search rectangles and waypoint in the preparer.

```text
python benchmarks/android/prepare_assets.py
python examples/genshin/airtest_teleport.py --serial SERIAL --assets artifacts/airtest-benchmark/assets --output artifacts/airtest-benchmark/preflight-airtest
python benchmarks/android/testkit_teleport.py --serial SERIAL --intent artifacts/airtest-benchmark/assets/teleport.json --output artifacts/airtest-benchmark/preflight-testkit
python benchmarks/android/compare_genshin.py --serial SERIAL --airtest-python PATH_TO_AIRTEST_PYTHON --testkit-python PATH_TO_TESTKIT_PYTHON --output artifacts/airtest-benchmark/holdout
python benchmarks/android/summarize.py artifacts/airtest-benchmark/holdout --output artifacts/airtest-benchmark/summary.md
```

Use a new output directory for each measured cohort. Failed runs stop the cohort;
do not overwrite or silently omit them. The harness force-stops the game and turns
the screen off when it finishes, so the phone does not continue heating while idle.
The summarizer produces the measured tables. Add a reviewed interpretation and
record charging-state changes before treating that output as a final report. The
completed November 7 comparison is in `docs/android-airtest-benchmark-2026-11-07.md`.

## Speed follow-up

The later 0.3.0 investigation is documented in
[`docs/android-speed-2026-11-07.md`](../../docs/android-speed-2026-11-07.md).
The historical table above belongs to the frozen earlier source, not the current
profile. The follow-up retains that profile locally under ignored
`artifacts/android-speed/baseline/` and verifies its hashes. It measures OCR size,
entry cooldowns and polling independently, then compares the combined settings
with the original profile in balanced order. The local baseline and screenshot
assets must exist before reproducing this experiment.

```text
python benchmarks/android/speed_compare.py --serial SERIAL --output artifacts/android-speed/new-holdout --order baseline,combined,combined,baseline --installed
```

The `--installed` option imports the installed wheel and records its location and
version. Install the local wheel into that Python environment first. Each run
retains the original cooling and telemetry protocol, and additionally asserts
five inputs and exactly one Teleport command. `speed_trial.py` records capture,
recognition and input timings. Sampled-frame-to-input time starts when a captured
frame is available, not when a button first became visible between samples.

To use the validated faster settings through the public CLI with the existing
intent and current profile:

```text
python -m swiftest run artifacts/airtest-benchmark/assets/teleport.json --profile examples/genshin/profile.py --serial SERIAL --timeout 240 --ocr-max-side 1280 --ocr-threads 4
```

This command can resume from gameplay. The benchmark additionally force-stops
the app before timing to measure a cold app launch. Smaller OCR inputs are an
application-specific choice; the default OCR size remains unchanged.

Primary references checked November 7, 2026:
[Airtest API](https://airtest.readthedocs.io/en/latest/all_module/airtest.core.api.html),
[Android battery telemetry](https://developer.android.com/reference/android/os/BatteryManager),
and the installed Airtest 1.4.3 capture and input implementation. The existing
comparison research in `docs/swiftest.md` supplies the broader context.

## OCR backend follow-up

OCR experiments are CPU-only. GPU choices have been removed from the runners;
previous DirectML results are retained only as historical evidence. OpenVINO in
these experiments targets CPU.

Version 0.4.0 adds optional OpenVINO execution with the same PP-OCRv4 models.
Install `.[ocr-fast]` in the benchmark environment. Use explicit backends when
comparing results, because `auto` depends on installed optional packages:

```text
python benchmarks/android/ocr_backends.py --engine legacy --images IMAGE_PATHS --output artifacts/android-ocr/new-cpu
python benchmarks/android/ocr_backends.py --engine legacy-openvino --images IMAGE_PATHS --output artifacts/android-ocr/new-openvino
python benchmarks/android/speed_compare.py --serial SERIAL --installed --order combined-cpu,combined-openvino,combined-openvino,combined-cpu --output artifacts/android-ocr/new-phone-holdout
python benchmarks/android/speed_summary.py artifacts/android-ocr/new-phone-holdout
```

The image benchmark defaults to four threads, 1,280-pixel inputs and three passes.
Its warm median excludes the first pass. Save construction and first-call costs
separately. The phone comparison includes setup costs and uses the same intent,
resolution, polling and input guards for both backends. It does not replay cached
task answers. Results and rejected candidates are in
[`docs/android-ocr-2026-11-07.md`](../../docs/android-ocr-2026-11-07.md).

The later [OCR solution evaluation](../../docs/android-ocr-solutions-2026-11-07.md)
compares MaaFramework, scoped queries and alternate-reader recovery. Its
`ocr_solution_probe.py` records target correctness, ambiguity and recovery costs
against fixed manifests; `capture_ocr_fixture.py` obtains native-annotated phone
examples. These are experiments, not part of the installed 0.4.0 runtime.

## Warm end-to-end follow-up

The [warm E2E report](../../docs/android-e2e-latency-2026-11-07.md) measures verified
gameplay through one teleport and two verified arrival observations. Startup and
model warmup are excluded. Task time includes evidence-write completion. Controls
load the frozen starting profile in `artifacts/android-e2e/baseline/genshin/`;
this local baseline and the original intent assets must exist before reproduction.
The `production` case executes the actual source Phone/Runner with default capture
and its chosen wait. It does not install or benchmark the existing 0.4.0 wheel.

```text
python benchmarks/android/warm_e2e.py --serial SERIAL --order baseline,production,production,baseline --output artifacts/android-e2e/new-holdout
python benchmarks/android/guarded_fixture.py --serial SERIAL --production --output artifacts/android-e2e/new-guard-contract
python benchmarks/android/capture_rotation.py --serial SERIAL --capture auto --output artifacts/android-e2e/new-capture-contract
```

Optional `stream` and `stream_exact` cases require `--scrcpy-server PATH` for
scrcpy 4.1. The latter compares complete decoded frames for exact equality before
reusing recognition. It is an isolated experiment, not approximate visual caching.
Each cohort records case configuration, source hashes, per-frame timings, battery
telemetry, results and independent arrival screenshots under ignored artifacts.

## Current-source comparison with Airtest

The [full comparison](../../docs/android-airtest-current-comparison-2026-11-07.md)
compares the current source runtime with pinned Airtest 1.4.3. It measures four
cold-process and four warm-workflow tasks per implementation in balanced order,
with the established cooling protocol and sensor sampler. Each warm task follows
the same tool's cold task, starts a fresh child process, and excludes verified
capture/recognition warmup. It does not send a warm-up tap.

```text
python benchmarks/android/full_comparison.py --serial SERIAL --output artifacts/android-airtest-current/new-holdout
python benchmarks/android/full_comparison_summary.py artifacts/android-airtest-current/new-holdout
python benchmarks/android/audit_full_comparison.py artifacts/android-airtest-current/new-holdout
python benchmarks/android/warm_airtest_matched.py --serial SERIAL --output artifacts/android-airtest-current/new-matched-poll
```

The environment paths default to this workspace's existing isolated environments;
override `--airtest-python` and `--testkit-python` elsewhere. The existing private
template assets and intent must be present. The follow-up initializes both input
transports without a tap, then compares twelve warm tasks: original Airtest,
Airtest with only its wait interval changed to 100 ms, and unchanged TestKit.
Neither harness changes the application profiles or installs a new TestKit wheel.

## Latency post-mortem

The [post-mortem](../../docs/android-latency-postmortem-2026-11-07.md) separates
game-transition bounds from controller costs and audits pixel/OCR waste.
`latency_postmortem_probe.py` records intermediate images, OCR-stage timings and
per-observation transfer bytes using the unchanged workflow. It is scoped to the
authorized phone and restores the prior app. Its instrumentation makes the runs
diagnostic evidence, not clean performance comparisons.

```text
python benchmarks/android/latency_postmortem_probe.py --serial RFCR91GWXLX --output artifacts/android-latency-postmortem/new-run
python benchmarks/android/latency_postmortem_analysis.py artifacts/android-latency-postmortem/new-run
python benchmarks/android/latency_roi_replay.py artifacts/android-latency-postmortem/new-run
```

The latter two commands are offline. Regional OCR replay compares HUD decisions
and component cost only; it does not implement full-screen error coverage or
establish a live task speedup. All frame traces and replay outputs remain ignored.

## Startup-inclusive follow-up

The [cold-start follow-up](../../docs/android-cold-start-followup-2026-11-07.md)
tests binary capture transport and 500-ms polling of unrecognized startup states
separately. Each cohort uses baseline/candidate/candidate/baseline cold starts,
including framework setup, app startup and loading, verified arrival and cleanup.
Both candidates remain experimental; neither established an E2E speedup.

```text
python benchmarks/android/binary_cold_comparison.py --serial RFCR91GWXLX --output artifacts/android-binary-capture/new-run
python benchmarks/android/startup_poll_comparison.py --serial RFCR91GWXLX --output artifacts/android-startup-poll/new-run
```

These harnesses are scoped to the authorized phone and restore its prior app.
Their generated helpers, telemetry and screenshots stay under ignored artifacts.
