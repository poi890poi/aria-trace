# Genshin benchmark setup and usage (supplement)

Use this guide for the current source checkout. Read the
[benchmark requirements](android-genshin-benchmark-requirements.md) for the task and timing contract and the
[API guide](swiftest.md) for custom selectors and tests. Commands below use
PowerShell from the repository root. They do not imply a new published package
release; the installed wheel may differ from this checkout.

## 1. Prepare the host and phone

Install a compatible Python interpreter and Android SDK Platform Tools. Enable
USB debugging on the phone, connect USB and authorize this computer. For the fast
JPEG helper, also install a JDK and Android SDK Platform/Build Tools; set
`ANDROID_HOME` or `ANDROID_SDK_ROOT` to that SDK. These must resolve:

```powershell
adb version
java -version
javac -version
adb devices
```

The measured setup used Python 3.12.10. For a new checkout, create an environment
and install the source. Skip environment creation if reusing the existing one.
If `py -3.12` is unavailable, use the full path to your Python 3.12 interpreter.

```powershell
py -3.12 -m venv .venv/androidtestkit
$atkPython = Join-Path (Get-Location) '.venv/androidtestkit/Scripts/python.exe'
& $atkPython -m pip install -e '.[ocr-fast]'
& $atkPython -c "import swiftest; print(swiftest.__file__)"
& $atkPython -m swiftest devices
```

For native-only tests, install `-e .` instead. For OCR without OpenVINO, use
`-e '.[ocr]'` and `--ocr-backend cpu`. `auto` can fall back to CPU; `openvino`
fails explicitly if unavailable. Verify the reported backend before comparing
performance. Package requirements are declared in [pyproject.toml](../pyproject.toml).

The printed module path identifies what code is actually running. Running from
the repository root can import the checkout even when an older wheel is installed.

In each new terminal, set the interpreter and the intended serial explicitly:

```powershell
$atkPython = Join-Path (Get-Location) '.venv/androidtestkit/Scripts/python.exe'
$phoneSerial = 'RFCR91GWXLX'
```

That serial is the study phone. Use your own authorized serial for general CLI
tests. The binary/polling/diagnostic research harnesses explicitly restrict
operation to the study phone; their settings are not portable without validation.

## 2. Check access and run an ordinary test

Read-only screenshot check:

```powershell
& $atkPython -m swiftest snapshot artifacts/android-usage/screen.png --serial $phoneSerial
```

Run the Samsung Settings example; this opens Settings and sends its test inputs:

```powershell
& $atkPython -m swiftest run examples/settings/test_connections.py --serial $phoneSerial --output artifacts/android-usage/settings --failure-screenshot
```

The Settings selectors cover the documented Samsung layouts/languages, not every
Android vendor. A normal test produces `report.json` in a run-ID subdirectory of its output directory and
returns a nonzero exit status on failure. In PowerShell, inspect `$LASTEXITCODE`.

## 3. Run the Genshin example

The game must already be installed with a usable account, working connection and
required assets. This command launches it when needed, handles recognized entry
screens, opens the map and sends a teleport after verifying the selected panel:

```powershell
$gameOutput = 'artifacts/android-usage/genshin-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $atkPython -m swiftest run examples/genshin/teleport.json --profile examples/genshin/profile.py --serial $phoneSerial --timeout 240 --capture auto --ocr-backend openvino --ocr-threads 4 --ocr-max-side 1280 --output $gameOutput --failure-screenshot
```

The profile is trusted Python code. The public intent proposes a visible unlocked
waypoint; it may abstain when its outline is hidden by the player icon. A test
author can supply a position or template from a reviewed current screen. Do not
copy another device's coordinates without checking its layout.

For the exact measured same-location task, substitute the local frozen intent
`artifacts/airtest-benchmark/assets/teleport.json`. That private file is not in a
fresh checkout. Ordinary CLI execution can start from an already-running game;
**this command by itself is not a cold-start benchmark**. It also normally leaves
the tested app open when it finishes.

Inspect `$gameOutput/<run-id>/report.json` and the reported evidence directory under
`$gameOutput/<run-id>/genshin/`. Successful profile output includes `teleport_completed`,
action counts and before/map/selected/arrival images. The benchmark wrappers add
separate loading and independent arrival screenshots. A successful input command
alone does not establish task completion.

## 4. Run complete cold-start measurements

The existing cold harnesses require the private assets directory, including
`teleport.json`. Airtest comparison additionally needs `manifest.json` and all
seven referenced template images. Do not substitute different assets within a
cohort. `prepare_assets.py` expects these reviewed 2400×1080 source captures under
`artifacts/airtest-benchmark/`:

```text
entry-setup2.png
preflight-progress.png
awake.png
map-setup.png
selected-setup.png
```

With those exact inputs, `& $atkPython benchmarks/android/prepare_assets.py` builds the
local assets. Its crop rectangles and waypoint belong to the study layout. It
cannot generate missing screenshots or automatically configure another phone.

To measure only the current TestKit baseline, with startup and loading included:

```powershell
& $atkPython -c "from swiftest.capture import build_capture; print(build_capture())"
$coldOutput = 'artifacts/android-startup-poll/baseline-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $atkPython benchmarks/android/startup_poll_comparison.py --serial $phoneSerial --order baseline,baseline,baseline,baseline --output $coldOutput
Get-Content (Join-Path $coldOutput 'results.json') -Raw | ConvertFrom-Json | Select-Object variant,success,process_seconds,rest_gate_met
```

This sends real game inputs. Each trial force-stops the game, rests the phone for
at least 60 seconds, seeks the thermal gate for up to 300 seconds, then starts a
fresh worker. Setup, launch, connection/loading, the complete teleport task and
worker cleanup are included in `process_seconds`. Cooling is excluded. The
harness restores the previous app on its normal completion/error path.

The first command only builds/caches the host-side capture JAR. It does not start
the game or warm OCR. This makes the documented cached-compilation boundary
explicit; helper push and launch still occur inside every measured worker.

Use a new output directory each time. A failed trial or unmet gate stops this
harness and preserves the results. `complete.json` appears only after the cohort
finishes. Allow several minutes; four cold starts are not a four-second operation.

The harness fixes OCR to OpenVINO/four threads/1280 and uses
`.venv/androidtestkit/Scripts/python.exe` for its workers. It does not inherit all
ordinary CLI settings or offer an `--ocr-backend` switch. Check source/configuration
before attempting a different setup.

| Output | Meaning |
| --- | --- |
| `metadata.json` | Order, source/asset hashes, prior app and timing definition |
| `results.json` | All recorded trials, including failures; primary `process_seconds` |
| `NN-testkit/task/result.json` | Worker timings, inputs, observations, acquisition and OCR metrics |
| `NN-testkit/console.log` | Worker events and errors |
| `NN-testkit/rest.json`, `temperatures.json` | Rest gate and device telemetry |
| `NN-testkit/independent-arrival.png` | Independent outcome screenshot outside the timed task |
| `complete.json` | Completed cohort and unchanged-source assertion |

## 5. Compare methods or investigate a delay

For framework test authoring, start with the [framework guide](android-usage.md).
These are research commands, not enabled runtime features. Use separate new
output directories and compare against their own controls.

```powershell
& $atkPython -c "from swiftest.capture import build_capture; from benchmarks.android.binary_capture import build; print(build_capture()); print(build())"
& $atkPython benchmarks/android/binary_cold_comparison.py --serial $phoneSerial --output artifacts/android-binary-capture/new-comparison
& $atkPython benchmarks/android/startup_poll_comparison.py --serial $phoneSerial --output artifacts/android-startup-poll/new-comparison
```

The binary harness compares ordinary and binary transport. The polling harness compares ordinary
and 500-ms unknown-state polling. Each defaults to four balanced cold starts.
Neither candidate has established a convincing cold-start speedup. See the
[results](android-cold-start-followup-2026-11-07.md).

The preceding Python command caches both helper builds without controlling the
phone. Do not count one variant's first compilation inside a comparison while
the other variant uses a cached build.

To capture intermediate images and OCR-stage timings, then analyze them offline:

```powershell
& $atkPython benchmarks/android/latency_postmortem_probe.py --serial $phoneSerial --output artifacts/android-latency-postmortem/new-diagnostic
& $atkPython benchmarks/android/latency_postmortem_analysis.py artifacts/android-latency-postmortem/new-diagnostic
& $atkPython benchmarks/android/latency_roi_replay.py artifacts/android-latency-postmortem/new-diagnostic
```

Only the first command controls the phone; it runs one cold task and three warm
tasks by default. Diagnostic writes affect timing. The offline commands create
`analysis.json`, contact sheets and `roi-replay.json`. Regional OCR agreement is
not independent correctness evidence or proof of faster E2E completion.

For Airtest, install `airtest==1.4.3` in the separate
`.venv/airtest-benchmark` environment. With the private assets present:

```powershell
& $atkPython benchmarks/android/full_comparison.py --serial $phoneSerial --output artifacts/android-airtest-current/new-comparison
& $atkPython benchmarks/android/full_comparison_summary.py artifacts/android-airtest-current/new-comparison
& $atkPython benchmarks/android/audit_full_comparison.py artifacts/android-airtest-current/new-comparison
```

This default comparison runs eight cold and eight warm tasks. The summary's
`cold process` row is the startup-inclusive metric. The harness accepts explicit
`--testkit-python`, `--airtest-python` and `--assets` paths. See the
[benchmark guide](../benchmarks/android/README.md) for other protocols and the
[current comparison report](android-airtest-current-comparison-2026-11-07.md).

## Troubleshooting and interpretation

| Symptom | Check / action |
| --- | --- |
| Device missing or `unauthorized` | Unlock the phone, accept its debugging prompt and check `adb devices`; verify the selected serial. |
| Missing SDK/JDK or PNG fallback | Check SDK paths, `java`, `javac`, platform and D8 availability. Inspect `capture_acquisition` and `fallback`; fallback timing is not a direct-bitmap result. |
| OpenVINO fails | Use the documented interpreter/extra. For a functional CPU run, explicitly select `cpu`; do not silently mix that result with the fast-backend benchmark. |
| Waypoint not found | Check marker visibility, intent and layout. Use a reviewed target; do not relax ambiguity or teleport-panel checks to force progress. |
| Long `unknown` interval | Retain the failure and diagnostic frames. Determine whether the game was unready or recognition missed it. Do not label all unknown time as unavoidable loading. |
| Low component time but unchanged E2E | Work may overlap game waits. Compare the same startup-inclusive endpoint and inspect stage/action timestamps. |
| Stopped benchmark | Read the trial/console/gate evidence. Do not overwrite failed runs. If the host was forcibly terminated, inspect phone focus and temporary helpers/ADB forwarding before restarting. |

Raw outputs remain private and ignored. Save concise conclusions and aggregate
tables in repository documentation. Temperature is not an energy measurement, and
the provisional 47–49-second favorable-run estimate is not a measured minimum.
