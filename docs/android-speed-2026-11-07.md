# Android response-time improvement

**The installed 0.3.0 package completed the same cold-launch task in a median
59.47 seconds versus 78.43 seconds for the original policy: 24.2% less time.
All four balanced holdout runs passed. Human-operator parity is not established.**

## Contract

The goal is human-operator speed. There is no measured human reference yet, so
neither Airtest time nor an assumed reaction time is a human baseline. Measure
cold-launch completion, map-to-arrival, and sampled-ready-frame-to-input separately.
Game loading is included in task time; it is not called framework computation.

This is a performance change in local OCR configuration and workflow scheduling.
Keep the same intent, full-screen error checks, language variants, focus and
coordinate checks, one Teleport command, loading transition and two consecutive
arrival observations. Only current captured pixels and explicit app rules may
authorize input. Independent arrival screenshots are evaluation evidence only.

The 0.2.0 source baseline is frozen under ignored `artifacts/android-speed/baseline/`.
Raw OCR output, timings, device telemetry and images remain under that ignored
directory tree. New device trials use the earlier cooling gate and sensor sampler.

## Candidates and acceptance

- CPU threads: compare two and four threads on identical images in ABBA order.
  Require unchanged recognized labels and states; do not infer host-wide gains.
- OCR size: test a bounded image size separately from threading. Require every
  known actionable state and target to remain available on development and fresh
  holdout frames. Keep the existing full-resolution setting available for small
  text and other applications. Lower resolution is never an automatic replacement
  for an application's accuracy check.
- Entry handling: give distinct, recognized entry screens independent cooldowns.
  Preserve the cooldown for the same screen and the total attempt limit. OCR
  spelling changes must not reset the same logical screen's cooldown.
- Polling: bound waiting during interactive transitions; retain adaptive waits
  for identified downloads and preparation. Compare separately before combining.

Require live success and independently inspected arrival, with unchanged input
count. Reject any candidate that fails recognition or weakens completion. Report
all development failures. Freeze candidates before the final repeated runs.

## Primary-source lessons

Checked November 7, 2026. [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
provides explicit thread controls; local measurement is needed because extra CPU
threads can add contention. The installed RapidOCR implementation separates
detection, orientation and recognition timing. Its `max` detector mode chooses
internal limits from input size: changing `det_limit_side_len` alone did not change
our actual work or latency. This no-op candidate was rejected. The upstream
[RapidOCR project](https://github.com/RapidAI/RapidOCR) is the implementation source;
its other inference backends are not evidence of a speedup on this host.

## Results

Development trials on the Samsung SM-G781B, Android 13, November 7, 2026:

| Isolated change | Process seconds | OCR seconds | Captures | Outcome |
|---|---:|---:|---:|---|
| Fresh original baseline | 80.11 | 21.22 | 23 | Pass |
| OCR input capped at 1,280 pixels | 79.81 | 13.65 | 26 | Pass; no end-to-end speed claim alone |
| Separate entry cooldowns | 79.61 | 21.12 | 23 | Pass; no end-to-end speed claim alone |
| Interactive wait capped at 0.25 seconds | 65.98 | 33.36 | 53 | Pass; faster, with more observations |

These single trials establish a development signal, not statistical significance.
The faster poller observed the second entry screen 13.03 seconds after the first
tap, but the old shared 15-second cooldown withheld input until a later observation.
That trace and the controlled retry tests justify separate logical-screen cooldowns
in the combined candidate. The same-screen retry limit remains unchanged.

Offline CPU-thread trials used two/four/four/two threads on six immutable images.
Median inference time was 1.335 versus 1.161 seconds; recognized labels were
unchanged in every comparison. A separate seven-image OCR-size comparison retained
all required task states and reduced median inference from 1.055 to 0.508 seconds.
Three saved native-fixture images produced identical label text at both OCR sizes.
Both settings missed the first glyph of the dark Traditional Chinese button;
that fixture's native selector succeeds. This known OCR limitation is retained,
and the size reduction remains opt-in rather than a global accuracy assumption.

Raw development evidence: `artifacts/android-speed/{threads-development,resize-baseline,
resize-development,input1280-development,native-text-baseline,native-text-holdout,
live-baseline,ablations}/`. `resize-development` is the rejected no-op detector
parameter experiment; the effective smaller-input test is `input1280-development`.

## Installed-package holdout

The combined preflight passed in 58.46 seconds. After that run, the runtime,
configuration and input assets were frozen. Four new trials ran in this order:
original, improved, improved, original. Each used the installed 0.3.0 wheel;
the control loaded the frozen original profile with original OCR defaults and
polling. Its semantics match the separate fresh 0.2.0 baseline. Package paths and
version are in every task result. All recorded source/asset hashes were unchanged.

| Trial | Policy | Process seconds | Map to arrival seconds | Inputs | Result |
|---:|---|---:|---:|---:|---|
| 1 | Original | 79.45 | 11.69 | 5 | Pass |
| 2 | Improved | 59.83 | 9.46 | 5 | Pass |
| 3 | Improved | 59.12 | 9.46 | 5 | Pass |
| 4 | Original | 77.41 | 10.97 | 5 | Pass |

| Median measurement | Original | Improved |
|---|---:|---:|
| Full process, seconds | 78.43 | 59.47 |
| Map tap to verified arrival, seconds | 11.33 | 9.46 |
| Sampled frame available to completed tap, seconds | 2.08 | 1.57 |
| OCR total, seconds | 20.52 | 19.97 |
| Capture total, seconds | 6.50 | 15.12 |
| Capture requests | 23 | 66 |
| Screenshot response bytes, decimal MB | 3.16 | 5.18 |
| Battery temperature change, Celsius | +1.40 | +1.25 |
| Processor AP peak, Celsius | 41.60 | 42.50 |
| Skin peak, Celsius | 38.45 | 38.75 |

Each run asserted two entry taps, three task taps and exactly one Teleport command.
The original loading-transition and two-consecutive-arrival checks stayed active.
All four independent arrival screenshots were visually inspected. The phone's
game was force-stopped and its screen turned off after the cohort. All nine device
tasks in this investigation passed, including development and preflight runs.

Ten dispatched inputs per policy give these latency distributions:

| Sampled-frame-to-input seconds | Original | Improved |
|---|---:|---:|
| Mean | 2.10 | 1.57 |
| Median | 2.08 | 1.57 |
| p95, interpolated | 2.54 | 2.13 |
| Minimum / maximum | 1.69 / 2.55 | 1.14 / 2.18 |

This clock starts after a captured image is available and ends when the tap call
returns. It includes recognition, policy checks, evidence saving and input dispatch;
it excludes the time a control was visible before that sampled frame. It is not a
human reaction-time measurement. Input dispatch and its checks alone still take
about 0.59 seconds; full-screen recognition remains another material cost.
Every observation requests a current capture; identical-frame OCR reuse retains
the existing byte-equality policy, not an assumed unchanged screen.

## Decision and limits

Keep the separate entry cooldowns and the 0.25-second interactive polling cap in
the Genshin profile. Keep identified download/preparation backoff and all input
and arrival checks. Expose `LocalOCR(max_side=1280, threads=4)` and equivalent CLI
options for application-validated faster OCR. Keep the generic default size and
two-thread policy unchanged; smaller images are not universally safe for small
text. The intent file and text/button/pop-up variant mechanisms are unchanged.

The speed gain repeated, but costs more capture work: 2.87 times as many capture
requests and 64% more screenshot response bytes. Faster individual OCR calls
offset the increased number of observations, leaving similar total OCR time.
No energy-saving or thermal-win claim follows. Trial 1 changed from battery status
4 to 2; trials 2–4 were charging. All cooling gates passed, but starting battery
temperatures ranged from 32.6 to 34.3 Celsius, skin from 34.4 to 36.6 Celsius, and
thermal status varied between 0 and 1. There were no sampler errors. Two trials
per policy on one phone do not establish statistical significance or general
performance across apps. The earlier Airtest median was 57.90 seconds, from a
different cohort; this result does not establish an advantage over Airtest.

The human-speed target remains open. A useful operator reference must perform the
same cold app launch and same-location teleport, under comparable cooling and
power conditions, and stop at stable gameplay. Record several runs and separate
launch-to-arrival from map-to-arrival. An arbitrary assumed human time would not
validate parity. The current 1.57-second median sampled-frame-to-tap delay is a
concrete remaining optimization target.

The 33 framework tests (including variant and speed contracts) and 19 Genshin tests
passed. Version 0.3.0 is installed in `.venv/androidtestkit`; source changes remain
uncommitted. Wheel SHA256:
`1a6715eb9a6818504f0d2b8763a297b0123a9ae66f7d270c02de850797669414`.
All 15 Python runtime modules in the wheel match the working source.

Raw holdout evidence is ignored under `artifacts/android-speed/holdout/`.
`aggregate.json` contains the distribution calculations; `metadata.json` contains
configuration and source/input hashes. Recompute with Python 3.9 or newer:

```text
python benchmarks/android/speed_summary.py artifacts/android-speed/holdout
```

See [benchmark commands](../benchmarks/android/README.md#speed-follow-up) and the
[public OCR controls](swiftest.md#capture-and-authoring) to use these settings.
