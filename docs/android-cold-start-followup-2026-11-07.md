# Startup-inclusive latency experiments — 2026-11-07

Eight complete cold-start trials found no convincing E2E improvement from either
binary screenshot transport or less frequent startup observation. Both experiments
are retained outside the runtime. The default workflow and installed wheel are
unchanged.

The earlier 3–4-second target applied only to teleporting from already-ready
gameplay. It excluded application launch, connection and initial loading. It was
not an estimate of complete startup. The earlier measured cold-process median was
52.57 seconds; these new trials took 51.15–54.31 seconds. No four-second cold-start
claim is supported. The [post-mortem](android-latency-postmortem-2026-11-07.md) now
states that scope explicitly at its start.

## Contract and protocol

The primary clock starts before spawning the fresh worker process and ends after
that process exits. It includes framework imports, helper setup, app launch, all
game connection/loading waits, two game-entry actions, the three teleport actions,
two verified arrival observations, evidence completion and worker cleanup.

The app is force-stopped before every trial. This is an installed game with an
existing account and cached assets; it is not an OS reboot, cleared app data, first
installation or credential-entry test. The 60-second minimum cooling period and
the independent final screenshot are outside the timed task. Both capture-helper
builds were cached before the binary cohort; pushing and starting the helper and
creating its forwarding rule remain inside the clock.

Each experiment used baseline/candidate/candidate/baseline order, two tasks per
variant. Battery ≤34.5°C and skin ≤37°C gates passed before every trial. All runs
used phone RFCR91GWXLX, full-resolution JPEG80 capture, exact-repeat suppression,
OpenVINO CPU OCR with four threads and 1280 maximum input side, the same intent,
fresh input guards, and the existing completion conditions. There were no tuning
changes within a cohort. Source and asset hashes were checked throughout and
again after completion.

## Results

Each row has only two samples. Mean equals median here. The sample p95 uses linear
interpolation and must not be interpreted as a population tail estimate.

| Experiment / variant | Individual cold process times | Mean / median | Sample p95 | Maximum |
| --- | --- | ---: | ---: | ---: |
| Transport baseline | 54.312, 51.666 s | 52.989 s | 54.180 s | 54.312 s |
| Binary transport | 54.028, 51.154 s | 52.591 s | 53.885 s | 54.028 s |
| Polling baseline | 51.719, 51.674 s | 51.696 s | 51.717 s | 51.719 s |
| Unknown/startup polling at 500 ms | 51.147, 52.314 s | 51.731 s | 52.255 s | 52.314 s |

### Binary transport: retain as an experiment

The candidate uses an ADB-forwarded local socket with a bounded, length-prefixed
binary frame. It removes Base64 while retaining the production capture, comparison
and JPEG mechanisms. It does not crop, reduce image quality, skip OCR or alter
action timing. The Android helper is generated from the frozen production Java
source under ignored artifacts. The ordinary helper was not edited.

Pooled fresh-frame capture medians were **92.60 ms baseline versus 87.54 ms binary**.
The complete task difference was **0.398 s, or 0.75%**. Both variants varied by
roughly 2.6–2.9 seconds between their two runs. This small cohort does not establish
a repeatable E2E win, so binary transport was not made the default.

A preceding read-only Settings check confirmed the expected approximately 25%
wire-byte reduction for the initial image. Exact-repeat reads took 56.5 versus
58.6 ms median: there is little payload to optimize on a repeat. The game cohorts
had median transferred bytes of 11.71 versus 9.56 MB, an 18.3% reduction, but their
frame contents and recognition paths differed. That observed percentage is not
an equal-frame codec comparison.

Four transport contract tests passed: fragmented frames, exact repeats, truncated
payloads, invalid/oversized headers and invalid first-frame repeats. Invalid data
forces a fresh PNG acquisition rather than reuse of an old image. Read-only device
checks and all binary game trials used the expected direct-bitmap path without
fallback. Rotation/reconnect holdouts have not been run for this experimental
transport; it is not a packaged or validated replacement for general use.

### Startup polling: substantial work reduction, no speed improvement

The candidate changes only the sleep after `unknown` observations when no task
action is pending: a minimum of 500 ms instead of the normal 100-ms cap. Recognized
entry screens still trigger actions immediately after observation. Pending map,
waypoint and teleport transitions retain their previous polling. Download and
preparation backoff, OCR coverage, thresholds and action guards remain unchanged.

| Median per complete cold task | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Captures | 191.5 | 78.5 | 59.0% fewer |
| OCR calls | 86 | 51 | 40.7% fewer |
| Cumulative capture time | 14.765 s | 6.770 s | 54.1% lower |
| Cumulative OCR time | 12.790 s | 8.776 s | 31.4% lower |
| Transferred bytes | 11.72 MB | 7.14 MB | 39.1% fewer |
| Complete process time | 51.696 s | 51.731 s | Essentially unchanged |

The first candidate was 0.57 s faster than its preceding baseline. The second was
0.64 s slower than its following baseline. The direction did not reproduce. The
method is useful evidence that much observation work overlaps game waits, but is
not adopted as a speed optimization. It also deliberately increases the possible
delay before newly visible text is inspected during an unknown state; a reduction
in work alone does not justify that trade for a latency-focused default.

## Why the savings did not add up to faster startup

In the transport cohort, observed phases took approximately:

* 16.9–17.3 s to complete the first game-entry action;
* another 12.8–13.5 s to complete the second entry action;
* another 15.5–18.7 s to recognize gameplay and begin the map action;
* another 4.3–4.9 s from map-input start through verified arrival and task evidence.

These include observation delays; they are not internal game-engine timings or a
proof of irreducible loading time. The largest variation occurred before verified
gameplay, not in screenshot transport. A two-second update-check sleep was followed
by several more seconds of unready observations, so deleting that sleep would not
automatically advance the next action.

The polling experiment removed roughly 12 seconds of accumulated capture/OCR work
without shortening the task. That work was largely taking place while the game
was not ready for the next action. A component's saved processing time cannot be
subtracted directly from wall-clock E2E time.

The remaining promising local direction is faster recognition immediately after
a screen becomes actionable, including the regional OCR prototype. It still needs
explicit full-screen error coverage and independent validation before deployment.
The earlier 40.98-second warm-arrival outlier remains unresolved and did not occur
in this cohort. These results do not prove that no further improvement is possible.

## Validation, limitations and delivered state

All eight tasks passed with exactly five sent inputs, one teleport action, loading
observed, and two final world observations. Independent final screenshots were
reviewed and showed normal gameplay at the expected waypoint. Every sensor sample
had battery, AP and skin readings without acquisition errors. No capture fallback
was used. Runtime and asset hashes remained unchanged.

USB charging status changed during the binary cohort. Android thermal status
reached 1 in the polling cohort, including a candidate and a baseline task. Cooling
gates passed, but they do not make all thermal states identical. Do not pool these
cohorts to estimate small speed or energy effects. Two samples per variant do not
support reliability, statistical-significance or general tail-latency claims. Since
neither candidate established the target E2E benefit, neither was promoted to a
fresh release holdout or production default.

Added tools:

* `benchmarks/android/binary_capture.py`: experimental transport and helper builder;
* `benchmarks/android/binary_cold_comparison.py`: transport-only cold comparison;
* `benchmarks/android/startup_poll_comparison.py`: startup-observation comparison;
* `tests/test_binary_capture_probe.py`: transport failure/reuse contracts.

The source runtime and installed wheel were not changed. Settings was restored,
temporary forwarding rules were removed, and no screenshot/input helpers remained
running. No commit or push was made.

Run with this workspace's `.venv/androidtestkit/Scripts/python.exe`:

```text
benchmarks/android/binary_cold_comparison.py --serial RFCR91GWXLX --output artifacts/android-binary-capture/NEW
benchmarks/android/startup_poll_comparison.py --serial RFCR91GWXLX --output artifacts/android-startup-poll/NEW
-m unittest discover -s tests -p test_binary_capture_probe.py -v
```

Raw results, telemetry, hashes and screenshots remain ignored in
`artifacts/android-binary-capture/cold-development/` and
`artifacts/android-startup-poll/cold-development/`. Aggregate calculations are in
`artifacts/android-cold-followup-summary.json`. The binary read-only preflight is
`artifacts/android-binary-capture/component.json`. These are local research
artifacts, not repository assets.
