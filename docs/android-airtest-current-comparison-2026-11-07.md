# Current TestKit versus Airtest: full comparison

**There is no overall TestKit speed win. Its cold-process median was 52.57 s
versus 57.24 s for the existing Airtest script. With all backends initialized,
TestKit took 5.18 s, original Airtest polling took 6.01 s, and Airtest at 100-ms
polling took 4.81 s. All 28 measured tasks passed, but a separate warm-app TestKit
run took 40.98 s. The median improvement does not remove that tail problem.**

These are current workspace source and Airtest 1.4.3 measurements on one phone,
not a published TestKit release or a general cross-platform ranking. Each table
states its timing boundary. The runtime policies were preserved; only benchmark
instrumentation and the declared Airtest polling variant were added.

## Protocol fixed before trials

This is benchmark tooling and analysis, not runtime optimization. Compare current
workspace TestKit with installed Airtest 1.4.3 on Samsung SM-G781B Android 13,
serial RFCR91GWXLX, using the existing Chinese Genshin scene and private assets.
Both must select the same waypoint, issue one Teleport, observe loading, and
verify consecutive gameplay arrivals. Cold trials require two entry and three
task taps; warm trials require exactly three taps. No data clearing, account,
graphics, brightness or network changes. Independent screenshots are evaluation
evidence, never runtime targets. Raw evidence stays ignored under
`artifacts/android-airtest-current/`.

Run one development cold+warm pair per tool, then freeze source/configuration.
Measured order: Airtest, TestKit, TestKit, Airtest, TestKit, Airtest, Airtest,
TestKit. Each pair starts with a fresh Python process and force-stopped app.
Record cold process wall time including imports/setup/task/teardown, task time,
and map-to-arrival. Then start a second process on the still-open game, warm its
capture/recognizer on verified gameplay, and time the warm workflow. Warm setup
is excluded from warm task time and reported separately. Both tasks include
evidence writes. Airtest's normal API input delays remain enabled.
No warm-up tap is sent, so lazy touch-transport initialization can remain inside
the first timed warm input. This differs from a long-lived session whose input
channel is already active; the prior 4.56-second TestKit result used that other
setup. Both sides here follow the same fresh-process warm-setup rule.

Before each cold trial, rest screen-off at least 60 seconds, seeking battery
<=34.5 C and skin <=37 C for at most 300 seconds. Retain unmet gates and all
failures. Sample battery/AP/skin every two seconds during child processes; no
energy claim follows from temperature. Fixed power and room conditions cannot
be assumed from telemetry alone. Stop a cohort on failure, diagnose, and retain
that failed cohort before any new holdout.

TestKit uses source `Session(auto)`, the current Phone/Runner, CPU OpenVINO OCR,
four threads and a 1,280-pixel input limit. Airtest uses the existing independently
written template workflow, JAVACAP/MAXTOUCH and recorded 2400x1080 assets. The
working capture fallback and rotation normalization were established earlier.
Instrumentation records capture-to-completed-input, capture and recognition costs,
frame counts, state/action events, versions and source/asset hashes. Check hashes
before/after collection and inspect independent arrivals. Do not compare the new
warm figures directly with the old cold-start headline.

The outcome contract is shared; recognition and intervention policies differ.
Report these explicitly with supported platform/API/authoring differences. A
timing winner for these scripts is not a general framework or reliability winner.
Four trials per tool/condition give descriptive medians/ranges/p95, not population
tail guarantees or statistical significance.

### Polling-policy sensitivity check

The stock Airtest script waits 500 ms for most states and 250 ms for loading;
the optimized TestKit profile caps interactive waits at 100 ms. To avoid
attributing that policy difference to the framework, after the main frozen cohort
run an additional fully initialized comparison: original Airtest (A), Airtest with
only its wait interval changed to 100 ms (B), and unchanged TestKit (T), order
A T B / B T A / T A B / B A T. Keep Airtest's touch delay,
templates, guards and synchronous evidence unchanged. Use one excluded cold
warmup and fresh child processes. Initialize each input backend without sending
input before timing: Airtest's touch-proxy setup and TestKit's read-only foreground
query. This addition was fixed before follow-up collection, after measured input
timings exposed a 3.40-second first MAXTOUCH dispatch in the main warm-app cohort.
It separates fully initialized steady operation from warm-app resume. No new input
or weaker postcondition is introduced. This isolates a practical
configuration alternative; it is not a search for optimal Airtest parameters.

## Main cohort: cold launch and warm-app resume

All 16 planned tasks passed, with four runs per tool per condition. All eight
cold trials met the cooling gate. Source and asset hashes were unchanged.
The independent screenshots were visually inspected and showed the expected
gameplay location. Values below are seconds unless stated otherwise.

| Measurement | Current TestKit | Airtest 1.4.3 |
|---|---:|---:|
| Cold full process median | **52.566** | 57.240 |
| Cold full process mean | 55.081 | 59.131 |
| Cold full process range | 51.93–63.26 | 56.43–65.61 |
| Cold full process empirical p95 | 61.684 | 64.384 |
| Cold task median | 51.188 | 56.838 |
| Cold map-input start through completed arrival median | 4.675 | 5.561 |
| Warm-app task median | **5.398** | 8.855 |
| Warm-app task mean | 14.187 | **8.954** |
| Warm-app task range | 4.97–40.98 | 8.40–9.70 |
| Warm-app task empirical p95 | 35.675 | **9.586** |
| Warm-app full child-process median, including setup/teardown | 9.752 | 11.350 |
| Warm-app capture-to-completed task input median, ms | 536 | 851 |
| Warm-app dispatch median, ms | 227 | 571 |
| Cold capture requests median | 195 | 77 |
| Cold capture wall time median | 14.689 | 14.852 |
| Cold OCR/template recognition wall time median | 12.739 | 0.304 |
| Warm-app capture wall time median | 1.561 | 2.086 |
| Warm-app OCR/template recognition wall time median | 1.415 | 0.016 |

TestKit's cold-process median is 8.2% lower and its warm-app median is 39.0%
lower, but its warm-app mean and worst case are worse because of the retained
40.98-second arrival-confirmation delay. Thus a median-only claim would hide a
material consistency problem. Four-sample p95 values are descriptive, not reliable
population tail estimates. The warm-app condition includes first-touch backend
setup; the separate fully initialized comparison addresses that cost.

Capture and recognition are not additive estimates of the entire game-loading
cost: the game continues while each tool processes observations. TestKit requests
more frames, but direct capture and exact-repeat reuse keep its capture total
similar. Its full-screen OCR still consumes far more host wall time than the
regional template matching. Phone thermal sensors do not measure that host CPU
cost, and stage wall times are not CPU utilization or energy measurements.

| Pair | Tool | Cold process s | Warm-app task s | Both passed |
|---:|---|---:|---:|---|
| 1 | Airtest | 57.41 | 8.40 | Yes |
| 2 | TestKit | 51.93 | 5.20 | Yes |
| 3 | TestKit | 63.26 | 40.98 | Yes |
| 4 | Airtest | 65.61 | 9.70 | Yes |
| 5 | TestKit | 52.76 | 5.60 | Yes |
| 6 | Airtest | 56.43 | 8.78 | Yes |
| 7 | Airtest | 57.07 | 8.93 | Yes |
| 8 | TestKit | 52.37 | 4.97 | Yes |

All 285 main-cohort sensor samples contained battery/AP/skin readings without
sampler errors. USB power remained connected and Android reported charging
status 2 throughout this cohort. Cold-trial medians were:

| Phone telemetry, C | TestKit | Airtest |
|---|---:|---:|
| Battery start | 34.5 | 34.5 |
| Battery end / peak | 35.65 | 35.70 |
| Battery increase | 1.15 | 1.30 |
| AP start | 35.6 | 35.7 |
| AP peak | 42.55 | 43.35 |
| Skin start | 36.45 | 36.50 |
| Skin peak | 39.40 | 39.45 |

These small differences do not establish a thermal or energy winner. Sensor lag,
USB charging, uncontrolled ambient conditions and different task durations remain
confounds. Power, joules, host peak memory, host CPU utilization and total wire
traffic were not measured. Framework-specific screenshot byte counters would not
be a valid substitute for a shared network measurement.

## Fully initialized follow-up

All twelve tasks passed, four per configuration, with unchanged source/asset
hashes. Capture, recognition and input backends were prepared without sending a
warm-up tap. Setup and teardown are outside the task clock. Both tools still
perform their ordinary fresh focus/input checks, and Airtest retains its normal
API operation delay. Its 100-ms case changes only the state-wait interval.

| Measurement | TestKit | Airtest original waits | Airtest 100-ms waits |
|---|---:|---:|---:|
| Task median, s | 5.182 | 6.012 | **4.809** |
| Task mean, s | 5.139 | 6.036 | 4.978 |
| Task range, s | 4.872–5.320 | 5.612–6.507 | 4.699–5.595 |
| Task empirical p95, s | **5.311** | 6.480 | 5.492 |
| Capture-to-completed-input median, ms | **535** | 783 | 744 |
| Capture-to-completed-input empirical p95, ms | **689** | 876 | 907 |
| Dispatch median, ms | 230 | 505 | 491 |
| Capture wall time median, s | 1.438 | 1.867 | 2.223 |
| Recognition wall time median, s | 1.384 | 0.016 | 0.020 |
| Frame count median | 15 | 10 | 12.5 |

Changing Airtest's wait reduced its observed task median by 20.0%. The resulting
median was 7.2% below TestKit, although its empirical p95 was slightly worse.
This is a small balanced comparison, not proof of a universal speed ordering.
It shows why lower capture-to-input latency alone does not guarantee lower E2E
time: recognition cost, polling, evidence writes and application transitions all
matter. The two scripts also retain different recognition and error policies.
No equivalent cold-start measurement was made for the 100-ms Airtest variant.

| Order | Configuration | Task s |
|---:|---|---:|
| 1 | Airtest original | 6.507 |
| 2 | TestKit | 5.320 |
| 3 | Airtest 100 ms | 5.595 |
| 4 | Airtest 100 ms | 4.906 |
| 5 | TestKit | 5.261 |
| 6 | Airtest original | 6.326 |
| 7 | TestKit | 5.103 |
| 8 | Airtest original | 5.697 |
| 9 | Airtest 100 ms | 4.699 |
| 10 | Airtest 100 ms | 4.713 |
| 11 | Airtest original | 5.612 |
| 12 | TestKit | 4.872 |

The independent arrival images were visually verified at the same waypoint.
All 75 follow-up telemetry samples had the three sensors and no sampler errors.
USB power remained connected, but Android reported status 4 (not charging) in
this cohort, versus status 2 in the main cohort. Do not pool their temperature
changes to claim an efficiency gain. The follow-up is an interleaved steady
sequence after one cooled warmup, not twelve independently cooled trials.

## Independent arrival-image replay

The sixteen independent ADB arrival PNGs were manually reviewed as gameplay,
then replayed through both unchanged recognizers on identical hashed images.
TestKit accepted 16/16 and Airtest accepted 15/16. Airtest missed its own pair-4
cold-arrival PNG, a bright daytime scene. This exposes a recognition false negative
on that independent image, not a failed live task: its preceding live capture
had met the template condition. No thresholds or assets were changed.

These are positive frames only, selected after successful arrivals. They cannot
measure false-positive rate, arbitrary-dialog safety or general reliability.
They also cannot establish when gameplay first became visible during TestKit's
40.98-second task, because those intermediate frames were not all retained.
Repeated `unknown` decisions are a concrete diagnostic lead, not proof of the
entire causal mechanism. A follow-up diagnosis needs time-aligned images around
the first missed arrival before changing the world detector or OCR size.

## Capability and policy comparison

These distinctions come from the current local source and the pinned installed
Airtest source, supplemented by primary documentation. They are not claims that
every capability was exercised by the teleport benchmark. PyPI currently lists
[Airtest 1.4.3](https://pypi.org/project/airtest/) as its release, matching the
installed comparison environment.

| Area | Current TestKit | Airtest ecosystem / this script |
|---|---|---|
| Target platforms | Android implementation; live verification here is Windows host + Android 13 | Android, iOS and Windows APIs; only Android exercised here |
| Authoring | Python, named JSON intent/catalog, application profiles | Python/image templates; AirtestIDE adds visual authoring and execution |
| Locators | Native Android nodes, OCR text/patterns, templates with region/scale/mask/size constraints, coordinates | Core template/coordinate APIs; Poco separately adds UI hierarchy access |
| This game's recognition | Full-screen text classification plus health-bar/waypoint checks | Seven cropped templates searched in recorded regions; explicit landscape normalization |
| Language/layout changes | Text aliases, native attributes and configured visual variants; still requires validation | Template variants or alternative locators/script logic; the measured assets cover this Chinese layout only |
| Ambiguity | Session input rejects multiple matching targets | Core `match_in` returns a best match; scripts can use multiple-match APIs to add rejection |
| Capture here | On-demand JPEG80, native bitmap capture on this supported phone, exact-repeat recognition reuse, fresh fallback | JAVACAP screenshot requests; repeated observations still perform template matching |
| Input here | Fresh geometry/overlay/focus checks batched beside shell tap; no uncertain-input replay | MAXTOUCH plus foreground-package check; API operation delay retained |
| Interruption policy here | Explicit network/login checks, Game Booster handling, bounded retries; generic Session popup/busy rules | Bounded known-template waits; unknown screens time out; this script lacks the additional full-screen error/overlay policies |
| Evidence/reporting | JSON results/events/OCR evidence and PNGs; asynchronous pre-action evidence joins before completion | Script JSON/PNGs here; Airtest also provides HTML reports and IDE workflow |
| Deployment | Python + NumPy/OpenCV; OCR adds models/runtime; fast JPEG helper needs SDK/JDK build tools, otherwise PNG | Python package plus platform capture/touch helpers; chosen JAVACAP/MAXTOUCH already available on this phone |
| Migration | Converter supports an Android scripting subset and reports unsupported constructs | Existing Airtest/Poco/IDE suites can stay in their original ecosystem |

Airtest's documented platform, authoring and report capabilities are described in
its [project overview](https://github.com/AirtestProject/Airtest); its
[core API](https://airtest.readthedocs.io/en/latest/all_module/airtest.core.api.html)
and [Android backend API](https://airtest.readthedocs.io/en/latest/all_module/airtest.core.android.android.html)
describe the selectable backends. [Poco](https://github.com/AirtestProject/Poco)
is a separate hierarchy automation library; some game integrations require its
SDK. Neither Poco nor the IDE was benchmarked here.

TestKit is not a drop-in replacement for the whole Airtest ecosystem. Its
compatibility layer does not implement Poco, iOS, Windows-desktop automation or
arbitrary internal Airtest APIs. Template scores and match ordering can differ.
Conversely, the tested Airtest script can be extended with additional guards;
their absence here is a script difference, not an inherent platform limitation.
Do not infer a general reliability winner from a successful single-scene run.

## Development and verification

The preflight completed all four tasks (one cold and one warm per tool), with
the expected input counts and postconditions. It intentionally skipped the
cooling delay to check the harness, so its timings are not included in measured
summaries. The phone's charging status changed during that development sequence;
charging is recorded for the holdout and remains a potential thermal confound.
Four benchmark completion-contract checks and two temperature-parser checks
passed before measurement. No application/runtime policy or template was changed
for the new comparison.

The measured cohort exposed an adverse case: TestKit pair 3 took 63.26 s for
the cold process and 40.98 s for the warm task. The cold task dispatched Teleport
at 48.35 s, close to pair 2's 48.05 s, but map-to-arrival increased from 4.43 to
15.48 s. In the warm task, Teleport was dispatched at 2.79 s and two accepted
world observations appeared only at 40.51 and 40.87 s. The state log contains
repeated `unknown` observations rather than additional inputs. Both runs are
retained. Final arrival screenshots show normal gameplay. Because intermediate
frames were not all saved, these logs alone cannot split the delay exactly into
game transition time versus conservative gameplay recognition. They do establish
that the earlier short warm holdout did not capture this tail behavior.

## Decision and reproduction

For this task, TestKit has a faster cold-start median and faster completed-input
response. Airtest's lightweight template workflow becomes competitive and has
the best observed warm median once its wait interval is matched. TestKit's
40.98-second warm-app task prevents a blanket claim of superior responsiveness.
Prioritize tracing the long arrival-confirmation delay with time-aligned rejected
frames before further optimization or replacement claims. Do not loosen the
arrival condition merely to improve the benchmark.

Keep Airtest where its IDE, reports, cross-platform APIs or Poco integrations are
important. TestKit offers a narrower Android workflow with native/text/visual
selectors and explicit shared input/popup policies. Choose on those requirements
as well as the measured latency; this comparison does not establish a general
reliability or energy winner.

The benchmark wrappers, audits and this report are saved in the workspace and
uncommitted. TestKit runtime source, the installed wheel, Airtest's original
script, templates and device graphics/network settings were not changed by this
comparison. The phone was restored to Settings after both measured cohorts.

See [benchmark commands](../benchmarks/android/README.md#current-source-comparison-with-airtest).
Raw local evidence is ignored under `artifacts/android-airtest-current/`:
`preflight/` contains four excluded development tasks; `holdout/` contains sixteen
main tasks; `initialized-holdout/` contains twelve follow-up tasks and one excluded
cold warmup. Each measured cohort has hashes, per-trial records, independent
arrival PNGs, `summary.json`, `audit.json` and a reviewed arrival contact sheet.
The sixteen-frame positive replay uses `holdout/world-review-manifest.json` and
the two `world-replay-*.json` outputs. These raw artifacts are not committed.

Six focused benchmark/parser checks passed. Audits verified every task's input
counts, action ordering, loading and arrival conditions, expected image dimensions,
and actual TestKit direct-bitmap/OpenVINO path. No Airtest touch-fallback warning
appeared in the main logs; the initialized cohort explicitly required MAXTOUCH.
There were no failed measured tasks, and no outlier was discarded. No new
cross-device, unseen-dialog, human-speed, memory or power guarantee is implied.
