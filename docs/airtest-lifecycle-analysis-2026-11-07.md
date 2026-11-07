# Airtest across development, deployment and runtime

Reviewed 2026-11-07. Scope: the reusable testing framework and ecosystem.
The Genshin measurements below are examples, not the framework's requirements.

Target contract: black-box games exposing rendered pixels and accepting input,
without a usable UI tree, game SDK integration or semantic application events.
Poco is outside this solution's scope. Native selectors elsewhere in our library
do not solve recognition or synchronization for these games.

**Assessment:** Airtest's principal framework weaknesses are repeated blocking
visual searches, polling-based synchronization, restrictive vision dependencies,
and the amount of wait/existence/recovery logic left to script authors. Visual
recording helps produce a first script; it does not remove those lifecycle costs.
Our narrow optimized benchmark cannot dismiss these broader weaknesses.

Evidence types: **documented** capabilities, **inspected** implementation,
**reported** upstream incidents, **measured** local results, and **inferred**
engineering implications. Online master sources can change. Local implementation
checks used installed Airtest 1.4.3. No new device experiments were run.

**Product boundaries matter.** Airtest supplies Python visual/coordinate
automation for Android, iOS and Windows.
AirtestIDE supplies visual authoring, inspection and execution. NetEase's private
cloud supplies a separate device-management and scheduling layer. Do not credit
all of these to the core library or assume one package deploys the whole system.
[Ecosystem documentation](https://airtest.doc.io.netease.com/en/),
[private cloud](https://airtest.doc.io.netease.com/en/commercial/commercial/).

**Development: quick first scripts, continuing locator and test-design work.**

The awkwardness is also in composing the APIs. `exists()` performs a timed search
with a default three-second timeout, rather than testing one already-acquired
observation. Three sequential absent-target checks can therefore take roughly
nine seconds plus overhead. This is an implication of the defaults, not a new
measurement. `exists(template)` followed by `touch(template)` searches again;
using the returned position avoids that search but requires attention to stale
coordinates. An outer retry loop can multiply these inner waits. The core has
automatic target searching inside `touch`, but does not combine it with the
application's busy states, interruptions, postconditions and retry policy.
[API implementation](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/api.py),
[defaults](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/settings.py).

Consequently, a test author can end up implementing a small scheduler and state
machine for each flow. Our framework should own observation scheduling and bounded
recovery; the test should declare its targets, expected results and permitted
interventions. Application-specific semantics still belong to the test project.

| Capability | Limitation or failure mechanism | Practical response |
|---|---|---|
| Visual capture and Python scripting | Recording interactions does not define business success, fixtures or recovery. Repeated inline templates can spread maintenance across tests. | Name controls centrally; separate actions from expected results; reuse flows and fixtures. This is a design recommendation, not a prohibition in Airtest. |
| Multiscale templates and feature matching | Layout, aspect ratio, localization, animation and similar icons can defeat matching. Lower thresholds trade misses for incorrect acceptance. A match score is not a calibrated correctness probability. | Validate both target-present and confusing target-absent screens. Scope searches and maintain explicit variants. |
| Portable action names | Common APIs do not make OS permissions, app layouts, key names or helper setup identical. | Share test intent; validate each platform adapter and locator set. |
| Screenshots and HTML reports | Reports explain execution; they do not establish that the assertions prove the intended outcome. Image changes also require visual review. | Include meaningful postconditions and an independent expected result. |

The [image guide](https://airtest.doc.io.netease.com/en/IDEdocs/airtest_framework/3_airtest_image/)
documents scaling and threshold tuning. Report export
is covered by the [official FAQ](https://airtest.doc.io.netease.com/en/IDEdocs/faq/2_common%20problems/).

**Deployment: a working IDE environment is only one part of the deployment.**

| Area | Verified capability or problem | Implication |
|---|---|---|
| Python distribution | PyPI lists 1.4.3, released December 4, 2025; its source distribution is 76.7 MB. | Cache a tested installation and assets; test clean installation separately from warm execution. This size is not an installed-memory estimate. |
| Dependencies | Inspected requirements constrain NumPy below 2 and OpenCV contrib to at most 4.6.0.66. | A new Python version or shared computer-vision environment can conflict. Use a dedicated, locked environment and verify wheel availability. |
| Android | ADB authorization, drivers, device settings and capture/input helpers participate in operation. | Record serial, OS, ADB/helper versions and actual backend; check installation, screenshot and input capabilities before the suite. |
| iOS | WDA deployment/signing is part of setup; an already prepared device can be controlled from other hosts. | Preserve and validate the provisioning procedure. A Python install alone is insufficient. |
| Windows targets | Official docs describe embedding failures, changing window handles and desktop-mode matches against IDE previews. | Reacquire the target window and validate the capture region and coordinate mapping. Validate desktop-session behavior in the intended CI environment. |
| Multiple devices | Core supports selecting a current device; a separate parallel runner sample exists. | Device reservation, worker isolation, cleanup, retry scheduling and aggregate results need an explicit owner. |

Sources: [PyPI](https://pypi.org/project/airtest/),
[requirements](https://raw.githubusercontent.com/AirtestProject/Airtest/master/requirements.txt),
[Android FAQ](https://airtest.doc.io.netease.com/en/IDEdocs/3.2device_connection/3_android_faq/),
[iOS setup FAQ](https://airtest.doc.io.netease.com/en/IDEdocs/faq/2_common%20problems/),
[Windows guidance](https://airtest.doc.io.netease.com/en/IDEdocs/3.2device_connection/5_windows_connection/),
[parallel runner sample](https://github.com/AirtestProject/multi-device-runner).

A separate environment contains the dependency conflict but does not solve
in-process compatibility with libraries requiring NumPy 2 or newer OpenCV. That
is a substantive integration limitation. Our dependency policy should test a
declared range of current library versions and keep optional recognition engines
behind adapters; loosening pins without validation is not a compatibility fix.

Inspected `G` stores the active device, device list and logger globally, and
settings are shared. **Inference:** concurrent threads using convenience APIs
can interfere through device selection and logging. Prefer a separate process
and output directory per independent device worker. This does not mean Airtest
cannot run multiple devices.
[Global state](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/helper.py).

The core is Apache-2.0 licensed. Audit the IDE, bundled helpers and any commercial
services separately before redistribution; a core license does not establish
every component's terms. The release page shows real platform work, including
iOS changes; neither issue counts nor release age alone proves abandonment.
[Repository](https://github.com/AirtestProject/Airtest),
[releases](https://github.com/AirtestProject/Airtest/releases).

**Runtime: finding and clicking a target is only part of reliable execution.**

The inspected image wait follows capture -> match -> optional callback -> sleep
-> repeat. It defaults to a 500-ms polling sleep. Failed searches can run multiple
matching algorithms; image loading, resizing and fresh capture add work beyond
the matcher itself. The inspected high-level loop does not share one observation
across independent waits or skip matching based on an unchanged-frame event.
These properties explain potential delay and repeated work; they do not establish
a universal per-match duration.
[Search implementation](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/cv.py).

**Event mechanism gap:** `intervalfunc` runs after a failed search; it is not a
subscription to UI state changes. This finding is scoped to the standard image
wait path, not a claim that every capture backend lacks streaming.
[Search implementation](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/cv.py).

Proposed framework direction: share fresh observations among active conditions;
use frame availability where supported and relevant image-region changes to
trigger reevaluation; invalidate affected cached results and generate condition
transitions from visual observations. These events are inferred by our framework,
not supplied by the game. Capture still requires a stream or scheduled acquisition;
bounded fallback reevaluation is needed when change filtering could miss a useful
transition. Continuous animation can require throttling. Our current
`androidtestkit/session.py` uses polling loops: event-driven observation is a
remaining design gap, not an implemented advantage.

| Mechanism inspected | Failure or cost | Required test policy |
|---|---|---|
| `touch(Template)` finds a position then dispatches input | The convenience API does not enforce target uniqueness, foreground-app identity, observation age or a business postcondition. The screen may change between observation and input. | Check ambiguity and relevant state; verify the result. A wrapper can add these checks. |
| `loop_find` captures, matches, runs a callback, checks elapsed time, then sleeps | A slow capture/matcher/callback can exceed the requested timeout before the next check. | Treat it as a polling deadline, not a hard interrupt; use bounded device operations and an external watchdog where needed. |
| Matching strategies stop when a result succeeds | A plausible wrong target can satisfy a template; more fallback algorithms do not prove semantic correctness. | Test distractors and duplicate controls, and assert a meaningful transition. |
| Configurable polling and operation delays | Excessive waits add latency; overly frequent observation adds capture and recognition work. | Measure complete task time, first-use setup and steady operation separately. |
| Capture selection probes backends and falls back | A fallback can keep a test working with different latency and behavior. Initial probing does not prove recovery from every later disconnect. | Report the selected backend and test mid-run failure separately. |
| Exceptions and explicit waits | They do not decide whether a failed action already took effect or whether retrying it is safe. | Reconnect, observe and reconcile state before retrying input. Never assume a lost acknowledgement means no action occurred. |

Sources: [action API](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/api.py),
[matching loop](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/cv.py),
[settings](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/settings.py),
[capture selection](https://raw.githubusercontent.com/AirtestProject/Airtest/master/airtest/core/android/cap_methods/screen_proxy.py).

Screenshots, encoding, logging and video also consume resources. Retain enough
evidence to diagnose failure, apply retention rules, and keep generated evidence
outside Git. For private remote devices and reports, use the existing tailnet;
do not expose device-control endpoints publicly. These are operational
recommendations, not features supplied automatically by Airtest.

**Concrete problems and the limits of their evidence.**

* **Reported:** [issue 1303](https://github.com/AirtestProject/Airtest/issues/1303)
  shows an Airtest 1.3.6 installation failing on Python 3.14 while preparing
  NumPy. It is not a reproduction against every current environment.
* **Reported:** [issue 1321](https://github.com/AirtestProject/Airtest/issues/1321)
  describes Airtest 1.4.3 `ADB.push` file/directory handling affecting Minicap
  installation. Do not assume this caused our separate local Minicap failure.
* **Measured locally:** default Minicap setup failed; explicit Javacap worked.
  A missing second-entry template caused a development timeout. Both were
  resolved before the measured cohort.
  Supporting workspace note: `docs/android-airtest-benchmark-2026-11-07.md`
  (not included in this analysis-only commit).

**Our narrow measurements do not resolve the framework's general latency costs.**

| Median, seconds | TestKit | Airtest original | Airtest 100-ms polling |
|---|---:|---:|---:|
| Cold process, including app startup/loading | 52.57 | 57.24 | Not measured |
| Fully initialized warm task | 5.18 | 6.01 | 4.81 |

All 28 tasks in the current comparison passed. Supporting workspace note:
`docs/android-airtest-current-comparison-2026-11-07.md` (not included in this
analysis-only commit). A separate warm-app TestKit trial took 40.98 seconds, versus Airtest's
maximum 9.70 seconds in that condition. Airtest also missed one of 16 independent
positive arrival images during replay. That was a false negative on one image,
not a failed live task. These small cohorts on one Android 13 phone, app and
language establish neither general reliability nor a population tail bound.
The measured Airtest workflow used selected templates and custom control logic.
It did not measure the authoring burden or a large stock wait/exists chain. Its
fast result cannot establish that default matching and synchronization are fast.

**Priorities for our reusable framework.**

1. Reduce script orchestration: one action/postcondition contract should own
   target readiness, bounded interruption handling and explicitly safe retries.
   Preserve easy inspection and readable evidence while reducing repeated code.
2. Build on our existing named catalogs, lazy selectors, ambiguity rejection,
   freshness/focus checks and explicit popup policies. They already address parts
   of this analysis; validate them across apps instead of adding another layer.
3. Evaluate shared observations and event-triggered reevaluation with a bounded
   polling fallback. Measure idle work, change-to-action latency and missed or
   duplicated actions; do not equate an asynchronous API with event-driven sensing.
4. Prioritize dependency compatibility, clean deployment checks and actionable failure classification:
   installation, device transport, recognition, input and postcondition failures
   should be distinguishable. Our SDK/JDK helper and optional OCR dependencies
   also need reproducibility; we do not yet have a universal deployment advantage.
5. Keep device-farm scheduling and broader platform support as explicit gaps.
   Our sequential Android runner is not a replacement for AirtestIDE/iOS/
   Windows or a managed device fleet.

Proposed next validation, **not run**: freeze both environments and use rendered
game UI fixtures through pixels only, with normal, missing, duplicate, moved and
overlay-covered targets. Compare stock Airtest and one guarded wrapper, changing
only the guard policy. Predetermine acceptance: zero wrong-target or duplicate
inputs in this small suite, all injected failures classified, healthy execution
within 10% median latency overhead, and watchdog exit within 30 seconds of a
disconnect. Reject the wrapper if any safety case regresses; do not weaken those
cases to meet latency. Separately test clean-host installation and reconnect.
Passing this pilot would justify broader testing, not a production reliability
claim.

Documentation-only impact: this note adds research conclusions and a proposed
validation. Runtime APIs, dependencies, device state and existing benchmark data
were not changed. No new benchmark or compatibility pass is claimed.
