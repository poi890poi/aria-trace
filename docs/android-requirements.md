# SwifTest framework requirements

SwifTest is a reusable Python library and CLI for testing Android
applications. A test describes the intended actions and results; the framework
owns device access, observations, target lookup, waiting, guarded input and reports.
An application profile is optional. The framework must work without Genshin,
AriaTrace's PC components, private screenshots or benchmark scripts.

Start with the [framework usage guide](android-usage.md) and
[copyable starter project](../examples/starter/README.md). The
[API reference](swiftest.md) describes the existing interfaces in more detail.

## Ownership and scope

| Owner | Responsibilities |
| --- | --- |
| Framework: `swiftest/` | Device/session lifecycle, lazy native/pixel/text observations, reusable selectors, ambiguity checks, waits, input policy, declarative execution, suite discovery, reports and conversion tools |
| Test project | Application package, logical UI names, resource IDs, text aliases, templates, expected results, known pop-ups and app-specific recovery rules |
| Optional profile | App-specific state recognition or a custom intent language through `run_intent(session, path)`; reuse the supplied session |
| Benchmarks: `benchmarks/android/` | Frozen experiments, device/environment configuration, independent outcome checks and performance analysis |

Do not move game labels, waypoint coordinates, a particular phone serial or a
game's loading sequence into the generic runtime. An optimization discovered in
one profile belongs in the framework only when its generic contract and evidence
support that move.

## User workflows the framework must support

1. Install the standalone package, select an authorized Android device and inspect
   its native hierarchy or screenshot.
2. Create a test project with named controls in a shared catalog. Keep test intent
   stable when labels, languages, appearance or layout variants change.
3. Write Python tests, ordinary JSON action plans, or embed `Session` in another
   Python runner. Run a file or a directory of tests without an app-specific profile.
4. Select controls using native attributes, rendered text, templates or explicit
   coordinates, including a child inside a verified parent dialog.
5. Wait for the intended condition, dispatch an action once by default, verify its
   expected result, and handle only explicitly registered interruptions.
6. Receive per-case failures, machine-readable results, acquisition metrics and
   optional failure images. Diagnose a failed test without silently accepting it.

## Functional requirements and verification

These are framework contracts, not a claim of exhaustive device compatibility.

| ID | Required behavior | Verification |
| --- | --- | --- |
| FR-01 | Use the explicitly selected device. Automatic selection is allowed only when exactly one authorized device is available. | Device-selection tests and `devices` output |
| FR-02 | Provide a context-managed session. Close owned capture/shell resources; do not clear app data or stop the app merely because a session closes. | Lifecycle tests; live cleanup checks |
| FR-03 | Acquire native hierarchy, screenshots and OCR only when requested. Share each observation's lazy data between its selectors. | Native-only tests record no screenshot/OCR requests; shared-observation tests |
| FR-04 | Reuse exact repeated images/recognition where supported. An acquisition failure must obtain fresh fallback data or fail; it must not turn stale data into a successful observation. | Capture/repeat/fallback contracts |
| FR-05 | Support native, text, template, coordinate, alternative, variant, dialog and absence selectors. Return screen-space bounds and preserve scoped coordinates. | Selector and coordinate tests |
| FR-06 | Require a unique target before input. Missing or ambiguous targets must time out or return the documented optional failure, never select an arbitrary match. | Missing/duplicate-target cases |
| FR-07 | For selector-based touch/swipe, check focus, bounds, observation age and applicable captured-display geometry. Reacquire after registered pop-up handling. Raw key/text operations check focus but do not perform selector observation or pop-up handling. | Focus, stale-frame, geometry and moved-target tests |
| FR-08 | Send a touch once by default. Repetition requires an explicit retry-safe policy and postcondition. A lost command acknowledgement must not cause automatic input replay. | Retry and transport-failure tests |
| FR-09 | Bound waits and known busy-state allowances. A registered pop-up needs a unique parent/child, bounded occurrences and verified disappearance. Unknown pop-ups get no implicit acceptance policy. | Busy/pop-up timeout, ambiguity and recovery cases |
| FR-10 | Support explicit language/text aliases and configured visual variants without hiding distinct targets. Fuzzy OCR must be bounded and scoped. | Alias, fuzzy matching and variant-negative tests |
| FR-11 | Use the same session and selectors for Python and standard JSON tests. Validate JSON steps before app launch or first input. | Equivalent Python/JSON cases and invalid-plan tests |
| FR-12 | Discover `test_*.py` and `test_*.json` recursively. Record each Python case and continue after ordinary case failures. Expose nonzero CLI status on failure. | Suite/report tests |
| FR-13 | Report actual capture/OCR backend, fallback, counts, actions and errors. Store optional failure screenshots without hiding the original failure if capture also fails. | Report and backend-failure tests |
| FR-14 | Keep optional OCR, streaming and Unicode dependencies optional. Native-only use must not initialize an OCR engine. | Dependency and lazy-loading tests |
| FR-15 | Permit custom selectors and optional profiles without requiring changes to the core for every application. | Extension examples and adapter tests |
| FR-16 | Convert supported Airtest scripts into a separate project and report unsupported constructs. Do not silently promise complete compatibility. | Converter cases and migration report |
| FR-17 | Reuse template/OCR results only on fresh, exactly equal region pixels and matching configuration. Bound retained pixels; preserve ambiguity and capture failures. | Exact-repeat, movement, duplicate, configuration and capture-failure tests |
| FR-18 | Evaluate named alternative conditions on one observation. Use frame notifications for streaming pixel waits and a bounded polling sleep otherwise. | Shared-frame and deterministic waiting tests |
| FR-19 | Record success or the last completed result/checkpoint and failed intent. Preserve per-case results and input-versus-postcondition confirmation. | Named Python/JSON failures, partial checkpoints, case isolation and runner report tests |

## Supported interfaces and present limits

* **Python API:** `Session`, selectors and context-bound convenience functions.
  The CLI creates one session per file; test functions in that file share it.
  It accepts functions with no arguments or one parameter named `session`.
* **JSON:** standard plans use `app`/optional `activity` and `steps`. A catalog
  uses `elements`, `busy` and `popups`. Application-specific schemas belong in
  an explicitly selected profile.
* **Execution:** the built-in runner is sequential. It does not supply pytest
  fixtures, test parametrization, parallel device scheduling, automatic app-state
  reset or credentials. External runners can own those concerns using `Session`.
* **Platforms:** this package targets Android through ADB. iOS, Windows desktop
  automation, Poco and AirtestIDE are not supported replacements.
* **Recognition:** aliases/templates must be supplied by the test project. The
  framework does not infer arbitrary unseen UI semantics or unknown dialog actions.
* **Region semantics:** `Text(..., region=...)` runs OCR on the crop with configurable
  context padding (eight pixels by default), restores screen coordinates, and
  constrains accepted boxes to the original region. Cropping can change recognition.
* **Timeout semantics:** session/CLI timeout supplies operation deadlines; an
  ordinary suite has no single enforced wall-clock deadline. Busy allowances and
  profile-owned deadlines must be reported separately. Capture/inference can run
  past the deadline; the default 0.1-second polling sleep is not a response bound.
* **Events:** scrcpy supplies frame notifications, not game-state events. On-demand
  capture still polls. Exact region reuse cannot skip inference on changing pixels.
* **Results:** a completed dispatch is not verified game success. Script returns
  and explicit checkpoints are author claims; assertions and postconditions supply
  observed checks. Reports are finalized by the runner, not persisted after each
  action, and cannot survive an abrupt process kill.

## Runtime prerequisites and portability

| Capability | Prerequisites | Current qualification |
| --- | --- | --- |
| Base package/native selectors | Python ≥3.9, NumPy, OpenCV, ADB, authorized device | Package metadata declares the Python floor; live studies used Windows/Python 3.12.10. |
| Rendered-text selectors | `ocr` extra | Local ONNX Runtime OCR; no cloud account required. |
| Accelerated OCR | `ocr-fast` extra | Current adapter pins RapidOCR 1.4.4; explicit OpenVINO selection fails if unavailable. Automatic selection reports CPU fallback. |
| Optional JPEG helper | JDK, Android SDK platform and D8 | Built on demand. Direct acquisition is limited to the tested Android 13/single-display path; fallback is reported. |
| Streaming capture | `scrcpy` extra and supported scrcpy server | Separate opt-in backend; not a universal latency improvement. |
| Non-ASCII input | `unicode` extra | Uses the optional uiautomator2 provider, which may install an auxiliary IME. |

Use [pyproject.toml](../pyproject.toml) as the dependency source of truth. Distinguish
the source checkout, installed wheel and tested versions. Compatibility must be
demonstrated for new Android versions, display arrangements and application layouts.

## Performance and release requirements

Framework performance must be measured through reusable operations as well as
complete application tasks. A game's startup time is not a framework-wide SLA.

| Layer | Measure |
| --- | --- |
| Initialization | Process imports, device setup, helper/model initialization and cleanup, separately from steady-state operation |
| Observation | Native query, fresh/repeated capture, transfer bytes, decode and OCR costs; identify actual backend and freshness |
| Action | Observation-to-input latency, guard/dispatch time and postcondition completion; distinguish polling lag from inference time |
| Suite | Case success/failure, complete wall-clock time and recorded recovery/retries |
| Application benchmark | Named start/end boundaries, app loading/network waits, completion verification and evidence cost |

Before promoting a performance change, freeze inputs/configuration, compare one
mechanism at a time, preserve failures, and validate surviving candidates on fresh
holdouts. Report sample size, mean/median/tail or maximum, success and important
resource tradeoffs. A lower byte count or OCR time alone is not an E2E speedup.
Do not weaken ambiguity, freshness, focus, retry or verification contracts to
improve a timing number. Keep application-specific evidence clearly scoped.

The recent 51–54-second results and provisional 47–49-second favorable-run model
belong to one Genshin cold-start benchmark. They are neither generic framework
requirements nor measured theoretical limits. See the
[benchmark requirements](android-genshin-benchmark-requirements.md) and
[results](android-cold-start-followup-2026-11-07.md) only when evaluating that task.

Keep generated datasets, raw images, telemetry, caches and copied dependencies
out of Git. Commit concise conclusions, small aggregate tables and deliberately
selected regression fixtures. Inspect staged paths and sizes before commit/push.

## Acceptance suites

Core behavior is covered by `tests/test_swiftest.py`,
`test_swiftest_variants.py`,
`test_swiftest_shell.py`, `test_swiftest_capture.py`,
`test_swiftest_guarded.py` and `test_swiftest_ocr.py`, with related
framework tests alongside them. Live fixtures verify behavior on actual devices;
offline tests alone do not establish hardware compatibility. Game profile tests
and performance harnesses supplement these contracts rather than define them.
