# SwifTest

SwifTest is an installable Python library and command-line runner for testing Android apps and games. Tests specify what to open, touch, or verify. The library owns device selection, capture, lookup, waiting, retries and failure reporting. App-specific screen recognition belongs in the test project.

Start with the [framework usage guide](android-usage.md) and
[copyable starter project](../examples/starter/README.md). The
[requirements](android-requirements.md) define framework ownership, supported
workflows, correctness, extension boundaries and performance acceptance. This page
provides the detailed API and historical validation reference.

The framework was renamed to **SwifTest** on November 8, 2026. The distribution,
Python import and executable are all `swiftest` (one `t` between `f` and `e`).
Install from this checkout, change `from androidtestkit ...` to `from swiftest ...`,
and run `swiftest` or `python -m swiftest`. Existing JSON catalogs and test plans
keep their format. Default reports go to `swiftest-results/`; `--output` still
selects another directory. The old import/command is not an alias. Earlier
measurement reports retain their original package names, artifact paths and
hashes; those measurements predate this rename.

## Install and run

Use Python 3.9 or newer and Android SDK Platform Tools. Enable USB debugging and authorize the computer on the device. From the repository root:

```text
python -m pip install .
python -m swiftest devices
python -m swiftest run examples/settings/test_connections.py --serial SERIAL
```

These commands work from any terminal with the chosen Python interpreter. For optional rendered-text recognition and streaming capture install with `python -m pip install ".[ocr,scrcpy]"`. For faster local OCR on supported CPUs, install `python -m pip install ".[ocr-fast,scrcpy]"`. No cloud account or VLM is involved.

Version 0.4.0 selects OpenVINO when the `ocr-fast` extra is present, retaining the
same PP-OCRv4 models, preprocessing and decoding. Otherwise it uses ONNX Runtime
CPU. `LocalOCR(backend="cpu")` selects the original backend;
`LocalOCR(backend="openvino")` requires acceleration and fails if it is unavailable.
The default `backend="auto"` falls back to CPU if acceleration fails and reports
the reason. A failed inference retries the same captured image once on CPU.
Session reports include the requested/active backend, model, thread count, size
limit and fallback reason. Native-only tests do not initialize OCR.

CLI equivalents are `--ocr-backend auto|cpu|openvino`, `--ocr-threads 4` and
`--ocr-max-side 1280`. Thread count and size defaults remain two and unrestricted;
validate reduced resolution against your app's small text. See the
[OCR measurements and limitations](android-ocr-2026-11-07.md).

The current source reuses one serialized ADB connection for text shell commands.
Focus and display checks still run for each action. A lost acknowledgement closes
the channel and raises an error; it never automatically repeats the command.
Windows ADB newline expansion is removed using a known response delimiter.
Text-command diagnostics share the response stream. Binary `exec-out` capture,
file transfers and installation retain their separate ADB commands.
For the original transport, use
`Session(device=Android(serial, persistent_shell=False))`.
See the [steady-state latency experiments](android-steady-latency-2026-11-07.md)
for measurements, compatibility limits and the separate capture prototype.

## Build pip packages

```text
python -m pip install "setuptools>=61" wheel
python scripts/build_swiftest.py
python -m pip install dist/swiftest-0.4.0-py3-none-any.whl
```

The script builds the current working-tree source, including uncommitted SwifTest
edits. It copies only package source, the Android helper, packaging metadata and
the API guide into a temporary directory. It builds a source archive, builds the
wheel from that archive, and verifies packaged source before writing both files
and `SHA256SUMS.txt` to ignored `dist/`. Artifact names follow the version in
`pyproject.toml`; `--output PATH` selects another directory. Build scratch files
under ignored `.tmp/` are removed when the script finishes. No upload occurs.

Use a trusted checkout and install the build prerequisites in the same Python
environment that runs the script. No repository-local tool cache is required.
Runtime/OCR dependencies are installed when you install the wheel, not when you
build it. Optional extras also work with the wheel path, for example
`python -m pip install "dist/swiftest-0.4.0-py3-none-any.whl[ocr]"`.

## Python tests

```python
from swiftest import Node, Position, Template, Dialog, open_app, touch, assert_exists

def test_checkout():
    open_app("com.example.shop")
    touch(Node(resource_id="com.example.shop:id/cart"))
    dialog = Template("assets/confirm-panel.png", mask="assets/panel-mask.png",
                      region=(0.1, 0.1, 0.9, 0.9), scale_range=(0.9, 1.1, 0.05))
    touch(Dialog(dialog, Position(0.82, 0.90)),
          expect=Node(resource_id="com.example.shop:id/receipt"))
    assert_exists(Node(resource_id="com.example.shop:id/receipt"))
```

This example illustrates an app's own test and assets. User-defined controls can implement `matches(observation, scope=None)` and return screen-space `Box` objects. Each observation obtains screenshots, rendered text and/or native UI attributes only when a selector requests them. Multiple selectors in one observation share that data. Use `with Session(serial) as session:` in pytest or other Python programs. The CLI supplies this context automatically and resolves template paths relative to each script.

The runner accepts a Python file, a JSON intent, or a directory of `test_*.py` and `test_*.json` files. Python files may contain top-level actions or plain `test_*` functions with zero arguments or one `session` argument. It runs files sequentially, returns a failing exit status when a test fails, and saves a JSON report under `android-test-results/`. The report records each test function separately and continues after case failures. Use `--failure-screenshot` to save visual evidence on failure.

## Selectors and actions

| API | Behavior |
|---|---|
| `Node(resource_id=..., text=..., description=..., class_name=...)` | Match native hierarchy attributes by equality. Each supplied attribute must match. Native actions can run without a screenshot. |
| `Text("Continue")` | Match a rendered line with optional local OCR. |
| `Position(x,y)`, `Pixel(x,y)` | Target normalized or pixel coordinates in the current display or parent dialog. |
| `Template(file, ...)` | Search a bounded region at specified scales. A binary mask excludes changing pixels. `expected` and `tolerance` constrain the center in normalized parent coordinates. `size` constrains width and height in pixels. |
| `Dialog(parent, child)` | Locate exactly one dialog, then resolve its child in the dialog's bounds on the same observation. Supports a verified local button template or a known relative anchor. |
| `AnyOf(...)`, `Absent(target)` | Select a localized alternative or verify disappearance. |
| `touch(target, expect=..., offset=(x,y))` | Wait for a unique target, apply an offset in pixels, screen fractions or target fractions, tap and verify the requested next state. |
| `swipe`, `keyevent`, `text`, `snapshot` | Swipe, send an Android key, type into the focused field, or capture an image. Printable ASCII uses ADB. The optional `unicode` extra routes other text through uiautomator2. |
| `wait`, `exists`, `assert_exists`, `assert_not_exists`, `session.find_all` | Query or assert screen conditions, including a nonthrowing missing-target check. |

Templates accept explicit `scales` or an inclusive `scale_range=(min,max,step)`, numeric thresholds or `"strict"` and `"relaxed"`. The region is `(left,top,right,bottom)` in normalized parent coordinates. Geometry constraints also apply at relaxed thresholds. The native API waits for a unique match before dispatch. A missing optional target returns `False` with `required=False`. The default sends a touch once. Explicitly repeatable actions may set `retries=1, retry_safe=True, expect=...`. Device or focus failures remain errors.

For Unicode input, install `python -m pip install ".[unicode]"`. The optional provider can use the Android clipboard and may install an auxiliary IME when needed. Device lifecycle methods include APK installation, uninstallation, clearing app data, file push/pull and app launch/stop. They act only when invoked by the test.

## Declarative tests

Test authors can keep UI definitions in one sibling `ui.json` or pass a shared file with `--catalog`. A test can then say `{"touch":"connections","expect":"wifi"}`. A catalog may add `"busy":[{"while":"download_indicator","allowance":120}]` to withhold input while an identified loading screen is present. The library applies a bounded extra wait and reduces its polling rate while busy. Zero-time checks still return immediately after their observation.

A project can express the same operations as JSON. Keys such as `node`, `template`, `position`, `pixel`, `dialog`, `absent` and `any_of` select the target. Supported steps are `touch` with an optional postcondition, `assert`, `assert_absent`, `swipe`, `key` and `text`. See `examples/settings/test_connections.json` for a complete runnable example on a Samsung phone with English or Traditional Chinese Settings. Substitute your app's package and selectors for another app.

## Required variant coverage (0.2.0)

Variant handling is a framework requirement. One logical control must be able to
cover language, wording, OCR spacing, appearance, scale and known dialog-layout
changes. Test intent stays stable; the shared UI catalog owns the variants and
interruption policies. This is explicit coverage, not a promise to infer every
unseen UI or meaning. Unknown pop-ups are not automatically accepted.

This is an additive feature in selectors, catalog parsing and the session action
loop. Existing native string equality, `AnyOf` priority, coordinate checks and
explicit touch retries remain intact. No pop-up input occurs without a registered
policy. The main failure risks are a broadened match selecting another control,
clicking through a modal, repeated dismissal and stale coordinates. The contracts
below and the variant tests cover those boundaries.

| Variation | Shared mechanism |
|---|---|
| Languages and synonyms | `Text(["Continue", "繼續", "继续"])`; native `Node(text=TextPattern([...]))` |
| Full-width characters, case and OCR spaces | `TextPattern` applies Unicode NFKC, case folding and whitespace removal by default; each can be disabled |
| Dynamic labels | `Text(regex=r"download\d+%")`; regex must match the whole normalized label |
| Minor OCR errors | `Text("輕按螢幕進入", max_edits=1, region=(.35,.85,.65,.98), min_confidence=.7)` |
| Button theme, icon, size or layout | `Variants(Template(...), Template(...))`, retaining each template's mask, region, scale, size and position constraints |
| Pop-up layouts | `Variants` of known parent selectors; resolve dismissal inside the detected parent, not across the screen |
| Optional interruptions | `session.on_popup(...)` or the catalog's `popups` list; the library dismisses and resumes the pending intent |

Text aliases are explicit: normalization does not translate Traditional/Simplified
Chinese or infer synonyms. List the expected labels. Native string filters still
use exact equality; opt into normalization with `TextPattern`. Native attributes
also accept a list of exact strings. Regex patterns operate on the normalized
input, so use `TextPattern(regex=..., ignore_space=False)` when spaces matter.

Fuzzy OCR is off by default. One edit requires a label of at least six characters;
two edits require ten. A search region or parent dialog is mandatory. Confidence,
geometry and ambiguity checks still apply. Use explicit aliases for short labels
such as OK/Cancel rather than fuzzy matching. OCR errors beyond the configured
budget remain failures.

`Variants` evaluates all alternatives on one shared observation and merges only
nearly identical boxes with compatible input points. Two distinct visible matches
remain ambiguous and cannot trigger input. A child template's click anchor is
preserved. `AnyOf` retains its existing first-available behavior: use it for an
intentional cheap-native-first fallback, and `Variants` when every alternate
candidate must participate in ambiguity checking. Pixel or OCR reads remain lazy;
native-only variants and pop-ups need no screenshot.

### Known pop-ups without branching in the test

```python
from swiftest import Node, TextPattern, Variants

session.on_popup(
    "optional_notice",
    when=Variants(Node(description="notice"), Node(description="notice.v2")),
    dismiss=Node(text=TextPattern(["Close", "關閉", "关闭"])),
    max_occurrences=1,
    timeout=20,
)
session.touch("continue", expect="completed")
```

Handlers run during positive-time target waits, including waits for action
postconditions. Their time counts toward the original operation's deadline. The
library requires one parent and one dismissal target within that parent, taps
once per occurrence, verifies disappearance, and reacquires the intended control
on a new observation. A stuck or over-limit pop-up raises `PopupHandlingError`;
ambiguous parents or buttons raise `AmbiguousTargetError`. It does not retry the
same dismissal tap. Limits apply across the session. `off_popup(name)` removes a
handler so a test can examine that dialog directly.

Zero-time queries such as `exists()` never dismiss pop-ups. `find_all` is a passive
inspection operation. Raw key and text commands do not invoke handlers because a
dismissal could change keyboard focus. Handle predictable dialogs as ordinary
explicit test steps; register policies for optional interruptions. Arbitrary
unknown overlays still require an application-specific selector or failure check.

The runnable [variant catalog](../examples/variants/ui.json) supports a small native
Android fixture in `tests/android_variants_fixture/`. Its test still contains only
`{"touch":"continue","expect":"completed"}`. The fixture varies English/full-width
text, Traditional/Simplified Chinese, colors, panel orientation and the control's
position after dismissal. Build its APK into ignored local storage with Python:

```text
python tests/android_variants_fixture/build.py --sdk PATH_TO_ANDROID_SDK
```

The fixture has no network permissions. After installing the generated APK on an
authorized device, run `python tests/android_variants_fixture/verify.py --serial SERIAL`
using the installed package environment. It checks three layouts and records
independent screenshots under ignored `artifacts/androidtestkit/variants-live/`.

### Variant acceptance results

Verified November 7, 2026 on the Samsung SM-G781B with the installed 0.2.0 wheel.
All 15 Python runtime modules in that wheel matched the working source. The same
catalog and intent passed all three native fixture variants:

| Variant | Result | Input events | Hierarchy reads | Screenshots / OCR calls |
|---|---|---:|---:|---:|
| English, full-width button text, vertical notice | Pass | 2 | 4 | 0 / 0 |
| Traditional Chinese, dark appearance, horizontal notice | Pass | 2 | 4 | 0 / 0 |
| Simplified Chinese, alternate notice identifier | Pass | 2 | 4 | 0 / 0 |

Each run dismissed one notice, reacquired the moved primary button and verified
the completion label. Independent screenshots taken after the measured actions
confirmed completion. Raw results and images remain ignored under
`artifacts/androidtestkit/variants-live-2/`. The temporary fixture was removed
after verification. An initial fixture run failed because its completion label
was clipped to zero-size bounds; fixing the fixture layout resolved that failure
without changing the framework.

The 10 core, 17 variant and 19 Genshin tests passed. Variant tests include masked
and scaled visual alternatives, distinct-button ambiguity, conflicting click
anchors, scoped fuzzy text, stuck and recurring pop-ups, and pop-ups during
postcondition waits. A separate replay of a captured Genshin frame matched the
OCR output `輕按幕進入` (confidence 0.9587) against `輕按螢幕進入` with the documented
one-edit, region and confidence constraints.

These checks establish configured variant handling on one device and bounded
offline cases. They do not establish coverage of arbitrary unseen dialogs or a
temperature improvement. The earlier [Airtest comparison](android-airtest-benchmark-2026-11-07.md)
used the previous framework revision; its measurements are not a 0.2.0 benchmark.

Design references checked November 7, 2026: [Playwright strict locators](https://playwright.dev/python/docs/locators#strictness)
support rejecting distinct matches; [overlay handlers](https://playwright.dev/python/docs/api/class-page#page-add-locator-handler)
support verifying disappearance and charging handling time to the action deadline;
[Android native selectors](https://developer.android.com/reference/androidx/test/uiautomator/BySelector)
support using native attributes and text patterns before image recognition. These
are design precedents, not SwifTest performance evidence. Android screenshot
and native hierarchy acquisition differ from browser DOM queries.

## Capture and authoring

The general Session loop now uses a maximum 0.1-second polling sleep rather than
exponential backoff. Streaming pixel waits consume scrcpy frame notifications
without an extra sleep. `wait_any({"ready": selector, "error": selector})` checks
named alternatives on one fresh observation. Black-box games still need visual
conditions; this does not require a UI tree or game SDK.

Template and OCR evaluation reuse exactly unchanged region pixels in a bounded
session cache. `Text(region=..., padding=8)` performs actual cropped inference and
restores screen coordinates, rather than only filtering full-screen OCR results.
Validate text near crop edges. See the [evaluation results](android-visual-evaluation-2026-11-07.md)
for the controlled workloads and limits.

Scripts can group actions with `session.intent("Claim reward")`, retain partial
progress with `session.checkpoint(value)`, and return a final result. CLI file and
case outcomes retain success or the last valid result and failed intent. Input
dispatch and verified postconditions are reported separately. See the
[usage guide](android-usage.md#7-run-suites-and-read-results) for examples.

OCR has explicit latency controls. The default retains the original image-size
policy and two CPU threads. For an application validated at a smaller OCR size:

```python
from swiftest import LocalOCR, Session
from swiftest.intent import load_catalog

with Session("SERIAL", ocr=LocalOCR(max_side=1280, threads=4)) as session:
    load_catalog(session, "ui.json")
    session.touch("continue", expect="completed")
```

The CLI equivalents are `--ocr-max-side 1280 --ocr-threads 4` on `run`.
RapidOCR returns coordinates in the original captured image; the input and
geometry checks still use that image. A smaller OCR input can miss small text,
so validate languages, button labels and pop-ups before choosing this setting.
No image-size limit is enabled by default. Native-only tests still do not load OCR.

The Genshin profile uses separate cooldowns for its recognized launch and entry
screens, so the second prompt need not wait for the first prompt's retry timer.
Each logical screen keeps its 15-second retry interval and the overall three-tap
limit. Interactive polling waits at most 0.10 seconds between observations;
identified downloads and preparation retain adaptive waits. Recognition, capture
and input costs are additional to this wait. This is not a 0.10-second response
guarantee. See the [speed investigation](android-speed-2026-11-07.md) for measured
results and the distinction between task time and operator reaction time.

| Mode | Requirements and tradeoffs |
|---|---|
| `auto` | On-demand, full-resolution JPEG with exact identical-frame suppression. Android 13 with one physical display tries direct acquisition and native bitmap comparison; unsupported devices use `screencap`. If the helper cannot build or run, the session uses a fresh ADB PNG and records the reason. |
| `jpeg` | Explicit legacy `screencap` acquisition through the JPEG helper, with PNG fallback. Useful for comparison or compatibility. |
| `png` | On-demand lossless ADB screenshots. Requires only ADB. |
| `scrcpy` | Low-latency continuous H.264 display capture with Python/PyAV. Supply a scrcpy 4.1 server via `--scrcpy-server PATH` or the `SCRCPY_SERVER` environment variable. Streaming consumes bandwidth while active. |

The optional JPEG helper is built with Java and D8 invoked from Python. It runs temporarily under the ADB shell on the device. The Python host invokes executables directly. Native selector tests avoid starting a capture backend unless they need pixels. Reports distinguish screenshot/stream payload bytes, setup bytes, screenshot requests, OCR calls, and native hierarchy reads. Byte and time measurements do not measure energy in joules.

`capture_acquisition` identifies the actual acquisition path. The Genshin profile
batches fresh geometry, overlay and focus checks beside each tap. Extra Session
popup/busy rules retain the shared lookup path. An uncertain input is never
replayed. Evidence PNG writes overlap game transitions and finish before the
task returns, including failure paths. See the
[warm E2E investigation](android-e2e-latency-2026-11-07.md) for device results and
the tested limits. These source changes are newer than the installed 0.4.0 wheel.

```text
python -m swiftest hierarchy ui.xml --serial SERIAL
python -m swiftest snapshot button.png --serial SERIAL --crop 100 200 90 40
python -m swiftest run my_test.py --serial SERIAL --capture scrcpy --scrcpy-server PATH
```

## Airtest migration

```text
python -m swiftest convert old.air --out converted
python -m swiftest run converted/old.py --serial SERIAL
```

The converter inspects Python syntax without executing the source. It writes a separate project, copies image assets, preserves supported call sites and writes `migration.json`. An exit code of 2 on a completed conversion means issues need review. Dynamic code, internal APIs, unrecognized imports, and unsupported call signatures require manual migration. Run trusted test code only.

The independent compatibility module covers Android connection/setup, `Template`, `touch`, `click`, `swipe`, `exists`, `wait`, `find_all`, visual assertions, key events, screenshots, text, app start/stop, installation, file transfer, sleep and equality assertions. `exists` returns coordinates or False, and `wait` returns coordinates or raises a target error. The matcher uses this library's scoring. Recorded resolution and `record_pos` are scaling and ranking hints. Converted scores, multi-match ordering and timing can differ from Airtest. The compatibility URI currently supports local ADB and one selected device per test session. Poco, AirtestIDE, iOS and Windows-desktop test APIs are outside the Android package's scope.

## Examples and evidence

The Settings tests use only the public API. The Genshin teleport workflow is an example with its own screen-recognition profile and intent. Use the same CLI with an explicit trusted profile file exposing `run_intent(session, path)`. See the Genshin prototype note for the earlier device measurements. The framework runtime has no dependency on that example, AriaTrace's rig packages or the repository's tool cache.

Public behavior references: [Airtest Android API](https://airtest.readthedocs.io/en/latest/all_module/airtest.core.api.html), [scrcpy stream protocol](https://github.com/Genymobile/scrcpy/blob/master/doc/develop.md), [RapidOCR](https://github.com/RapidAI/RapidOCR).

## Verification in this workspace

Live checks used the Samsung SM-G781B on November 7, 2026. A native Settings test opened Connections, verified Wi-Fi, and returned with three hierarchy reads and zero screenshots. A temporary `.air` project with two visual assets converted without warnings and passed on the phone. The visual test used six JPEG captures and transferred 465,728 bytes of screenshot responses. A separate scrcpy test acquired a real Settings frame.

The adapted Genshin profile passed using a position supplied for a waypoint hidden by the player icon. The run observed map selection, one Teleport command, a loading screen and two subsequent gameplay frames. It took 16.06 seconds and transferred 1,414,830 bytes of screenshot responses across nine observations. Candidate detection correctly abstained when the waypoint outline was obscured. A test author can supply a known position or template in that case.

Host execution was verified on Windows with Python 3.12. The CI workflow defines the offline contracts for Windows, Linux and macOS. Live hardware results here cover the Windows host. The optional Unicode provider was not exercised on the connected phone.
