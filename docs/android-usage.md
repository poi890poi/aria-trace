# Using the SwifTest framework

SwifTest lets you write tests for your own Android app using Python or JSON.
You define controls and expected outcomes. The framework selects the device,
obtains observations, waits for a unique match, sends guarded input and reports
the result. **You do not need a game profile or a benchmark harness to use it.**

This guide builds a small test project. See the
[framework requirements](android-requirements.md) for contracts and limits and the
[API reference](swiftest.md) for more selector options.

## 1. Install and select a device

Install Python and Android SDK Platform Tools, enable USB debugging on your phone
and authorize the computer. The package declares Python ≥3.9; this workspace was
tested with Python 3.12.10. From the source checkout in PowerShell:

```powershell
# Create this environment only if it does not already exist.
py -3.12 -m venv .venv/swiftest
$swiftestPython = Join-Path (Get-Location) '.venv/swiftest/Scripts/python.exe'
& $swiftestPython -m pip install -e .
adb devices
& $swiftestPython -m swiftest devices
$phoneSerial = 'YOUR_DEVICE_SERIAL'
```

Replace `YOUR_DEVICE_SERIAL` with the authorized device's serial. If the Python
launcher is unavailable, use your interpreter's full path to create the environment.
Keep `$swiftestPython` pointed at that environment in later commands. Native selectors
need no OCR extra. Add one only if your tests read rendered text:

```powershell
& $swiftestPython -m pip install -e '.[ocr]'
# Alternatively, for the supported OpenVINO CPU adapter:
& $swiftestPython -m pip install -e '.[ocr-fast]'
```

The optional JPEG helper needs Java/Javac and an Android SDK platform plus D8.
Set `ANDROID_HOME` or `ANDROID_SDK_ROOT` for that SDK. Without the helper, capture
can fall back to PNG; inspect the report's actual backend. `--capture png` selects
PNG explicitly. Other extras are `scrcpy` for supported streaming capture and
`unicode` for the optional non-ASCII input provider.

Check which source is imported with:

```powershell
& $swiftestPython -c "import swiftest; print(swiftest.__file__)"
```

## 2. Create a test project and identify its controls

Use the [starter project](../examples/starter/README.md) as a template:

```text
my-tests/
  ui.json                 logical control definitions
  test_navigation.py      Python test
  test_navigation.json    equivalent JSON test
  assets/                 optional template images
```

Copy it to a new directory, then replace its placeholder app package and resource
IDs with your app's values. The starter is not an installed demonstration app.

```powershell
Copy-Item -LiteralPath examples/starter -Destination my-tests -Recurse
```

Open your app on the phone and inspect the current screen:

```powershell
& $swiftestPython -m swiftest hierarchy artifacts/inspection/ui.xml --serial $phoneSerial
& $swiftestPython -m swiftest snapshot artifacts/inspection/screen.png --serial $phoneSerial
```

Use `resource-id`, `text`, `content-desc` and bounds from the XML to define native
controls. If the app renders its UI into a canvas and exposes no useful nodes,
use text or image selectors instead. The framework does not automatically name
controls from a screenshot.

Create `my-tests/ui.json`:

```json
{
  "elements": {
    "open_details": {"node": {"resource_id": "com.example.app:id/open_details"}},
    "details_title": {"node": {"resource_id": "com.example.app:id/details_title"}}
  }
}
```

The names `open_details` and `details_title` belong to your test. Changing an app's
resource IDs should require a catalog edit rather than rewriting every test.
The CLI loads a sibling `ui.json` automatically; `--catalog PATH` selects another
catalog. Template paths in a catalog are relative to that catalog file.

## 3. Write and run a Python test

Create `my-tests/test_navigation.py`:

```python
def test_open_details(session):
    session.open("com.example.app")
    session.touch("open_details", expect="details_title")
    session.key("BACK")
    session.assert_exists("open_details")
```

Replace `com.example.app` with your installed package. This test opens the app,
waits for one matching control, taps it once and waits for the details title. It
then returns and checks the original control. You write the outcome; you do not
need a fixed sleep between these actions.

```powershell
& $swiftestPython -m swiftest run my-tests/test_navigation.py --serial $phoneSerial --timeout 12 --output artifacts/my-app-tests --failure-screenshot
```

The CLI creates and closes the session. Test functions accept either no arguments
or one argument named `session`. Top-level module code also executes, so do not
call the same test function yourself in a file that the CLI will discover.

Without a catalog, use selector objects directly:

```python
from swiftest import Node

def test_open_details(session):
    session.open("com.example.app")
    session.touch(
        Node(resource_id="com.example.app:id/open_details"),
        expect=Node(resource_id="com.example.app:id/details_title"),
    )
```

The convenience API is equivalent inside the CLI-created session:

```python
from swiftest import Node, open_app, touch

def test_open_details():
    open_app("com.example.app")
    touch(Node(description="Open details"), expect=Node(description="Details screen"))
```

## 4. Write the equivalent JSON test

Create `my-tests/test_navigation.json` alongside the same catalog:

```json
{
  "app": "com.example.app",
  "steps": [
    {"touch": "open_details", "expect": "details_title"},
    {"key": "BACK"},
    {"assert": "open_details"}
  ]
}
```

```powershell
& $swiftestPython -m swiftest run my-tests/test_navigation.json --serial $phoneSerial --output artifacts/my-app-tests --failure-screenshot
```

JSON and Python use the same selectors and session policies. Standard JSON supports
`touch`, `assert`, `assert_absent`, `wait_any`, `swipe`, `key` and `text`; selectors can be inline
instead of named. Use Python for branching, loops, custom logic or advanced retry
options. The ordinary schema is `app` + `steps`; a game's custom `open`/`touch`
schema requires its explicit profile and is not the framework's standard format.

## 5. Choose the right selector

| Screen information | Selector | Example |
| --- | --- | --- |
| Native attributes | `Node` | `Node(resource_id="com.example.app:id/save")` |
| Text drawn on screen | `Text` | `Text(["Save", "儲存"], region=(.5, .7, 1, 1))` |
| Icon or image | `Template` | `Template("assets/save.png", region=(.5, .7, 1, 1))` |
| Child inside a known panel | `Dialog` | `Dialog(Node(resource_id="com.example.app:id/panel"), Node(text="Close"))` |
| Explicit relative location | `Position` | `Position(.8, .9)` uses normalized coordinates |
| Explicit pixel location | `Pixel` | `Pixel(200, 300)` |
| Ordered fallback | `AnyOf` | `AnyOf(Node(description="Save"), Text("Save"))` |
| Equivalent appearances/labels | `Variants` | `Variants(Template("assets/light.png"), Template("assets/dark.png"))` |
| Disappearance | `Absent` / `assert_not_exists` | `session.assert_not_exists(Node(text="Loading"))` |

Prefer useful native attributes when available: native-only operations can avoid
screenshots and OCR entirely. `AnyOf` selects the first matching alternative;
`Variants` checks all alternatives and must not hide distinct targets. Absence is
a condition, not a target to touch. Coordinates alone do not prove that the right
screen is open; use a verified parent/condition and a postcondition where needed.

For native aliases with normalization, use
`Node(text=TextPattern(["Continue", "繼續", "继续"]))`. Native plain strings retain
exact equality. Fuzzy text matching is opt-in and requires a bounded region or
parent; do not solve ambiguity by blindly lowering a threshold.

Capture a template from a known current screen with pixel crop coordinates:

```powershell
& $swiftestPython -m swiftest snapshot my-tests/assets/save.png --crop 100 200 160 60 --serial $phoneSerial
```

Those crop numbers are illustrative; use the bounds from your screenshot. Python
template paths resolve against the session's base directory, which the CLI sets
to the test file's directory. `Text(region=...)` runs OCR on that crop plus eight
pixels of context on each side, clamped to the image. Set `padding=16`, for example,
when labels need more context. Returned boxes use screen coordinates and must
still lie inside the original region. Cropping can change recognition: validate
your labels, languages and edge cases. Without a region, OCR sees the full image.
The default RapidOCR detector can enlarge narrow crops and erase the speed benefit.
For a validated application, `LocalOCR(max_side=1280)` (CLI `--ocr-max-side 1280`)
also selects its maximum-side detector policy. This changes recognition scaling;
measure your actual selectors before choosing it.

Template matching and OCR reuse results only when a fresh acquisition confirms
exactly identical pixels in the relevant region and the evaluation settings agree.
Animation outside the region does not invalidate it. Pixel storage is bounded to
16 MiB and 64 entries per session; capture failures still fail. Continuous animation
inside the region can eliminate the benefit.

## 6. Handle waiting, pop-ups and failures explicitly

```python
from swiftest import Node, TextPattern

def test_continue(session):
    session.open("com.example.app")
    session.on_popup(
        "optional_notice",
        when=Node(resource_id="com.example.app:id/notice"),
        dismiss=Node(text=TextPattern(["Close", "關閉"])),
        max_occurrences=1,
        timeout=5,
    )
    session.busy_while(Node(resource_id="com.example.app:id/loading"), allowance=30)
    session.touch(Node(description="Continue"), expect=Node(description="Next screen"))
```

A registered handler locates its dismissal inside the matching parent, verifies
that the pop-up disappeared, then reacquires the intended target. Handling time
counts against the pending operation's deadline. A known busy state can add only
its bounded allowance. Unknown dialogs are not automatically dismissed.

| Operation | Result / behavior |
| --- | --- |
| `session.wait(target, timeout=5)` | Waits for a unique match; returns its point or raises |
| `session.wait_any({"ready": target, "error": other}, timeout=5)` | Evaluates alternatives on one observation; returns `{"condition": name, "point": (x, y)}` for the first unique match in mapping order |
| `session.exists(target)` | Immediate lookup by default; returns a point or `False`, not necessarily a Boolean |
| `session.find_all(target)` | Returns matching boxes without sending input |
| `session.touch(target, expect=result)` | Sends once by default, then verifies the postcondition |
| `session.touch(target, required=False)` | Returns `False` when the required lookup/result is unavailable; device/focus errors still propagate |
| `session.touch(target, retries=1, retry_safe=True, expect=result)` | Explicit bounded repetition; use only when repeating the action is safe |

Zero-time queries do not dismiss pop-ups. Missing targets raise
`TargetNotFoundError`; multiple targets can raise `AmbiguousTargetError` at the
lookup deadline. Narrow the selector or scope rather than clicking an arbitrary
match. `--timeout` is the default operation timeout, not a suite-wide time limit.

Waits use a fixed maximum sleep of 0.1 seconds, configurable with
`Session(poll_interval=...)` or `--poll-interval`. They no longer increase sleep as
the wait continues. Scrcpy pixel waits use its fresh-frame notification directly,
without an additional sleep. These are frame events, not semantic events from a
game. Capture and inference take additional time and can exceed an operation's
deadline; 0.1 seconds is not an end-to-end response guarantee.

`wait_any` does not send input. If several conditions match uniquely, mapping order
sets priority. A duplicate match in one alternative does not prevent a different
unique alternative from winning. If none wins, the usual missing/ambiguous error
applies. JSON supports `{"wait_any":{"ready":"home","error":"connection_error"}}`.

## 7. Run suites and read results

```powershell
& $swiftestPython -m swiftest run my-tests --serial $phoneSerial --output artifacts/my-app-tests --failure-screenshot
$LASTEXITCODE
```

Directory runs discover `test_*.py` and `test_*.json` recursively in sorted order.
Keeping both starter files runs the same flow twice; retain either format if you
want one execution. The CLI creates one session per file. Functions within a
Python file share that session and app state; there is no automatic data reset,
fixture system or app stop between cases. Establish the state each test needs.
Ordinary case failures are recorded and later cases/files continue.

Each run writes a timestamp-named directory:

```text
artifacts/my-app-tests/<run-id>/
  report.json
  case-0/test_open_details.png     if a Python case failed and capture succeeded
  failure-0.png                   for an applicable file-level failure
```

The CLI prints the exact report path. `report.json` is an array of file results,
with `status`, `cases`, error/traceback when applicable, `elapsed_s` and `metrics`.
Metrics include captures, hierarchy reads, OCR calls/pixels, visual cache hits/misses,
input events, capture bytes,
actual acquisition backend and fallback information. Profile runs have a profile
result instead of ordinary per-function cases. Exit status is 0 on suite success,
1 on recorded test failure, or 2 on a top-level CLI/runtime usage error.

Each file and case also has an `outcome`: `status` (`succeeded` or `failed`),
`last_valid_result`, and `failed_intent`. Automatic action names work without extra
script code. Use a named intent for a useful failure label, `checkpoint` to retain
partial progress, and a function return value for the final result:

```python
def test_claim_reward(session):
    with session.intent("Claim daily reward"):
        session.touch("claim", expect="reward_received")
        session.checkpoint({"reward": "received"}, "Reward verified")
        session.touch("close", expect="home")
    return {"reward": "received", "screen": "home"}
```

If the final touch fails, the report retains `{"reward":"received"}` and identifies
`Claim daily reward` as the failed intent, with its error. Checkpoints survive even
inside a failed group. Without a checkpoint, a failed group retains the previous
completed intent. A case starts with no previous result; a later passing case
cannot overwrite an earlier failure in the file outcome. The CLI prints the failed
intent and last result as well as writing JSON. A process kill cannot produce a
final report.

The `confirmation` field distinguishes `dispatch` (input sent), `observation`
(query completed), `postcondition` (touch result verified), and `checkpoint`
(script-supplied result). An optional lookup returning `False` is a completed query,
not proof of the desired screen. Use assertions or an explicit postcondition when
that outcome is required. Automatic text-entry details omit the typed payload;
user-supplied checkpoint data and exception messages are recorded as supplied.

JSON steps accept the same readable labels:

```json
{"steps":[{"intent":"Claim daily reward","touch":"claim","expect":"reward_received"}]}
```

You can inspect the newest report in PowerShell:

```powershell
$reportFile = Get-ChildItem artifacts/my-app-tests -Filter report.json -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $reportFile.FullName -Raw | ConvertFrom-Json | Select-Object file,status,elapsed_s,error
```

## 8. Embed the library in your own runner

The CLI is optional. Manage a session explicitly in a Python script or pytest test:

```python
from pathlib import Path
from swiftest import Session
from swiftest.intent import load_catalog

project = Path("my-tests").resolve()
with Session("YOUR_DEVICE_SERIAL", base=project, timeout=12) as session:
    load_catalog(session, project / "ui.json")
    session.open("com.example.app")
    session.touch("open_details", expect="details_title")
    session.snapshot("artifacts/embedded/details.png")
    metrics = session.report()
```

For rendered text, pass `ocr=LocalOCR(backend="auto", threads=4)` after importing
`LocalOCR`. Choose `backend="openvino"` when fallback would invalidate a test or
benchmark. Outside the CLI, your runner owns result files, test fixtures and
teardown policy; `session.report()` returns metrics and the current outcome rather than writing a suite
report automatically. Closing the session releases its resources but leaves the
app running unless you explicitly call `session.stop(package)`.

## 9. Extend the framework only where the project needs it

* **Custom selector:** implement `matches(observation, scope=None)` and return a
  list of screen-space `Box` objects. Use observation data lazily. Return no match
  when uncertain; normal ambiguity and input policy still apply to your selector.
* **App-specific workflow:** a trusted profile file exposes
  `run_intent(session, path)` and can interpret a custom intent. Select it with
  `--profile PATH`. Reuse the supplied session; the profile owns its custom state
  machine, deadlines and completion rules. It is not necessary for ordinary tests.
* **Airtest migration:** run `python -m swiftest convert old.air --out converted`
  with the chosen interpreter, review `migration.json`, then run the converted
  entry file. Unsupported APIs require manual changes; conversion is not full
  Airtest/Poco/IDE compatibility.

## Troubleshooting and next references

| Symptom | What to check |
| --- | --- |
| Device absent / unauthorized | USB debugging approval, cable, `adb devices`, selected serial |
| Unknown named target | Sibling catalog or `--catalog` path; spelling in `elements` |
| Target missing / ambiguous | Current UI hierarchy/image, attributes, parent bounds, language and duplicate controls |
| Text unavailable | OCR extra/interpreter, actual backend, small-text quality; explicit aliases |
| Slow capture or fallback | Actual acquisition report, SDK/JDK availability, chosen capture backend |
| Tests affect each other | Shared app state; per-file session is not an app-data reset |
| Import differs from installed wheel | Printed `swiftest.__file__`; current directory and editable install |

For a device-specific runnable example, see the existing
[Samsung Settings test](../examples/settings/test_connections.py). For advanced
selectors, catalog busy/pop-up syntax and capture options, use the
[API reference](swiftest.md). The
[Genshin benchmark supplement](android-genshin-benchmark-usage.md) documents one
application experiment; its timings and private assets are not framework setup
requirements.
