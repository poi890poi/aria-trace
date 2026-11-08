# Genshin benchmark requirements (supplement)

This supplement defines the Genshin startup-to-teleport benchmark discussed in the
latency studies. These are application benchmark requirements, not the generic
[framework requirements](android-requirements.md). See [usage instructions](android-usage.md)
for general framework use, the [benchmark instructions](android-genshin-benchmark-usage.md)
for these experiments, and the [API guide](swiftest.md) for selectors.

## Intended outcome

On one explicitly selected Android phone, launch the installed game, pass the
recognized entry screens, wait for gameplay, open the map, select the intended
unlocked waypoint, teleport once, and verify arrival. Reduce complete elapsed
time and unnecessary observation work without weakening target recognition,
input guards, failure handling or completion evidence.

The benchmark is a same-location teleport with frozen private assets. It does
not measure navigation, first installation, account creation, credential entry,
cleared app data or an Android reboot. The public example can select a visible
waypoint and is not automatically equivalent to that benchmark.

## Environment requirements

| Area | Requirement | Tested configuration / limitation |
| --- | --- | --- |
| Host | Python ≥3.9 as declared in `pyproject.toml`; dependencies compatible with the chosen interpreter | Performance studies used Windows and Python 3.12.10. Other combinations need validation. |
| Device access | Android SDK Platform Tools, authorized USB debugging, explicit serial when running these workflows | Study phone: Samsung SM-G781B, serial `RFCR91GWXLX`, Android 13. Experiment harnesses that enforce this serial must not be treated as portable runners. |
| Basic library | NumPy and OpenCV, installed with the package | No OCR initialization is needed for native-only selectors. |
| OCR | `ocr` extra for ONNX Runtime CPU; `ocr-fast` for the tested OpenVINO path | Current fast extra pins RapidOCR 1.4.4 and permits OpenVINO ≥2025,<2027. Studies used four threads and maximum input side 1280. Library defaults remain two threads and no size limit. |
| JPEG helper | JDK with `java`/`javac`, Android SDK platform `android.jar`, Build Tools with D8 | Configure `ANDROID_HOME` or `ANDROID_SDK_ROOT`. Direct bitmap capture is restricted to Android 13 and one physical display. Unsupported paths fall back; actual acquisition must be reported. |
| Game | Installed `com.miHoYo.GenshinImpact`, existing usable account, working game network, required assets present | Layout, language, waypoint and small-text recognition must match the profile/assets. No unattended login or unknown-dialog acceptance is provided. |
| Airtest comparison | Separate environment with Airtest 1.4.3 and the private template manifest/images | Required for Airtest comparisons, not ordinary TestKit use. |
| Evidence | Writable output directories; sufficient local storage | Raw images, datasets, telemetry and generated helpers stay under ignored `artifacts/` or `.tmp/`. |

## Functional and correctness requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| F1 | Act only on the selected device. Do not clear game data or change account, graphics or network settings as an implicit optimization. | Recorded serial/configuration; reviewed command scope. |
| F2 | Derive actions from current observations and the configured intent. Reject ambiguous or missing targets. | Target/transition tests and live evidence. |
| F3 | Preserve fresh package focus, display geometry, bounds and known overlay checks before guarded game input. Reacquire the target after overlay recovery. | Guard and Genshin tests; device checks. |
| F4 | Do not replay an input after a lost acknowledgement. Only explicitly retry-safe workflow actions may use their bounded retry policy. | Transport/guard tests and recorded input events. |
| F5 | Recognize login/network errors and retain bounded timeouts. Unknown screens or pop-ups must not authorize blind input. | Error cases and full-screen recognition coverage; regional shortcuts need separate coverage evidence. |
| F6 | A normal cold benchmark must record two recognized entry inputs, then minimap, waypoint and exactly one teleport input. | Five sent inputs, expected order, no hidden recovery/retry. The general runtime may report bounded recovery separately. |
| F7 | Completion requires a loading transition followed by two consecutive accepted world observations. UID text alone is insufficient. | Ordered observations plus independently reviewed arrival screenshot. |
| F8 | Finish required evidence writes before reporting task completion. Capture failure must obtain a fresh fallback image or fail, never silently reuse stale pixels. | Evidence files and capture failure tests. |
| F9 | Benchmark harnesses must clean up temporary helpers/forwarding and restore the prior app on their normal completion/error path. | Post-run focus, helper and forwarding checks. A forcibly killed process may require inspection. |

## Timing requirements

**The primary optimization metric is complete cold-process E2E time.** Every table
or speed claim must name its start/end boundaries and whether initialization is
included.

| Metric | Start | End | Use |
| --- | --- | --- | --- |
| Cold-process E2E, primary | Before spawning a fresh worker, with the game force-stopped | Worker exits after verified arrival, evidence and cleanup | Includes imports, helper push/start, app startup, connection and loading waits. |
| Worker task | Harness-defined task start | Workflow returns with required evidence complete | Diagnostic only; its import/setup boundary can differ between wrappers. |
| Warm workflow | After the already-running game and required backends have reached the stated ready condition | Verified arrival and task evidence complete | Report separately. It does not measure game startup. |
| Map-to-arrival | Start of minimap input dispatch | Verified arrival and task evidence complete | Subtask measure; excludes startup and initial loading. |
| Component latency | Explicit capture, OCR or input-stage boundary | That component completes | Never label this full E2E time. |

Cooling, template preparation, cached helper compilation and independent post-run
screenshots are outside the cold-process clock. Helper push/start and forwarding
setup are inside the current cold harness clock. Historical reports can use other
boundaries; retain those labels and do not pool their numbers.

### Current baseline and planning target

Recent startup-inclusive trials took approximately **51–54 seconds**. The earlier
52.57-second median and the latest individual values are measurements, not service
guarantees. See the [cold-start follow-up](android-cold-start-followup-2026-11-07.md).

The provisional **47–49-second** favorable-run estimate is an optimistic model,
not a demonstrated minimum or acceptance guarantee. For example, one 51.67-second
run contained 2.32 seconds across five ready-frame-to-input intervals and 0.68
seconds from the first arrival-frame request through completion. Making those
intervals instantaneous would model 48.67 seconds, assuming earlier actions leave
the game's subsequent response durations unchanged. Some setup and sampling
delay remains, while real guards and verification cannot take zero time.

**Under 50 seconds is a proposed engineering target**, with the same full task and
verification. No four-second cold-start target is supported. A 3–4-second target
in older discussion concerned only teleporting from already-ready gameplay.
An actual lower bound requires independent timestamps for the earliest actionable
screen at every stage; `unknown` from the tested recognizer is not that evidence.

## Experiment and release acceptance

1. Freeze runtime, harness, assets and settings before a cohort; record their
   identities and every scheduled result, including failures and missed gates.
2. Change one mechanism at a time. Use balanced order and a fresh holdout before
   promoting a promising candidate. Reuse the existing 60-second minimum rest,
   battery ≤34.5°C and skin ≤37°C gates; record charging and thermal status.
3. Report success counts, per-trial cold E2E values, mean, median and maximum.
   Label sample percentiles and sample size; tiny cohorts do not establish tails
   or statistical significance. Report warm and component figures separately.
4. Check action order, loading/arrival conditions, independent screenshots,
   fallback status, freshness and relevant regressions. Do not trade those checks
   for a lower timing result. Investigate long `unknown` intervals with rejected
   intermediate images rather than assuming they are game loading.
5. Promote only a reproducible improvement under the unchanged completion
   contract. Lower bytes, fewer OCR calls or lower summed component time alone do
   not establish faster E2E. Report any responsiveness/resource trade separately.
6. Keep concise conclusions and aggregate tables in repository documents. Do not
   commit raw screenshots, telemetry, generated helpers or copied dependencies.
   Inspect staged paths and sizes before any commit/push.

Binary screenshot transport and 500-ms unknown-state polling remain experiments:
neither established a convincing cold-start speedup. Regional HUD OCR remains an
offline candidate without complete error coverage. The 40.98-second warm-arrival
outlier remains unresolved. These limits must remain visible in future claims.
