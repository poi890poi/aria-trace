# Genshin: Android TestKit versus Airtest

This is the historical comparison before the later TestKit optimizations. See
the [current-source full comparison](android-airtest-current-comparison-2026-11-07.md)
for new cold-process, warm-app and fully initialized measurements. Do not combine
the old cold-start headline with a new warm-task result.

**All eight measured tasks passed. Airtest completed this workflow in a median 57.90 seconds versus 77.80 seconds for Android TestKit: 25.6% less time. The observed temperature differences do not establish a clear thermal advantage.**

Measured November 7, 2026 on one Samsung SM-G781B (Android 13), USB-powered, fixed brightness 255/255, 2400×1080 landscape. Four scheduled runs per implementation. Each trial force-stopped the app, launched it, handled both entry prompts, and teleported to the same unlocked waypoint at the player's position.

Timing includes a fresh Python process, imports, device setup, the task and cleanup. Helper installation, cooling and independent screenshots are outside timing. OS disk caches and installed helpers were warm.

## Observed results

Values are median (minimum–maximum). All four holdout trials per implementation passed. Development failures are documented separately below. Every measured trial met the cooling gate, starting at 34.4–34.5°C battery temperature.

TestKit requested about 70% fewer frames, but its OCR consumed a median 20.48 seconds; Airtest's template recognition took 0.30 seconds. TestKit spent less time capturing frames. These measurements explain part of the timing difference; the workflows also have different polling and input policies, so this is not an isolated OCR ablation.

| Measurement | Android TestKit | Airtest 1.4.3 |
|---|---:|---:|
| Full process, s | 77.80 (74.62–78.53) | 57.90 (57.55–59.36) |
| Task, s | 76.48 (73.31–77.22) | 57.34 (56.98–58.80) |
| Map tap to verified arrival, s | 11.30 (10.96–11.47) | 5.02 (4.93–5.70) |
| Captured frames | 23 (22–23) | 78 (77–80) |
| Frame capture time, s | 6.40 (5.79–6.70) | 15.01 (14.77–15.23) |
| OCR / template recognition time, s | 20.48 (20.19–20.69) | 0.30 (0.30–0.31) |
| Battery start, °C | 34.5 (34.4–34.5) | 34.5 (34.4–34.5) |
| Battery end, °C | 36.0 (35.9–36.0) | 35.8 (35.7–36.1) |
| Battery peak, °C | 36.0 (35.9–36.0) | 35.8 (35.7–36.1) |
| Battery change, °C | 1.5 (1.4–1.6) | 1.3 (1.2–1.6) |
| Processor AP start, °C | 35.4 (35.3–35.6) | 35.6 (35.2–35.7) |
| Processor AP end, °C | 41.4 (40.3–41.5) | 43.1 (42.5–43.4) |
| Processor AP peak, °C | 42.9 (42.2–43.0) | 43.1 (42.7–43.4) |
| Processor AP change, °C | 5.8 (5.0–6.1) | 7.7 (7.0–7.7) |
| Skin start, °C | 36.2 (36.1–36.3) | 36.3 (36.2–36.5) |
| Skin end, °C | 39.5 (39.1–39.5) | 39.7 (39.5–39.8) |
| Skin peak, °C | 39.5 (39.3–39.5) | 39.7 (39.5–39.8) |
| Skin change, °C | 3.2 (3.0–3.3) | 3.3 (3.2–3.5) |

## Every trial

| # | Implementation | Passed | Full process s | Map-to-arrival s | Battery start → end °C | AP start → end °C | Skin start → end °C | Cooling gate |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | airtest | True | 59.36 | 5.70 | 34.5 → 36.1 | 35.2 → 42.9 | 36.2 → 39.7 | True |
| 2 | testkit | True | 74.62 | 11.47 | 34.5 → 35.9 | 35.3 → 40.3 | 36.1 → 39.1 | True |
| 3 | testkit | True | 78.53 | 11.45 | 34.4 → 36.0 | 35.4 → 41.2 | 36.2 → 39.5 | True |
| 4 | airtest | True | 57.94 | 4.93 | 34.5 → 35.7 | 35.7 → 43.4 | 36.5 → 39.7 | True |
| 5 | testkit | True | 77.35 | 11.15 | 34.4 → 36.0 | 35.4 → 41.5 | 36.2 → 39.4 | True |
| 6 | airtest | True | 57.55 | 4.97 | 34.4 → 35.7 | 35.5 → 42.5 | 36.3 → 39.5 | True |
| 7 | airtest | True | 57.86 | 5.07 | 34.5 → 35.8 | 35.7 → 43.3 | 36.4 → 39.8 | True |
| 8 | testkit | True | 78.24 | 10.96 | 34.5 → 35.9 | 35.6 → 41.5 | 36.3 → 39.5 | True |

## Interpretation limits

- **Charging confound:** trial 8 changed from Android battery status 4 (not charging) to 2 (charging) during the task. Trials 1–7 remained in status 4, although USB power stayed connected throughout. All eight results are retained. A post-run sensitivity check excluding trial 8 gives TestKit medians of +1.6°C battery change, 42.9°C AP peak, and 39.4°C skin peak; Airtest remains +1.3°C, 43.1°C, and 39.7°C. This does not establish a thermal winner, and the exclusion leaves an unbalanced sample.
- These are different complete workflows. TestKit uses on-demand JPEG, local OCR and adaptive polling; Airtest uses Javacap, Maxtouch and bounded template matching. Results do not isolate framework overhead.
- Temperature comes from Android battery and thermal HAL sensors, sampled every two seconds. Sensor lag, different starting temperatures, USB power and uncontrolled room temperature limit causal conclusions. This experiment does not measure energy in joules or establish a general thermal advantage.
- This is a same-location teleport: a verified selected waypoint, one Teleport action, a loading transition, and two gameplay observations. It does not test travel to a different location.
- The sample is four planned runs per tool on one phone and one UI language. No statistical-significance claim is made.
- TestKit's current Chinese start screen was initially unsupported. A bounded start-label plus server-selector check was added and regression-tested before holdout collection. A development run then stalled when OCR omitted one character from the second entry prompt; it was stopped, and an exact alias requiring the release footer was added and tested. No core framework timings were tuned.
- Airtest's default Minicap setup failed. Its working Javacap fallback was selected explicitly before holdout. The first Airtest development run timed out because the second entry prompt was missing; this was corrected before holdout. Development runs are retained separately and excluded from the measured table.

## Verification and decision

All eight independent ADB arrival screenshots were visually reviewed. Task records contain one Teleport action, a loading transition, and consecutive gameplay confirmation. All 258 temperature samples include the three reported sensors with no sampler errors. Source and asset hashes were unchanged across the measured cohort. The game was stopped and the phone screen turned off afterward.

The relevant offline checks passed: 10 framework tests, 19 Genshin tests and two temperature-parsing tests. Windows live behavior is verified; other host platforms were not tested in this benchmark. The new scripts and startup-recognition fixes are saved locally and are not committed.

Retain this benchmark as the baseline for future work. It supports a speed advantage for this Airtest script on this phone. It does not support an energy-saving claim for either implementation, nor a general claim that one framework is faster for all Android tests. No performance tuning was applied to the core TestKit runtime.

## Reproduce and inspect

See [protocol and commands](../benchmarks/android/README.md). The scripts are [Airtest](../examples/genshin/airtest_teleport.py) and [TestKit adapter](../benchmarks/android/testkit_teleport.py).

Raw local evidence: `artifacts/airtest-benchmark/holdout-20261107/`. `metadata.json` records code/assets hashes, environment packages and settings; per-trial directories contain temperature samples, logs, state captures and independent ADB arrival screenshots. These generated files are ignored by Git.

Primary references: [Airtest API](https://airtest.readthedocs.io/en/latest/all_module/airtest.core.api.html) and [Android battery telemetry](https://developer.android.com/reference/android/os/BatteryManager).
