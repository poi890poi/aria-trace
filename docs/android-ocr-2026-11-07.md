# OCR latency investigation

The previous 0.3.0 result did not establish an efficient OCR backend. This follow-up
compares inference engines and current lightweight models directly, rather than
treating one-second text recognition as an acceptable limit.

## Contract and baseline

Performance change: local OCR execution and optional backend selection. Preserve
full-screen text coverage, original-image coordinates, confidence, aliases, strict
ambiguity, pop-up guards and verified task completion. No VLM, cloud OCR, threshold
relaxation, region omission or cached earlier scene can authorize input. Benchmark
initialization separately from warm inference on fixed image hashes. Require all
previously correct task labels and states on development and unseen holdout images;
inspect any changed label against the actual image. New variants retain native
selectors and OCR fallbacks. Raw experiments stay ignored in `artifacts/android-ocr/`.

The host is an Intel Core i5-9400 (6 cores), Intel UHD 630 and NVIDIA GTX 1650.
The installed baseline is RapidOCR ONNX Runtime 1.4.4 with PP-OCRv4, ONNX Runtime
1.31.1, CPU provider only. The previous faster setting used four threads and a
1,280-pixel OCR input limit. Existing model files total about 16 MB. The recognizer
processes every detected line, including long waypoint descriptions.

## Primary-source candidates

Checked November 7, 2026:

- [ONNX Runtime DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html):
  test the same PP-OCRv4 models on this host's GPU before changing models. Explicitly
  verify the chosen device/provider and preserve CPU fallback behavior.
- [RapidOCR inference engines](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/how_to_use_infer_engine/):
  compare OpenVINO on the same model files to isolate backend cost.
- [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md):
  tiny/small are current lightweight non-VLM candidates. Published timings and
  accuracy use different data/hardware and are not local performance evidence.
  Tiny excludes Japanese; it cannot silently replace a model that must read it.
- [ONNX Runtime thread controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html):
  inspect session spin/memory behavior before assuming thread count alone fixes
  CPU latency. Keep inference engine initialization out of warm timing.

## Results and decision

Use OpenVINO CPU with the existing PP-OCRv4 models. The production adapter keeps
RapidOCR 1.4.4 preprocessing, decoding and model vocabulary. It replaces only the
three per-instance inference sessions. It makes no global runtime patches.

Seven development images cover entry, door, world, map, waypoint selection,
loading and arrival. Each engine runs three passes; the table uses the median of
the last two passes (14 calls). All engines use four threads and a 1,280-pixel
input limit. Times include detection, classification, recognition and decoding;
they exclude screenshot capture and model construction.

| Candidate | Warm median, ms | Decision |
|---|---:|---|
| Existing RapidOCR 1.4.4 / ORT 1.31.1 CPU | 520.5 | Baseline |
| Same pipeline, ORT thread spinning disabled | 427.9 | Smaller gain |
| Same pipeline, ORT CPU memory arena enabled | 461.5 | Smaller gain |
| Same pipeline, ORT 1.24.4 CPU | 462.2 | Paired baseline for DirectML |
| Same pipeline, ORT 1.24.4 DirectML / GTX 1650 | 206.8 | Labels/states unchanged; large first-call cost |
| RapidOCR 3.10 / v4 / ORT CPU | 495.9 | New frontend alone does not resolve latency |
| RapidOCR 3.10 / v4 / OpenVINO | 179.7 | Fast; different frontend |
| RapidOCR 3.10 / v6 tiny / ORT CPU | 283.5 | Model-change exploration |
| RapidOCR 3.10 / v6 tiny / OpenVINO | 136.6 | Rejected: required door prompt not recognized |
| RapidOCR 3.10 / v6 small / OpenVINO | 257.1 | Required states pass; slower than retained v4 |
| **Production adapter / v4 / OpenVINO** | **212.9** | **Selected; existing frontend preserved** |

DirectML selected DXGI adapter 0, NVIDIA GTX 1650. Each model reported DirectML
as its first provider. Its recognized labels and task states matched its
same-version CPU baseline in all 21 calls. GPU inference is a viable alternative,
but is not required by the selected implementation.

The tiny model reads the door prompt as `輕按瑩幕進入`, which does not pass the
existing entry guard. No alias or confidence threshold was relaxed to rescue its
benchmark score. The small model reads that prompt correctly. A newer or smaller
model is therefore not automatically the best replacement for this task.

Eight withheld saved images cover a different game scene and three UI language/
layout fixtures. Their 16 warm calls measured **370.2 ms CPU versus 182.2 ms
OpenVINO**, a **50.8% reduction**. The production adapter preserved every recognized
label and task state in all 45 development/holdout calls (15 distinct images).
All text-box corner coordinates also matched; confidence differences were below
0.00002. Required entry/Teleport target presence was unchanged.
This verifies preservation, not perfect OCR: both backends omit one glyph from a
Traditional Chinese fixture label. Native selectors already cover that fixture.
No claim is made about unseen languages or every possible pop-up.

The earlier modern-frontend experiment measured 111.2 ms on the withheld set.
The shipped adapter measured 182.2 ms; the smaller number is not its performance
claim. Model construction measured 2.71–2.74 seconds for the adapter versus
1.02–1.19 seconds for the original engine. Its first image took 0.29–0.38 seconds
after construction. DirectML's first image took 4.77 seconds after 1.23 seconds
of construction. These startup costs matter for short-lived processes.

## Public behavior and verification

Install `.[ocr-fast]` to enable the optional backend. `LocalOCR(backend="auto")`
chooses OpenVINO if available and reports any CPU fallback. Explicit `cpu` keeps
the reference engine; explicit `openvino` raises if acceleration cannot be used.
A runtime failure under `auto` retries the same captured image once on CPU.
All models must compile before any session is replaced, so failed initialization
leaves the CPU engine intact. Unsupported RapidOCR versions reject acceleration;
the extra pins the validated 1.4.4 interface.

Text aliases, button variants, scoped pop-ups, confidence filters, ambiguity
rejection and bounded input remain unchanged. OCR is still lazy for native-only
tasks. The default resolution and two-thread configuration remain unchanged;
the measurements use the previously validated four-thread/1,280-pixel setting.

All 41 framework tests and 19 Genshin tests pass against both the working source
and installed wheel. The installed environment passes `pip check`. The additional tests cover
initialization failure, strict backend selection on repeated calls, same-image
CPU fallback, original coordinate/confidence preservation, atomic compilation
and per-instance isolation. A sandbox temporary-directory failure was resolved
by using ignored workspace temporary storage; it was not a product failure.

Version 0.4.0 is installed in `.venv/androidtestkit`; source remains uncommitted.
All 16 Python runtime modules in the wheel and installed package match source.
Wheel SHA256: `53d128b1705b007dd7fdb61d183f004631a12a7d624f89fb92496c128fc0e436`.

Raw images, model downloads, environment copies and per-call results remain in
ignored local storage. `benchmarks/android/ocr_backends.py` reproduces the backend
experiment; `--engine legacy-openvino` is the selected production adapter.
The original 0.3.0 source snapshot remains in
`artifacts/android-ocr/baseline-0.3/` with hashes. The initial modern-v4 attempt
failed because its configuration required enums; the corrected `-r2` run is the
table's measurement. Failed attempts are not counted as successful trials.

## Fresh phone comparison

All four installed-package trials passed on the Samsung SM-G781B, Android 13.
The order was CPU/OpenVINO/OpenVINO/CPU. The cohort retained the same cold app
launch, two entry taps, map opening, waypoint selection, single Teleport command,
loading transition and two-consecutive-world confirmation. Each run dispatched
exactly five inputs. Both accelerated runs reported OpenVINO with no fallback.
All four independent arrival screenshots were visually checked. The harness
force-stopped the game and turned the screen off afterward.

| Median metric | CPU | OpenVINO |
|---|---:|---:|
| Complete process, including setup | 70.80 s | 59.50 s |
| Map-open command to verified arrival | 12.96 s | 7.13 s |
| Captured frame available to completed tap | 2.27 s | 1.05 s |
| Total OCR work per task | 30.62 s | 11.86 s |
| Total capture work per task | 15.52 s | 18.71 s |
| Capture requests | 63.5 | 80 |
| Screenshot response bytes | 5.46 MB | 7.62 MB |

Complete-process times were 76.780, 59.595, 59.403 and 64.812 seconds in trial
order. Median full-task time decreased **16.0%**; the map portion decreased
**45.0%**. Ten dispatched inputs per backend give a **53.8%** reduction in median
sampled-frame-to-tap delay. This interval excludes time a button was visible
before the sampled image, so it is not a human reaction-time measurement.
Input dispatch and its checks still take about 0.62 seconds on both backends.

| Trial / backend | Battery start / rise, °C | Skin start / peak, °C |
|---|---|---|
| 1 / CPU | 29.0 / +2.4 | 30.4 / 35.7 |
| 2 / OpenVINO | 30.7 / +1.8 | 32.7 / 36.8 |
| 3 / OpenVINO | 31.6 / +1.5 | 34.1 / 37.6 |
| 4 / CPU | 32.4 / +1.4 | 34.7 / 37.9 |

All cooling gates passed, all samples reported battery status 4 (not charging)
with USB power connected, thermal status stayed 0, and no sampler errors occurred.
Starting temperatures differed, and faster polling increased capture work.
These four runs do not establish lower heat or energy use. They measure phone
temperature, not host CPU temperature. The CPU controls also varied by 12 seconds;
two trials per backend are not a general performance guarantee.

The earlier Airtest median of 57.90 seconds belongs to a different cohort and is
not evidence of a current win/loss against Airtest. The human-speed target remains
unverified without a measured operator baseline. This change removes a measured
OCR bottleneck; capture, input dispatch and app/network loading still contribute.

`artifacts/android-ocr/phone-holdout/aggregate.json` retains aggregate calculations.
All source/input hashes recorded before the cohort remained unchanged at its end.
Task reports confirm installed version 0.4.0 and its site-packages path. Offline
output and source hashes are retained in `offline-provenance.json`. Recompute the
phone aggregate with:

```text
python benchmarks/android/speed_summary.py artifacts/android-ocr/phone-holdout
```
