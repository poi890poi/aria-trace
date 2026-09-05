# CPU XFeat versus the current tracking methods

Follow-up: [explicit cursor/exterior input masking](XFEAT_MASK_RESULTS_20260906.md)
tests the pixel masking that was absent from this initial feature-location-only
comparison.

Keep the current SIFT global initializer and current-image atlas refinement.
None of the tested XFeat configurations improves this system. Native-resolution
XFeat never initializes in either 45-second E2E clip. The best partial XFeat
variant, 2× extraction with the existing ratio matcher, initializes the demo at
41.14 seconds versus SIFT's 2.17 seconds. Production code and dependencies are
unchanged; the learned models and their CPU environment are experimental only.

## Paired feature-proposal comparison

Each main configuration sees the same 46 samples from the first 45 seconds of
Run11 (demo) and Run17 (outdoor). Queries are independent, unrestricted searches
over both atlas layers. No reference position or route proposal enters them.
The pose geometry, correlation, coverage and ambiguity checks are unchanged.

| Configuration | Demo accepted / 46 | Outdoor accepted / 46 | Demo query mean / P95 ms | Outdoor query mean / P95 ms |
|---|---:|---:|---:|---:|
| Current SIFT + ratio 0.80 | 44 | 21 | 221 / 623 | 95 / 146 |
| XFeat + ratio 0.80 | 0 | 0 | 49 / 73 | 62 / 85 |
| XFeat + mutual nearest neighbors | 0 | 0 | 58 / 69 | 66 / 84 |
| XFeat at 2× + mutual nearest neighbors | 0 | 0 | 170 / 189 | 158 / 186 |
| XFeat at 2× + ratio 0.80 | 11 | 0 | 218 / 638 | 109 / 122 |

These are whole two-layer query wall times, excluding decode and setup, not
feature-extraction times or 30 Hz update-loop times. Failed XFeat proposals
usually skip map correlation, so their shorter time is not a useful successful
localization speedup. Mean process CPU times for the baseline are 255/160 ms
(demo/outdoor); 2× XFeat + ratio uses 326/218 ms. Windows process CPU accounting
is quantized, and includes all threads in the process.

XFeat + official LighterGlue was also checked on two starting samples per
session, with the native resolution and default match confidence 0.1. It accepts
0/2 in each session. Mean full-query times are 3,852 ms demo and 3,968 ms outdoor.
This was stopped at the bounded check; four frames do not establish its behavior
on every scene. No larger LighterGlue replay or tuning sweep was warranted.

There are 464 probe evaluations across 94 unique source observations, including
the extra endpoint samples from the one-second LighterGlue checks. Corresponding
observation hashes are checked. The 464 evaluations are not 464 independent
scenes or statistically independent trials.

## Actual Workbench replay

Only the frame source is replaced with original-cadence recorded delivery.
The current direct atlas-refinement path, transitions, heading worker,
initialization consensus and publication loop remain active. All rows below
use free-roam, without route guidance, and are limited to the first 45 seconds.

| Configuration | Session | First pose s | Longest loss including acquisition s | Longest loss after acquisition s | Unavailable / processed frames |
|---|---|---:|---:|---:|---:|
| Current SIFT | Demo11 | 2.17 | 2.17 | 0.402 | 64 / 1,341 |
| Current SIFT | Outdoor17 | 22.42 | 24.43 to verified acquisition | 0.000 within clip | 672 / 1,349 |
| XFeat + mutual matching | Demo11 | Never | 44.96, unrecovered | Not applicable | 1,338 / 1,338 |
| XFeat + mutual matching | Outdoor17 | Never | 44.95, unrecovered | Not applicable | 1,349 / 1,349 |
| XFeat at 2× + ratio | Demo11 | 41.14 | 41.14 | 0.000 within remaining 3.8 s | 1,220 / 1,340 |

The last row's position P95 is 1.30 px versus SIFT's 2.96 px. This is selection
by late availability, not better tracking: XFeat reports only 120 final frames
and misses the earlier route and transition. The apparent zero post-acquisition
loss is similarly uninformative over that short remainder.

SIFT's steady publication P95 is 20.01 ms demo and 24.61 ms outdoor on these
clips. The late XFeat row is 20.60 ms on its 120 final frames. No comparable
steady latency or position P95 exists for the never-initializing XFeat rows.
Neither unavailable poses nor their cheap processing count as timely control.

An additional baseline startup replay used the runner's route-assisted default
and initialized Outdoor17 at 22.35 s. It is preserved as `baseline-startup`,
explicitly excluded from the free-roam paired table. Total evidence is six
45-second replays, 269.73 source seconds. No full-lap or fresh holdout XFeat pass
is claimed. Broader tests stopped after the development comparisons failed.

## What the diagnostic checks establish

- The adapter gives exact identical-image correspondence: 233/233 demo and
  219/219 outdoor matches. Its mask and coordinate conversion tests also pass.
- On the demo's first native minimap, full-atlas mutual matching gives 56
  matches, none within 3 px of the reference-implied location. Against a
  reference-centered crop, 22/71 are within 3 px and RANSAC finds 31 inliers.
- On outdoor frame749, the same comparison improves from 5/68 near the
  reference-implied location to 47/91 when using the small reference crop.
- Native-resolution ratio matching fails to obtain six matches in all 184
  layer queries. Mutual matching fails the existing 0.60 inlier-ratio check in
  all 184. At 2×, mutual matching still fails that check in every layer query.
- The native minimap is 138×138, contains cursor/UI overlays and fog, and
  differs visually from the atlas. These results support poor correspondence
  across this appearance/context change. They do not isolate blur, feature
  context, image normalization or training domain as the unique cause.

The reference-centered crops are post-run diagnostics, not an admissible global
initializer. They are never supplied to the benchmark estimator. Visual pairs
were opened and inspected at:
`artifacts/poc/tracking-xfeat-20260905/diagnostics/run11-native-and-reference-crop.png`
and `run17-native-and-reference-crop.png`.

## Decision and confidence

| Candidate | Decision | Confidence and scope |
|---|---|---|
| XFeat + ratio | Reject as the current global proposal replacement | High for these inputs/settings: zero accepted development queries. |
| XFeat + mutual matching | Reject as the current global proposal replacement | High: zero accepted probes and no acquisition in both E2E clips. |
| 2× XFeat + mutual matching | Reject | High for the measured samples: greater cost, still no accepted fixes. |
| 2× XFeat + ratio | Reject for integration | High for the measured regression; the exact 41.14 s startup is scheduling-dependent and was not repeated. |
| XFeat + LighterGlue | Stop this unrestricted full-atlas configuration | High for measured cost; limited quality confidence from only four starting frames. |

The current method remains the strongest tested integration described in
[the prior results](TRACKING_INTEGRATION_RESULTS_20260905.md). This experiment
does not retest every historical relative-motion or transition ablation, and
does not invalidate XFeat for different image sizes, tiled retrieval, nearby
gameplay keyframes, fine-tuning or other games. Those would be separate methods
with new evidence requirements. No pretrained-model benefit has been established
for the present global initialization problem.

## Reproduction and limits

- Production baseline: `f607fa6`; frozen initializer/query sources and executed
  method snapshots accompany the runs. Earlier full harness source is preserved
  as `xfeat_cpu_initial.py`, and later runs include `candidate-source/harness.py`.
- Official [XFeat repository](https://github.com/verlab/accelerated_features),
  revision `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`.
  `xfeat.pt`: 1,544,758 parameters, 6,247,949 bytes; SHA256
  `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`.
- Python3.12.10, CPU-only PyTorch2.14.0, one PyTorch inference thread,
  OpenCV5.0.0 with six threads, NumPy2.5.2. LighterGlue uses upstream-pinned
  Kornia0.7.2. All learned matcher parameters were verified present. The training
  checkpoint's extra extractor entries are excluded; Kornia's deterministic
  `confidence_thresholds` buffer comes from its constructor. A preliminary
  overly strict checkpoint check aborted before inference and is not a result.
- Model setup is separate. In the first native XFeat probe, weight loading took
  83 ms; the world/town localizer setup took 309/2,434 ms, including atlas feature
  extraction. Other setup measurements are in each candidate manifest.
- Frozen slow references were reused and hash checked; current atlas and
  calibration pixels were checked against their saved input hashes. These
  references use SIFT and the same atlas, so reference agreement can favor the
  baseline and is not external ground truth. Earlier outdoor truth remains
  unknown. Total output unavailability does not depend on reference accuracy.
- 25 focused tests passed: adapter masks/resizing/MNN equivalence, Workbench
  replay, tracking loss and production integration. No production code changed.
- Evidence: `artifacts/poc/tracking-xfeat-20260905`. `comparison.json` contains
  aggregates, full loss episodes, CPU/wall distributions and reference identities;
  `COMPARISON.md` is generated from preserved raw rows. Negative candidates,
  setup costs and unscored intervals remain available.

Example, using the isolated environment (which reuses the production NumPy and
OpenCV packages):

```powershell
& .\.tools\xfeat-cpu\Scripts\python.exe -m benchmarks.localization.xfeat_cpu --variant xfeat-mnn --action probe --runs 17 11 --max-seconds 45 --output artifacts/poc/NEW-UNUSED-DIRECTORY
& .\.tools\xfeat-cpu\Scripts\python.exe -m benchmarks.localization.build_xfeat_report artifacts/poc/tracking-xfeat-20260905
```
