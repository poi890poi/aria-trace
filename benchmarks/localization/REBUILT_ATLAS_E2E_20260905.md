# Rebuilt atlas: production tracking evaluation, 2026-09-05

## Decision

Current tracking is **not yet reliable across the tested sessions**. Route-assisted
local map matching is strong after initialization on five of six recordings, but
one cruise misses a reverse scale transition, outdoor initialization is slow,
the demonstrated transition holds XY for 1.31 seconds, and heading publication
does not meet the 30 FPS deadline. Free-roam fails both reference agreement and
latency. No production behavior or defaults were changed in this evaluation.

This continues archived task **Ariadne Game Tracker**, task ID
`01a03479-43ba-70e2-a81c-d0650649c570`.

## Evaluation protocol

- Atlas: `08b6f2d6-820a-4bfd-875a-6a55d1986a4e`, rebuilt at 09:09 UTC.
- Calibration: `segments-df624035-833-bd07601f-708`; real-time tracking profile.
- Frozen free-roam cohort: Runs 06, 09, 11, 12, 13, 14, 15, 16, 17, 18.
- Frozen route-assisted cohort: Runs 11, 14, 15, 16, 17, 18, using only Run 11
  as the demonstrated route. A separate Run 11 replay enables video recording.
- 17 full replays, 10 distinct recordings, 35,380 source-frame releases,
  29,188 processed frames, and 1,221.97 seconds of recorded input.
- The actual `AcquisitionWorkbench.start_live_tracker` loop runs, including
  latest-frame capture queue, production engine, asynchronous global and
  representation workers, cursor process/IPC, fusion, evidence recording,
  map diagnostics, and Workbench publication. Only the capture adapter is
  substituted with original-cadence video decoding.
- No initial reference pose is supplied. Free-roam receives no route package.
  Route-assisted mode receives the declared demo as search proposals. Other
  sessions' inferred positions are evaluation-only.
- Physical GDI/camera capture, HTTP consumption, HUD/browser rendering, display
  scanout, and game control are not measured. These are recorded-source E2E
  results, not certification of a new physical live trace.
- Source cadence varies by recording. A high fraction within a deadline does
  not imply 30 fixes/second from a source recorded below 30 FPS.
- The deadline criterion is at least 95% fresh output within 33.3 ms; the
  minimum deadline tier is 66.7 ms. Source-frame denominators include queue
  drops after initialization. Cold start and longest holds remain separate.
- References are slow inferred atlas estimates, not external ground truth.
  They share map assumptions with tracking. On Run 11, the same reference also
  supplies demo proposals, so that session is not an independent accuracy test.
  Sparse gaps and mode changes are not interpolated. No fresh post-change
  recording was collected, and no per-frame external heading truth exists.

## Production results

Route-assisted results, with errors in canonical map pixels over scored samples:

| Session | First pose | Fresh XY after init | Reference P95 | Publication P95 | Longest XY hold |
|---|---:|---:|---:|---:|---:|
| 11, demonstration | 2.17 s | 97.9% | 2.81 px | 25.21 ms | 1.31 s |
| 14, cruise / reverse transition | 2.08 s | 97.4% | 58.70 px | 27.77 ms | 0.13 s |
| 15, cruise | 2.09 s | 100% | 3.50 px | 24.48 ms | 0 |
| 16, long cruise | 3.82 s | 99.9% | 3.73 px | 25.07 ms | 0.08 s |
| 17, outdoor laps | 24.27 s | 100% | 3.93 px | 20.84 ms | 0 |
| 18, town laps | 2.17 s | 100% | 1.34 px | 26.86 ms | 0 |
| 11, video recording enabled | 2.14 s | 97.9% | 2.80 px | 24.60 ms | 1.31 s |

Fresh means a measured XY output on a processed frame; it does not certify
correctness. The complete machine report includes fresh output against all
source frames, unavailable frames, loss episodes, frame drops, mode agreement,
steps/jumps, error mean/median/P95/worst-scored, and mean/median/P95/max timing.

Free-roam reports 100% fresh relative-motion-derived XY after initialization on
every session, but reference P95 ranges from **40.71 to 202.46 px**, while
publication P95 ranges from **100.88 to 151.19 ms**. It still computes full-scene
motion and integrates mini-map deltas, whereas route-assisted mode performs
direct current-frame map correlation. Smooth output is therefore not evidence
of correct position. Outdoor free-roam initialization takes 29.89 seconds.

The town lap reference discovers 30 recurrent waypoint groups with 139 visits.
After subtracting natural reference path variation, repeatability residual P95
is **19.05 px free-roam versus 1.30 px route-assisted**. Only 75 versus 136
aligned visits are scored, respectively; dropped frames affect coverage. The
outdoor reference has no qualifying three-visit groups under the unchanged
protocol. Its repeatability result is unavailable, not zero error. Outdoor
trajectory-reference coverage is only about 40% of processed frames.

Heading is available throughout initialized output, but route-assisted heading
age at Workbench publication has **63.87–89.55 ms P95** without video and
67.06 ms with video. This timing includes work after the engine consumes the
cursor worker result. The full 30 FPS XY-plus-heading target therefore fails.
Availability does not establish heading accuracy.

Video recording did not change the demonstrated trajectory or materially worsen
XY latency in this single comparison. Its manifest still reports **56 dropped
frames and 258 repeated frames out of 1,800 written**, so video smoothness is
not a pass. The small latency difference between recording off/on is not a
supported speedup claim.

## Reverse-transition diagnosis

Run 14 begins in town scale and changes to world scale. Native frame 418 visibly
matches the world raster, while the live output remains in town scale. The slow
reference is corroborated by the native map's changed field of view.

The representation observer ran 45 times. The learned zone radius is
**1.6553 px**, but the nearest observed pose was **2.4774 px** from its center.
Every observed position therefore misses the arming zone. Runtime calls
`TransitionController.update` without its supported `position_uncertainty_px`,
although scheduling uses an uncertainty-expanded approach corridor. Mode
observations cannot authorize a switch while the controller remains unarmed.

A controller-only replay reproduces zero switches as shipped. Passing the
recorded position uncertainty switches to world at frame 379 on the same
frozen observations. This isolates the missing uncertainty input as a concrete
candidate correction. It is **not a validated production fix**: changing mode
would change subsequent poses and observations, and reverse transition anchor
handling also needs a full E2E regression. No threshold was relaxed or
production mechanism changed during this test.

Evidence:
`artifacts/poc/workbench-rebuilt-atlas-20260905/reverse-transition/comparison.png`
and `diagnosis.json`. The generator is retained at
`.tmp/inspect_reverse_transition.py` for this diagnostic.

## Native map versus stitched map

Twenty deterministic Run 19 native viewports omitted from composition were
registered against the rebuilt mosaic using translation only. Matched stable
feature residual P95 ranges from **0.89 to 1.25 px** across these views. The
stitch's existing held-out pose-graph residual remains 0.67 px P95, with zero
interior holes. Inspected road, terrain, and city geometry aligns well.

The mosaic is still visually imperfect: changing coastal overlays and source
brightness produce visible boundaries. Frame 1982 shows a large coastal block
boundary that does not exist in the native frame. Source frame 1694 also contains
an external notification near the viewport edge. Stable-feature residuals omit
unmatched dynamic/occluded content, so they cannot certify seam cleanliness.
This supports improving source-content selection and exclusion of overlays,
not another broad geometric or blending change without evidence.

Verified native-scale comparisons and source identities:
`artifacts/poc/workbench-rebuilt-atlas-20260905/native-map-comparison-verified/`.

## Reference reuse

Earlier work saved route packages and already had cursor-cache support, but those XY
packages identify older atlases. They were not silently reused in the new
canonical space. This evaluation adds a content-addressed XY reference cache at
`artifacts/benchmark_cache/atlas_references/<sha256>/`.

Each entry retains route states, descriptors, transitions, rejected samples,
input/configuration/source hashes, reference role, output hashes, and build
duration. Reuse verifies video, timestamps, atlas contents, calibration,
settings, implementation, and cached outputs. Incomplete or corrupt entries
are rejected. The cache is conservative: unrelated changes within its hashed
source directories may also invalidate it.

Ten references contain **1,128 accepted samples from 1,406 sampled frames**.
Run 11 uses 5 Hz and the other sessions use 2 Hz. The recorded build durations
sum to **346.21 seconds**. A fresh verification command reused all ten entries
in **6.09 seconds**, including process startup and input/output hashing. This
avoids rerunning expensive inference; it does not turn inferred estimates into
ground truth or fill their gaps.

## Reproduction and traceability

```powershell
$python = '.tools\standalone-release-py31210\Scripts\python.exe'
& $python -m benchmarks.localization.run_workbench_replay `
  --output artifacts/poc/<new-run>/free-roam `
  --runs 6 9 11 12 13 14 15 16 17 18
& $python -m benchmarks.localization.run_workbench_replay `
  --output artifacts/poc/<new-run>/route-assisted `
  --mode route-assisted --runs 11 14 15 16 17 18
& $python -m benchmarks.localization.run_workbench_replay `
  --output artifacts/poc/<new-run>/route-assisted-recording `
  --mode route-assisted --runs 11 --record-video
& $python -m benchmarks.localization.build_workbench_report artifacts/poc/<new-run>
```

Tooling commits: `4d9b125` (actual-loop replay/cache/native comparison),
`a0bb9f2` (fresh motion versus available pose and aggregate reporting),
`822cdb5` (heading age at Workbench publication). Production code is identical
across both cohorts. Raw recordings and unscored telemetry remain intact.
The free-roam scoring correction was applied to saved traces; initial scoring
artifacts were retained beside corrected versions. Per-run reports preserve
capture implementation identities separately from evaluation implementation.

Full table: `artifacts/poc/workbench-rebuilt-atlas-20260905/REPORT.md`.
Machine aggregate: the sibling `results.json`. Per-run directories contain raw
source/processing telemetry, references, visual incidents, and repeatability
evidence. Cache verification is in `cache-verification.log` and
`cache-verification-timing.json`. The original recordings retain physical
capture timestamps, but this replay does not credit them as newly measured
capture latency.

The next production priorities are transition arming/uncertainty handoff,
moving cold-start consensus, heading result publication scheduling, and
free-roam's relative-motion path. Keep the established direct map matcher;
these results do not justify replacing it or adding smoothing.

Verification: 66 focused benchmark, live tracker, frame-pump, evidence,
architecture, cross-session, and wide-control tests passed. Unrelated untracked
UVC, Android, and rig work was preserved.
