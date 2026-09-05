# Tracking mechanism experiments — 2026-09-05

The strongest improvement came from replacing accumulated relative XY with the
existing current-image atlas matcher, operating without route guidance. Extra
recovery logic did not solve the drift. Several seemingly protective mechanisms
also made transitions worse. The combined candidate substantially improves
recorded position tracking, but still fails the intended reliable-cruising
standard because transitions, cold acquisition and observation age remain weak.

This is a benchmark decision report. Production defaults are unchanged.

## Scope and evidence

- Full recordings run at original source cadence through the actual Workbench
  tracking/publication loop. Only the capture adapter supplies recorded frames.
  These are recorded-source E2E tests, not physical live capture or autonomous
  steering tests.
- Ten mechanisms A–J were tested individually, with targeted combinations and
  removals. Implementation snapshots, hashes, inputs, telemetry and requests are
  saved under `artifacts/poc/tracking-candidates-20260905/<cohort>/runNN`.
- 55 completed full replays cover 3,863.98 seconds of recorded input,
  113,142 source frames and 102,221 processed frames. This count excludes the
  unsupported free-roam video configuration that failed before tracker startup.
- Benchmark implementations were committed before measurement: `d3c8ad2`,
  `901ab4e`, `0b41bd9`. Baseline production revision: `1386101`.
- Frozen slow references were reused from the rebuilt-atlas benchmark, with
  atlas identity and output hashes checked. They never enter free-roam tracking.
  Route-assisted experiments explicitly use the Run11 demonstration package.
- All recordings predate these candidates. Additional recordings in the wider
  comparison are cross-session validation, not fresh post-change holdouts.
- Error units are canonical map pixels against a same-atlas inferred reference.
  They are diagnostic evidence, not external ground truth or path-clearance
  tolerances. No arbitrary 5/10 px threshold determines a cruising pass.

`REPORT.md`, `results.json` and `comparison.json` in the artifact root contain
the complete per-session results. The tables below select causal comparisons;
they do not average away unsuccessful sessions.

## Decisions and confidence

Confidence describes the stated mechanism or observed effect, not a probability
of successful autonomous cruising. Repeated frames and visits are correlated;
they do not justify an artificial statistical confidence percentage.

| ID | Change | Evidence and decision | Confidence |
|---|---|---|---|
| A | Pass observation-time position uncertainty to transition controller | Run14 P95 58.70 → 3.04 px; post-acquisition loss 4.21 s unresolved → 0.57 s recovered. Repeated with the same switch timestamp and trajectory. Keep this correction. | High for the diagnosed missed arming; moderate across crossings. |
| B | Reverse trained transition anchors | Alone does not fix Run14. AB holds from 12.48 s until layer confirmation at 16.02 s; loss becomes 3.50 s versus A's 0.57 s. Reject as implemented. | High for the observed regression. |
| C | Use latest consensus position instead of averaging moving fixes | Free-roam Run11 P95 252.99 → 198.73 px, but Run17 69.55 → 77.92 px. Still grossly inaccurate. No standalone system benefit established. | Low/mixed benefit. |
| D | Preserve recovery consensus during accepted local updates | Regression test reproduces erased consensus. One full Run17 recovery now completes, but residual error is 14.80 px and session P95 80.86 px. Repairing this state bug does not rescue the estimator. | High for state-machine defect; low for tracking-quality benefit. |
| E | Consume same-frame cursor result within remaining source deadline | Run14 heading publication P95 90.93 → 24.73 ms. On 442 shared source frames, heading angles and acceptance are unchanged. Keep the scheduling direction; it does not fix XY. | High for removing the delivery lag; moderate for deadline performance. |
| F | Remove hold when transition is merely armed | Run11 worst error 14.45 → 28.11 px; jumps over 8 px 1 → 7; loss 0.54 → 0.71 s. ABF improves AB but is worse than A. Reject broad hold removal. | High for measured adverse tradeoff; no general safety claim. |
| G | Consume cursor after XY work | Alone gives little benefit. AEG improves Run14 XY P95 timing 36.69 → 29.97 ms versus AE, but smaller timing differences vary by replay. Retain as an interaction candidate, not an independently proven win. | Moderate for scheduling rationale; low for precise incremental size. |
| H | Remove scene KLT and camera-derived minimap rotation trials | Free-roam XY publication P95 falls from 127–204 to 22–23 ms, but Run17 position P95 worsens 69.55 → 126.70 px. CDH is still inaccurate. Reject as a complete tracking solution. | High for cost reduction; mixed accuracy. |
| I | Remove relative accumulation; use existing direct atlas matching with empty route states/transitions/proposals | Run11/17 P95 becomes 2.92/3.93 px, versus fresh free-roam baseline 252.99/69.55 px. Timing becomes 22.07/22.64 ms. Strongest architecture candidate. | High for these recorded comparisons; moderate generalization. |
| J | Wider current-image refinement only on first frame after global seed | With I, Run11 first scored error 25.61 → 3.41 px; first-ten-second worst 27.31 → 3.86 px. Subsequent frames retain the normal radius. Alone does not fix Run14's transition. | Moderate; behavioral regression plus limited delayed-seed evidence. |

I is an estimator-path replacement, not a pure single-line ablation: it reuses
the existing map matcher and thereby also bypasses scene KLT. H separately
isolates much of that computation's cost. These experiments support choosing
the simpler current-image path; they do not isolate every internal contributor
to I's accuracy gain.

## Combinations tested

| Family | Baseline, singles and combinations | Outcome |
|---|---|---|
| Transition | baseline, A, B, AB; F, ABF | A wins. B creates an early hold; F removes that hold but introduces wrong measurements. |
| Heading scheduling | baseline, E, A repeat, AE, G, EG, AEG | E removes stale delivery. G is mainly useful with E, since its wait must account for XY work. A is independently needed for correct layer tracking. |
| Global initialization/recovery | baseline, C, D, CD | None fixes accumulated XY drift. Small apparent improvements are confounded by substantial replay/async variability. |
| Motion removal | H, CDH, I, IJ; J separately on route-assisted runs | Removing scene processing alone is insufficient. Direct current-image matching is the large gain; J improves the delayed seed. |
| Integration | AEGIJ across ten sessions, with targeted repeats and route recording | Strong position improvement, with remaining transition, acquisition and timing failures. Not a cruising release candidate yet. |

The fresh free-roam baseline itself varies from the earlier unchanged baseline:
Run11 P95 202.46 → 252.99 px and Run17 99.31 → 69.55 px. Asynchronous solves and
original-cadence scheduling affect which observations reach the estimator.
Consequently C/D/CD's small relative differences do not establish improvement.
The order-of-magnitude gain from I is much larger than that variation.

## Wider AEGIJ comparison, free-roam without route proposals

Loss below is the longest post-acquisition **reference-based tracking loss**.
It includes wrong layer and confidently wrong XY, not only unavailable output.
Recovery requires 0.5 s of consistent fresh measurements. Unknown references
cannot establish recovery. This metric excludes publication delay; timing is
reported separately. Longest actual control loss remains unmeasured.

| Run | First pose s | Position P95 px | Longest loss after acquisition s | Reference coverage | XY publication P95 ms | Fresh XY + heading within 33.3 ms / source frames |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 4.82 | 3.65 | 0.00 | 44.8% | 39.06 | 90.3% |
| 9 | 4.17 | 33.99 | 1.52 | 75.1% | 35.11 | 82.7% |
| 11 demo | 5.88 | 2.84 | 0.74 | 84.9% | 46.53 | 83.2% |
| 12 | 3.94 | 3.72 | 0.00 | 78.3% | 23.44 | 98.8% |
| 13 | 4.39 | 4.33 | 0.00 | 78.2% | 37.31 | 92.8% |
| 14 reverse crossing | 2.80 | 3.19 | 0.57 | 82.1% | 75.91 | 68.1% |
| 15 | 13.59 | 3.86 | 0.00 | 30.4% | 23.61 | 99.4% |
| 16 | 3.44 | 3.70 | 0.05 | 68.8% | 32.86 | 94.3% |
| 17 outdoor laps | 30.02 | 3.87 | 1.14* | 38.8% | 31.18 | 94.7% |
| 18 town laps | 2.76 | 1.34 | 0.00 | 98.5% | 230.95 | 80.6% |

*Run17's 1.14 s episode contains 1.00 s without reference support. It is time
to verified recovery, not 1.14 s of continuously proven position error. There
are 81.03 s of reference-unknown time across that recording. Zero observed
loss likewise cannot establish reliability across unknown regions. Coverage
uses processed frames, including acquisition frames; acquisition is reported
separately so delayed startup cannot be hidden by a good steady-state score.

All-source timing denominators begin at first available pose and include dropped
frames. They are availability/deadline diagnostics, not proof that the fresh
measurements are correct. In Run11, first available pose is 5.88 s but first
verified acquisition is 7.81 s, including 1.67 s without reference support.

Town laps contain 30 repeated waypoint groups, 139 visits, 134 aligned available
observations. Position repeatability residual P95 is 1.30 px after subtracting
reference displacement between passes. Outdoor reference gaps yield no eligible
repeated waypoint groups, so outdoor lap repeatability is unestablished.

Targeted repeats and the separate recording check:

| Configuration | Position P95 px | Longest post-acquisition loss s | XY publication P95 ms | Joint fresh deadline fraction |
|---|---:|---:|---:|---:|
| AEGIJ Run9 repeat, free-roam | 33.88 | 1.52 | 29.59 | 87.6% |
| AEGIJ Run18 repeat, free-roam | 1.34 | 0.00 | 78.92 | 84.2% |
| AEGIJ Run11, route-assisted + recording | 2.81 | 0.54 | 40.83 | 87.2% |

Run9 repeats the same switch timestamp, nine steps over 8 px and 1.5219454 s
loss. Run18 repeats the low position error and zero measured post-acquisition
loss, but its deadline miss remains despite a large timing improvement.
The recording run uses route proposals and therefore is not a free-roam
recording comparison. Its 1280×720 H.264 output contains 1,800 frames at 30 FPS;
first, middle and last frames decode successfully. The recorder reports 17
dropped and 225 repeated frames, with no error. Thus successful video creation
does not imply every source frame reached the recording without substitution.

## Remaining failures and interpretation

1. **Transition observations arrive after wrong-layer XY has already drifted.**
   Run9 arms at 12.02 s, begins target confirmation at 13.35 s and switches to
   town at 13.63 s. At the switch, position error is still 33.64 px; verified
   recovery starts at 14.56 s. The repeat retains approximately 34 px P95.
   Removing the spatial arming gate is not justified by this case: it did arm.
   The next transition experiment should address observation timing and
   current-image position recovery together, retaining visual confirmation.
2. **Cold global acquisition remains slow.** Run15 takes 13.59 s and Run17
   30.02 s. Run17 rejects ten global hypotheses before its two accepted
   consensus observations. J corrects an available delayed seed; it does not
   make unavailable or rejected global observations arrive sooner. C/D did
   not solve this. A bounded initial-region/current-image hypothesis needs its
   own false-initialization test, without relaxing existing quality gates.
3. **Good position tracking can still deliver stale control observations.**
   Run18's first integration replay has publication P95 230.95 ms, engine
   P95 12.15 ms and source release lateness P95 177.89 ms. For its 424
   publications over 100 ms, mean publication age is 490.14 ms, mean release
   lateness 437.50 ms, mean decode 33.36 ms and engine computation 10.32 ms.
   This locates most extreme delay before tracker processing in that replay;
   it does not establish the cause of every host stall or physical-capture
   latency. Preserve the E2E failures and investigate the source path separately.
4. **Cruising control is not yet measured.** The replay does not issue steering,
   verify turn/stop response, measure path clearance, or establish heading truth.
   Route-stage correctness, wrong/missed turns, steering oscillation, longest
   continuous control loss and intervention-free route completion are needed
   before claiming automatic cruising. Low position error alone is insufficient.

## Cache and reproducibility

The slow inferred cruise references are saved for reuse. This evaluation froze
the existing ten reference caches rather than rerunning inference for each
variant. The cache validates source/configuration identities and result hashes;
the frozen reference list also rejects atlas mismatches. These are reusable
inferred references, not independently certified ground truth.

Example reproduction from the repository root, using the pinned Python:

```powershell
& .\.tools\standalone-release-py31210\Scripts\python.exe -m benchmarks.localization.tracking_candidates --variant AEGIJ --runs 9 11 14 17 18 --mode free-roam --output artifacts/poc/my-combined-replay
```

Choose a new output directory to preserve previous evidence. Use
`--mode route-assisted --record-video` for the separate supported Workbench
recording check. Free-roam recording is rejected before tracker startup;
that attempted configuration is not counted as a successful replay.

Verification after the final experiments: 71 tests passed across candidate
behavior, live tracker, route tracker, minimap transition, Workbench replay
scoring and tracking-loss semantics. No new physical capture or closed-loop
game-control test was performed.

Follow-up production work should first separate direct atlas matching from
route assistance, preserve compatible fallback behavior for other map types,
and integrate the supported transition/scheduling corrections. Keep the
remaining failures as regression targets. Neither the full candidate nor the
rejected mechanisms have been enabled by default in this experiment.
