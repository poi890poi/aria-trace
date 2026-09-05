# Live tracking benchmark — 2026-09-05

## Decision status

Production is unchanged. The recorded-video localization core has passing
candidates, but the latest real workbench trace does **not** pass the full
capture-to-publication control target.

Pending implementation candidate:

1. gradient `TM_CCORR_NORMED`, 12 px local radius, no global score threshold;
2. accepted-time physical continuity gate;
3. active-layer search outside a learned transition zone;
4. all-layer search inside the zone, with two consecutive target-layer wins;
5. hold the previous pose while a representation change is unconfirmed.

Laplacian CCORR is retained as a secondary candidate. It reduced the largest
transition-reference outlier, but its lower long-run fresh coverage and larger
compute spike make it a higher-risk default.

## Evidence contract

- Estimation is chronological and causal.
- A declared start pose may initialize a replay once.
- Future reference positions never feed tracking.
- Demo timing, heading, motion vector, and progress never feed tracking.
- Held output is available state, not a fresh localization fix.
- Accuracy reports mean/median/P95/worst only where a sparse post-run map
  reference exists.
- Run 09 uses an explicitly labeled transition-calibration proxy, not ground
  truth.
- Unlabeled Runs 14–16 support availability and latency, not accuracy claims.

## Layer-policy experiment

Only the map-layer search policy changed. The estimator was gradient CCORR,
12 px radius, score threshold 0, accepted-time continuity.

| Replay | Policy | Fresh | Mode agreement | RMSE | P95 | Worst | Core P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run 11 transition | sticky | 88.89% | 60.16% | 71.63 px | 148.66 px | 167.33 px | 1.89 ms |
| Run 11 transition | all layers | 99.39% | 100% | 2.06 px | 3.49 px | 16.70 px | 3.24 ms |
| Run 11 transition | transition zone | 99.39% | 100% | 2.06 px | 3.49 px | 16.70 px | 2.16 ms |
| Run 09 transition proxy | sticky | 86.17% | 62.99% | 47.91 px | 103.02 px | 123.13 px | 1.89 ms |
| Run 09 transition proxy | all layers | 97.17% | 100% | 3.26 px | 3.33 px | 23.82 px | 3.33 ms |
| Run 09 transition proxy | transition zone | 97.17% | 100% | 3.26 px | 3.33 px | 23.82 px | 2.42 ms |

On Runs 12, 13, 17, and 18, sticky, all-layer, and transition-zone policies
produced identical XY errors. Transition-zone search therefore recovered the
transition behavior without a measured fixed-scale accuracy regression and
without paying the all-layer cost throughout the route.

Confirmation counts 1, 2, and 3 and zone-radius floors 6, 12, and 24 px
produced the same trajectory on both transition replays. Two confirmations and
a 12 px floor remain the conservative candidate.

## Estimator screen

The initial screen changed one factor at a time on Runs 11, 12, and 16:
gradient/intensity/Canny/Laplacian features, CCORR/phase matching, 8/12/18 px
radii, score 0/0.50, and recovery/no-recovery variants. Intensity, phase
correlation, and Canny were rejected before wide validation:

| Candidate | Fresh | Reference P95 | Worst |
|---|---:|---:|---:|
| intensity CCORR | 94.3% | 107.15 px | 155.08 px |
| gradient phase correlation | 95.6% | 6.16 px | 38.34 px |
| Canny CCORR | 99.6% | 3.60 px | 23.51 px |

Four finalists were replayed over Runs 11–18: 15,933 chronological attempts,
including two ordinary routes, three fresh/unlabeled areas, two long
repeatability sessions, and one route containing a representation transition.
There were 10,587 sparse-reference comparisons.

| Finalist | Fresh | Worst-session fresh | Serial decode-to-XY P95 | Reference P95 | Worst |
|---|---:|---:|---:|---:|---:|
| gradient CCORR, r12, score 0 | 99.84% | 96.99% | 4.88 ms | 2.87 px | 16.70 px |
| gradient CCORR, r8, score 0 | 99.85% | 97.42% | 4.82 ms | 2.87 px | 16.70 px |
| Laplacian CCORR, r12, score 0 | 99.43% | 96.99% | 4.78 ms | 2.94 px | 12.52 px |
| gradient r12 with 0.50 final hold | 88.65% | 62.93% | 4.86 ms | 2.87 px | 16.70 px |

The 0.50 score gate is rejected: it discards many correct outdoor matches.
Radii 8 and 12 are statistically tied here; 12 retains a larger motion basin
for dropped or delayed frames. Laplacian lowers the single largest reference
error but has slightly lower availability and a 55.48 ms serial worst case,
versus 32.81 ms for gradient r12.

Run 09 provides a second, proxy-only transition check. Gradient transition-zone
tracking reached 97.17% fresh, 3.33 px proxy P95, and 23.82 px worst. Laplacian
reached 98.00% fresh, 4.50 px P95, and 18.39 px worst. This supports retaining
Laplacian for further live testing, but does not turn the proxy into truth.

## Resolved contradiction: Laplacian and continuity time

An older wide report showed catastrophic Laplacian drift. A one-variable
reproduction identified the cause:

| Run 17 | Fresh | P95 | Worst | Final-gate rejects |
|---|---:|---:|---:|---:|
| Laplacian + frame clock | 52.55% | 310.64 px | 329.64 px | 47.45% |
| Laplacian + accepted-time clock | 98.65% | 3.42 px | 5.65 px | 1.35% |
| Gradient + either clock | 100% | 3.48 px | 5.59 px | 0% |

The frame clock reset elapsed time after every held frame, so the physical gate
never allowed the tracker to catch up. Accepted-time uses time since the last
accepted pose. The old result remains valid for the rejected frame-clock
combination; it is not evidence against Laplacian with the corrected gate.

## Actual live E2E baseline

The latest workbench trace
`20260905T033031737181Z-f5a85105` is real capture-to-publication evidence, not
an offline replay. After initialization it contains 1,861 telemetry rows:

- observed update rate: 22.27 FPS;
- fresh XY: 93.82% of frames, or 20.91 fresh fixes/s;
- capture-to-control-publication median/P95/worst: 58.81 / 123.97 / 322.06 ms;
- within 33.3 ms: 5.70%; within 66.7 ms: 60.40%;
- engine update P95: 53.24 ms.

Therefore the current live system fails the 30 FPS E2E target even though the
offline localization candidates pass their serial decode-to-XY budget. The
remaining gap is outside the measured local matcher: live capture age,
scheduling, asynchronous worker interaction, fusion, evidence/recording,
rendering, IPC, and publication must be profiled end to end. The trace also
ended on the wrong map mode and 331.18 px from the demonstrated endpoint, so it
is not an accuracy pass.

## Traceability

- Transition-zone benchmark implementation: commits `76b36be`, `2b7a221`.
- Transition proxy builder and explicit evidence role: `1cc5662`.
- Serial recorded-video report rules: `c5818b0`.
- Finalist machine report:
  `artifacts/poc/live-e2e-finalists-report-20260905/localization_realtime_control_results.json`
- Finalist narrow report:
  `artifacts/poc/live-e2e-finalists-report-20260905/REPORT.txt`
- Layer matrix: `artifacts/poc/live-e2e-matrix-20260905/`
- Candidate screen: `artifacts/poc/live-e2e-candidate-screen-20260905/`
- Second transition: `artifacts/poc/live-e2e-transition2-20260905/`
- Transition tuning: `artifacts/poc/live-e2e-transition-tuning-20260905/`
- Clock causality: `artifacts/poc/live-e2e-clock-causality-20260905/`

All raw chronological telemetry is retained beside each `report.json`.
