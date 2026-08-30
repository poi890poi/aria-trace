# Route tracer POC — round 2

## Decision

Keep gradient correlation and the 18 px local window in production for now.
Land only the missing global-recovery-to-local-tracker reseed. Grayscale intensity
and a 12 px window remain candidates, but their latency advantage was not
consistent enough across fresh holdouts to justify changing production.

The route continues to supply bounded search proposals only. Every high-rate XY
measurement comes from the current mini-map matched against atlas pixels. Route
compliance is post-run diagnostics and never feeds estimation or acceptance.

## One-variable Session 11 results

All methods used the same 0.50 acceptance threshold and causal frame order.

| Change from gradient/all-layers/18 px baseline | Fresh measurements | Mean / p95 | Reference RMSE | Max adjacent step | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 98.9% | 3.79 / 6.96 ms | 1.84 px | 7.47 px | Keep |
| Grayscale intensity | 99.1% | 3.11 / 4.99 ms | 1.82 px | 5.48 px | Promising; cross-check |
| Canny edges | 56.6% | 11.18 / 25.76 ms | 1.86 px on sparse accepted frames | 3.88 px | Reject |
| Laplacian magnitude | 55.8% | 8.84 / 19.07 ms | 78.97 px | 5.48 px | Reject |
| 12 px local window | 98.9% | 3.02 / 4.91 ms | 1.84 px | 7.47 px | Promising; cross-check |
| Sticky current layer | 61.1% | 6.48 / 17.01 ms | 77.78 px | 11.63 px | Reject; trapped at representation changes |

## Independent forward-session cross-check

Runs 12 and 13 replayed in opposite directions with the other run supplying only
route proposals. Neither timing, heading, motion vectors, nor demonstrated pose
fed the tracker.

| Replay | Method | Fresh measurements | Mean / p95 | Reference RMSE |
| --- | --- | ---: | ---: | ---: |
| Run 13, Run 12 proposals | Gradient baseline | 100% | 8.35 / 28.43 ms* | 2.29 px |
| Run 13, Run 12 proposals | Intensity | 100% | 3.00 / 4.66 ms | 2.31 px |
| Run 13, Run 12 proposals | 12 px window | 100% | 3.10 / 5.48 ms | 2.29 px |
| Run 12, Run 13 proposals | Gradient baseline | 100% | 3.12 / 4.78 ms | 2.16 px |
| Run 12, Run 13 proposals | Intensity | 100% | 3.00 / 4.67 ms | 2.15 px |
| Run 12, Run 13 proposals | 12 px window | 100% | 2.72 / 4.07 ms | 2.16 px |

`*` The isolated baseline timing outlier did not reproduce; quality stayed
consistent. This is why latency was compared across multiple fresh runs rather
than inferred from one execution.

## New ordinary-cruise holdouts

Run 15 achieved 100% fresh visual measurements with baseline, intensity, and the
12 px window. Mean latency was 2.77, 2.91, and 3.07 ms respectively, so the
Session 11 speed ranking did not reproduce.

Run 14 exposed an initialization error in the original POC. A route-only first
proposal barely crossed threshold at 0.5007 in the wrong place and continuity
then correctly refused large corrections. Production instead starts from a
verified global fix. With that production-equivalent boundary:

| Run 14 mechanism | Fresh measurements | Longest hold | Mean / p95 | Result |
| --- | ---: | ---: | ---: | --- |
| Global initialization + local gate | 76.1% | 111 frames | 4.60 / 9.93 ms | Stale local anchor after rejection |
| Weaken gate using time since last accepted pose | 95.7–97.2% | 3–7 frames | 3.05–4.05 ms | Reject: Session 11 RMSE rose to 2.20–2.26 px |
| Existing two-fix global recovery + reseed local tracker | 97.0% | 13 frames | 8.16 / 4.24 ms** | Land reseed only |

`**` Mean includes synchronous POC recovery spikes of 414 ms. Production already
runs global localization asynchronously, so the landed change only reconnects
the accepted recovery pose to the local tracker.

The recovery-and-reseed mechanism preserved Session 11 accuracy: reference RMSE
was 1.82 px versus 1.84 px baseline, and max adjacent step was 8.11 px. It changes
no free-roam, calibration, cursor, rig, or artifact-schema behavior.

Runs 14 and 15 do not have independent sparse map-reference packages, so their
results support continuity and recovery conclusions only—not absolute accuracy.
Held-pose availability was never counted as a fresh visual measurement.

## Evidence locations

- Session 11 experiments: `artifacts/route-tracer-poc/round2/run11/` and
  `artifacts/route-tracer-poc/round2/run_11/`
- New cruise holdouts: `artifacts/route-tracer-poc/round2/run_14/` and
  `artifacts/route-tracer-poc/round2/run_15/`
- Opposite-direction cross-check:
  `artifacts/route-tracer-poc/round2/cross-forward/`

Each experiment directory contains `report.json` and frame-level
`telemetry.jsonl`. Generated evidence stays outside source control.
