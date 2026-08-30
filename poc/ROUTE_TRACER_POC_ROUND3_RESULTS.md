# Route tracer POC — round 3

## Decision

Keep gradient correlation. Reject grayscale intensity for unseen-area tracking.
The 12 px local search window is now the preferred production candidate, but this
round does not change production.

The new Run 16 recording is a fresh ordinary-cruise holdout in an area absent
from the calibration recordings. Candidate parameters were declared before this
replay and were not tuned on Run 16. Recorded controls were not used for pose,
acceptance, initialization, or scoring.

## Run 16 holdout

All variants received the same independent global initialization:
`(721.37, 717.27)` canonical pixels, score `0.6293`, margin `0.1899`.
The demonstrated route is far away (436 px cross-track RMSE and 0% coverage for
the gradient variants), which is useful negative evidence that the route proposal
did not pull the tracker onto the demonstration.

| One-variable variant | Fresh visual measurements | Holds / loss episodes | Mean / p95 | Max adjacent step | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Gradient, 18 px baseline | 100% | 0 / 0 | 2.93 / 4.55 ms | 5.48 px | Control |
| Grayscale intensity, 18 px | 85.5% | 256 / 10 | 25.63 / 207.88 ms | 5.48 px | Reject |
| Gradient, 12 px | 100% | 0 / 0 | 2.54 / 3.74 ms | 5.48 px | Preferred candidate |

The intensity median remained low, but rejected measurements triggered long
hold/recovery work. Its tail latency and fresh-measurement loss make it unsuitable
despite its performance on earlier routes.

The 12 px window produced the same accepted path and continuity as the 18 px
baseline while reducing mean latency by 13% and p95 latency by 18%. Both are far
inside the 33.33 ms algorithm budget for 30 fps.

## Aggregate evidence

The 12 px window preserved the reported accuracy or path statistics on Session 11,
both opposite-direction forward replays, Run 15, and Run 16. It was faster in four
of those five comparisons; Run 15 was the exception (3.07 ms versus 2.77 ms mean).
The new holdout resolves the earlier uncertainty in favor of 12 px without changing
the correlation representation.

Run 16 has no independent reference-pose package, so its results establish
continuity, non-snapping, and runtime—not absolute localization accuracy. The
earlier reference-backed replays remain the accuracy evidence.

## Input evidence

Run 16 contains synchronized keyboard and relative-mouse events, but they were
deliberately excluded from this experiment. The visual baseline already achieved
100% fresh measurements, so input fusion cannot improve this holdout's XY lock and
would add a second variable. Inputs remain useful later as soft control-intent
evidence for segmentation, latency measurement, yaw proposals, and recovery
scheduling; they must not become pose truth.

## Benchmark isolation correction

During this round, the POC harness revealed that experimental feature modes replaced
the atlas gradient map before independent global initialization. That compared a
gradient observation with a different map representation. Commit `ab7d990` defers
the experimental representation until local replay tracking starts. The correction
is POC-only and is covered by a regression test. The corrected intensity run above
therefore shares the exact baseline global fix.

## Evidence locations

- Baseline: `artifacts/route-tracer-poc/round3/run_16/baseline/`
- Corrected intensity: `artifacts/route-tracer-poc/round3/run_16/intensity/`
- 12 px window: `artifacts/route-tracer-poc/round3/run_16/radius12/`
- Previous development and reference-backed evidence:
  `artifacts/route-tracer-poc/round2/`

Each variant contains `report.json` and frame-level `telemetry.jsonl`. Generated
evidence remains outside source control.
