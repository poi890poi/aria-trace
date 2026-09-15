# Local course relation: evidence before product trial

Run21 first three traversals, before 124 s, prepare the route. Later Run21
frames are excluded from preparation but have been inspected previously.
Current route context comes from causal ALIKED/VLAD retrieval. Runtime map
measurements use current and past pixels. No future query image is available
to the controller; future pixels appear only in the separate prediction scorer.

| Comparison | Result | Decision |
|---|---|---|
| Known native-map image translations | P95 0.038 px, maximum 0.053 px across 160 controls | Supports local measurement, not absolute world accuracy |
| One-second teacher time target | 515/559 hints; can suggest reversing when a target has been passed | Replace time-based target selection |
| Target advances along prepared local path | 516/559 hints; removes the roughly 164° reverse suggestion | Retain advancing target |
| Paths from up to three teaching laps | 541/559 hints; window width median 5.12°, P95 17.18° | Retain a window that permits demonstrated variation |
| Separate map regions | All 541 accepted relations pass; disagreement P95 0.347 px, max 0.505 px | Retain as a support check, not a confidence interval |
| Raw mouse / course calibration, Run21 | Gain 0.064399°/count | Compatible with Run20's 0.062730°/count |
| Recent input compensates averaged course | Approximate reference P95 17.96° → 10.21° | Retain for opt-in trial; reference is map-derived and retrospective |

Input calibration's fitted 0.4 s lag includes the 0.5 s motion window and
0.3 s course derivative. It is not evidence of 0.4 s actuator delay. A route
window is not a collision-free-space estimate. The design has not executed
automatic navigation or established independent position/clearance accuracy.

Evidence root: `artifacts/poc/local-route-control-20260915`. Raw per-sample
files include `rows.json`, `path-rows.json`, `corridor-rows.json`, `support.json`,
`motor-fit.json`, `run20-motor-fit.json` and `current-course-prediction-score.json`.
Source cache and timestamps are recorded in `cache.json`. Native masks use
the existing `segments-df624035-833-bd07601f-708` calibration.
