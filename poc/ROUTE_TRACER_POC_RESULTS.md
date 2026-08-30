# Causal route-tracer POC

## Decision summary

The simplest viable mechanism is `continuous_gated` with a `0.50` map-correlation
threshold:

1. Use the compiled route descriptor only to initialize or recover a bounded map
   search.
2. Return the current frame's map-correlation pose, never the demonstrated pose.
3. On ordinary frames, search only around the last accepted visual pose.
4. Reject an implausible one-frame position jump and temporarily hold the last pose.
5. Calculate route compliance after replay; it never feeds localization.

This is a POC recommendation, not a production-live-tracker change. No workbench
default or artifact schema was changed.

## Causal constraints

Every replay processed frames in recorded order and had no access to future frames,
demonstration timestamps, demonstration motion vectors, demonstration headings, or
demonstration progress. The input may pause, vary speed, deviate from the route, or
move in a different direction. After initialization, the route does not constrain the
accepted pose; it only remains available for visual recovery.

The 30 fps gate applies to algorithm wall time. Decode time is measured separately.
Session 11 is a real 30 fps recording. Session 6 was captured at a lower effective
rate, but the same per-frame algorithm budget was used.

## One-change experiment ladder

| Step | Only change from prior step | Session 11 result | Decision |
|---|---|---|---|
| Descriptor baseline | Stored route descriptor supplies pose | 0.95 ms mean, 59.3% lock, 68-frame worst loss | Reject: fast but discontinuous; compliance is circular |
| Refine top 1 | Current frame/map correlation supplies pose | 5.02 ms mean, 100% lock, 2.82 px reference RMSE, 56.8 px max jump | Keep as evidence; proposal every frame is unnecessary |
| Refine top 3 | Proposal count 1 -> 3 | 14.20 ms mean, identical accuracy and continuity | Reject: 2.8x cost, no benefit |
| Continuous local | Route proposal only for initialization/recovery | 2.88 ms mean, 4.35 ms p95, 100% lock, 2.79 px reference RMSE | Keep; simplest fast causal mechanism |
| Continuity gate | Reject implausible one-frame jumps | 2.80 ms mean, 4.07 ms p95, 100% pose availability, 1.84 px reference RMSE, 7.47 px max jump | Keep; removes transition outlier |
| Threshold 0.50 | Correlation threshold 0.55 -> 0.50 | No regression versus 0.55 | Preferred after cross-route result |

`route_descriptor` reports stored route coordinates, so its 0 px route-compliance
error is not an accuracy result and must not be compared with refined methods.

## Cross-route check

Session 6 is a different cruise, not a held-out repeat of Session 11. Its route starts
at canonical `(514, 808)` rather than `(743, 836)` and only 46.6% of the Session 11
corridor overlaps within 35 px. It was therefore compiled as its own sparse 5 Hz
route and replayed with the identical tracker code and parameters.

Only frames 0-635 have usable offline map evidence. The remaining frames reproduce
the known incomplete/fog-of-war map limitation, so full-session failures are retained
in raw reports but not presented as a route-tracer regression.

| Method | Map-supported visual lock | Mean / p95 algorithm time | Reference RMSE | Compliance RMSE | Max adjacent step |
|---|---:|---:|---:|---:|---:|
| Descriptor baseline | 32.9% | 0.65 / 0.92 ms | 1.82 px on accepted frames | Circular 0 px | 10.97 px |
| Top-1 map refinement, threshold 0.55 | 90.9% | 6.08 / 10.64 ms | 2.23 px | 1.50 px | 5.48 px |
| Continuous + gate, threshold 0.50 | **100.0%** | **2.95 / 4.64 ms** | **2.23 px** | **1.58 px** | **5.48 px** |

The preferred method is comfortably inside the 33.33 ms frame budget on both
routes. Session 11 used route recovery once, local correlation on 1,779 frames, and a
20-frame continuity hold across the map-representation transition. It preserved a
pose for every frame, but the hold must remain visible as reduced measurement
confidence rather than be called a visual lock.

## Raw evidence

- Session 11 reports: `artifacts/route-tracer-poc/run11/`
- Session 6 reports: `artifacts/route-tracer-poc/run06-own/`
- Session 6 sparse route reference: `artifacts/route-tracer-poc/references/run06/`
- Independent forward replays: `artifacts/route-tracer-poc/cross-forward/`
- Per-frame records: `telemetry.jsonl` inside each variant directory

## Landing recommendation

Land only the causal mechanism, not all explored variants:

- route descriptors initialize or recover a bounded search;
- local map correlation owns every normal pose;
- the route never supplies heading, motion, timing, progress, or compliance feedback;
- use the 0.50 correlation threshold with a broad continuity jump gate;
- expose pose availability separately from fresh visual-measurement acceptance;
- keep top-3 recovery and descriptor-supplied poses out of production.

## Independent opposite-direction replay

The newer, fully mapped forward sessions close the independent-replay evidence gap.
Sessions 12 and 13 cover the same general corridor but differ in all of the ways the
tracker must tolerate:

- Session 13 traverses the corridor in the opposite direction;
- its path is laterally offset by about 26 px;
- timing, motion vector, and player heading therefore differ;
- the tracker receives no route progress or time alignment.

| Demonstration -> replay | Fresh visual lock | Mean / p95 algorithm time | Reference RMSE | Compliance RMSE | Recovery use |
|---|---:|---:|---:|---:|---:|
| Session 12 -> Session 13 | **100.0%** | **2.71 / 3.69 ms** | **2.29 px** | 25.65 px | 1 initial + 600 local |
| Session 13 -> Session 12 | **100.0%** | **2.78 / 3.92 ms** | **2.16 px** | 26.62 px | 1 initial + 540 local |

The 25-27 px compliance error is expected and useful: it measures the real lateral
difference between the independently localized replay and the demonstration. If the
tracker had snapped to route states, this value would be artificially near zero.

This makes `continuous_gated` at threshold `0.50` suitable to land as the live route
tracer mechanism, subject to normal production integration tests. The isolated POC
still does not change live defaults by itself.
