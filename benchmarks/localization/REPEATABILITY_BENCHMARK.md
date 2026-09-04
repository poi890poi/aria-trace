# Route repeatability benchmark: method and current findings

## Purpose

Measure whether localization and pose pipelines remain continuously available,
repeatable, and responsive when a person traverses roughly the same route more
than once. The protocol requires no pause, lap marker, equal speed, equal
timing, or identical path. It is an offline benchmark and supplies no future
information to the tested online algorithm.

## Reusable data contracts

Repeated-waypoint inputs are timestamped states with canonical XY, map mode,
and heading. The heading semantics must be declared as `cursor_pose`,
`trajectory_tangent`, or `external_body_heading`. A continuous passage through
one state neighborhood counts once.

Turn-response inputs are timestamped JSONL signals containing
`session_time_ns` and a signed `value`. Optional validity and confidence remain
candidate provenance. Raw mouse delta is control intent, KLT is observed camera
motion, and cursor pose is candidate player heading; none may silently replace
another semantic.

## Method 1: automatic repeated complete-state waypoints

1. Build an independent post-run reference trajectory.
2. Find recurrent neighborhoods with the same map mode, nearby canonical XY,
   and nearby heading.
3. Merge adjacent samples into one visit and require temporally separated
   visits. Opposite directions at the same XY remain distinct states.
4. Sample every candidate at the discovered visit timestamps. The candidate
   never receives waypoint identity or reference position. Same-stream frame
   identity is preferred; otherwise nearest synchronized timestamps are used
   within an explicit tolerance, and alignment deltas are reported.
5. Report availability, fresh coverage, latency, absolute reference
   disagreement, and repeatability residual. The residual is dispersion of
   candidate-minus-reference error after removing per-group median bias, so a
   naturally wider or tighter human path is not counted as estimator error.

One reference observation may belong to only one subgroup. This prevents
overlapping neighborhoods from duplicating easy samples and inflating coverage.

Only an external ground-truth reference permits an absolute accuracy claim.
Offline atlas anchors and leave-one-family-out consensus provide comparative
evidence only.

The default neighborhood is 5 canonical pixels and 45 degrees, with distinct
waypoints spaced by 10 pixels or 35 degrees. Visits merge within 1.5 seconds,
must recur at least 8 seconds apart, and require at least three visits. These
parameters are stored in every result and must be sensitivity-tested before a
method is promoted.

## Method 2: sharp-turn temporal response

`benchmarks.build_turn_evidence` aggregates raw horizontal mouse input at the
requested sampling rate, measures scene rotation with forward/backward-checked
KLT, and optionally measures stateless cursor pose. UI exclusion rectangles are
explicit command inputs; the generic extractor has no game-specific default.

`benchmarks.temporal_turns` standardizes channel magnitude robustly, searches
both sign conventions and a bounded time-lag window, and then detects sharp
sign reversals from local medians. It reports:

- fitted sign, lag, and correlation;
- reversal response coverage;
- onset lag and settling time;
- wrong-direction persistence;
- normalized peak response and overshoot.

A pose direction defect requires corroborating input and scene evidence after
sign and lag fitting. If input and KLT disagree, the event is inconclusive.
Camera turn and player heading can legitimately differ, so correlation alone
is not an accuracy metric.

Each temporal filter or hysteresis policy should export its own causal 30 FPS
turn-rate signal and be passed as `--candidate NAME=SIGNAL.jsonl`. The current
stateless pose extraction is a baseline for evidence validation, not yet a
comparison of temporal policies.

## Reproduction

Build waypoint groups from a trajectory-tangent atlas reference:

```powershell
python -m benchmarks.localization.build_repeatability_report `
  --reference artifacts/poc/route-repeatability-reference-20260904/run18/route_states.jsonl `
  --reference-kind offline_atlas_anchor `
  --heading-semantics trajectory_tangent `
  --output artifacts/poc/route-repeatability-waypoints-20260905/run18
```

Build Genshin PC turn evidence with explicit UI masks:

```powershell
python -m benchmarks.build_turn_evidence `
  --session sessions/workbench/recordings-genshin-impact-pc/run_18 `
  --output artifacts/poc/route-repeatability-turn-evidence-20260905/run18 `
  --sample-hz 15 `
  --cursor-calibration artifacts/workbench/minimap_calibrations/genshin-impact-pc/segments-df624035-833-bd07601f-708/calibration.json `
  --excluded-rect 0,0,0.24,0.30 `
  --excluded-rect 0.72,0,1,0.28 `
  --excluded-rect 0,0.76,1,1
```

## Findings from Runs 17 and 18

Run 17 is the outdoor loop. Its current offline-atlas reference accepted 397
samples and produced only 3 disjoint repeated-state groups with 9 visits. This is too
sparse for a broad localization ranking. The temporal channels are nevertheless
useful: input-to-KLT correlation is 0.805 with opposite sign and 66.7 ms fitted
lag; all 19 sharp input reversals received a scene response. Stateless pose has
the consistent sign composition but weaker correlation (0.316 to input and
0.430 to scene), responds to 18 of 19 input reversals, and has 404.3 ms onset
lag P95.

Run 18 is the town loop. Its reference accepted 823 samples and produced 35
groups with 163 visits, normally four or five visits per group. This is the
stronger current repeatability dataset. Input-to-KLT correlation is 0.748 with
opposite sign and 66.7 ms fitted lag; 13 of 15 sharp input reversals received a
scene response. Stateless pose again has consistent sign composition but weak
correlation (0.356 to input and 0.272 to scene), responds to 14 of 15 input
reversals, and has 778.7 ms onset lag P95.

The two sessions do not support the hypothesis that cursor-pose direction is
globally inverted. They do support further testing of pose availability and
temporal policies: pose response is consistently weaker and slower than the
input-to-scene relationship, especially in town. This is a testable hypothesis,
not yet a causal attribution to hysteresis, because the reported pose baseline
is stateless and camera direction is not player-heading truth.

## Required next comparison

Replay every promising pose and localization temporal policy causally at 30
FPS, preserve rejected and held outputs, and feed each standardized signal to
the same evaluator. Compare fresh coverage, final availability, error mean,
median, P95 and worst, plus sharp-turn response and wrong-direction duration.
Use Run 18 for development and keep Run 17 or a later well-anchored outdoor loop
as a holdout; do not tune on both and call the result independent.
