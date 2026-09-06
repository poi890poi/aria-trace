# Narrow-lap tracking, map transitions, and human aiming

Status: **benchmark only**. No experimental tracking policy or atlas replaces
production. The recording milestone and session-transfer feature are separately
committed as `89ae9fa` / `session-recording-known-good-20260906` and `464962f`.

## Main result

Run 20 exposes a spatial transition-gate failure that short held-output gaps
conceal. The production tracker switches town → world once, then misses all
three world → town entries. It therefore also produces no new events at the
second and third exits. Removing the learned spatial gate, while retaining
visual margins and repeated confirmation, detects all six transitions.

| Run 20 variant | First fresh XY s | P95 / worst error px | Longest lost s | Longest nonfresh XY s | Largest XY step px | Fresh / held / unavailable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Production, unknown start | 2.923 | 142.756 / 173.273 | 30.215 | 0.424 | 34.967 | 4935 / 468 / 89 |
| Production, image-verified known start | 0 | 143.522 / 172.087 | 31.988 | 0.471 | 35.601 | 4892 / 516 / 0 |
| Known start + remove spatial transition gates | 0 | 2.903 / 34.638 | 1.229 | 0.230 | 24.534 | 5598 / 74 / 0 |
| Above + confirmed-switch search reset | 0 | 2.872 / 34.638 | 0.946 | 0.230 | 34.330 | 5786 / 66 / 0 |

Loss includes incorrect fresh XY, wrong map layer, and missing output, with a
0.5-second recovery confirmation. It is evaluated against an inferred atlas
reference, with unknown intervals retained. These are not validated gameplay
pixel tolerances. The verified-start control publishes **1,158 fresh positions
on the wrong layer** and 1,126 fresh positions over 20 px from the proxy. Both
visual-transition variants reduce these counts to 15 and 7 respectively.

The reset combination reduces the longest sampled loss by 0.283 seconds but
raises the largest correction from 24.534 to 34.330 px. It does not reduce the
worst error. Retain this tradeoff; do not call the combination an unconditional
quality improvement.

The shared worst error occurs at 19.184 s, while a town-layer match is still
published as fresh; the switch is confirmed at 19.254 s. A reset applied after
confirmation cannot remove that earlier bad measurement. The next transition
benchmark should therefore examine current-image scale estimation during the
animation, including its interaction with XY acceptance and subpixel template
refinement, rather than assume that a post-switch reset solves the transition.

| Run 20 variant | Publication P95 ms | Jointly fresh XY + cursor heading within 33.3 ms / all source frames |
| --- | ---: | ---: |
| Unknown start | 33.325 | 79.58% |
| Verified start | 33.788 | 78.13% |
| Visual transitions | 29.767 | 91.01% |
| Visual transitions + reset | 24.040 | 95.68% |

Recorded-source CPU replays ran sequentially with the same bounded decode-ahead
adapter. Ordinary repository/analysis work and OS scheduling were not isolated.
Frame processing counts differ, and these single-run timing differences are
**not a proven speedup**. Raw source, decoder, processing, and publication times
are retained. The measurement excludes physical GDI acquisition and browser or
display latency, and never sends game controls.

## Causal evidence

The atlas already owns its transition model. Its only learned zone has radius
**1.655 canonical px**, estimated from just two boundary samples in run 9.
The runtime permits a switch only after entering that zone; its wider 40 px
corridor permits observation but does not independently permit arming.

At approximately 63–64 seconds in the baseline, town image correlation reaches
0.773 versus world 0.515. The controller remains `outside_transition_zone`,
with zero confirmations, because the current trajectory never armed the tiny
zone. Wrong-layer XY subsequently drifts beyond the observation corridor, which
stops new mode observations. Local template matches can still be accepted and
the global recovery path is not triggered merely by being on the wrong layer.
This explains the long incorrect-fresh episodes and recovery when the player
physically exits the town again.

The single-variable ablation removes `TransitionController.transition_zones` in
a benchmark context only. Score margins, confirmation count, local XY rules,
and search radii are unchanged. The combination additionally uses the existing
confirmed-switch timestamp reset experiment. No reference modes, future XY, or
route-progress labels enter either transition detector.

| Direction | Last old / first new proxy sample s | Visual detector event s | Offset from first new proxy sample s |
| --- | ---: | ---: | ---: |
| Town → world | 18.502 / 19.145 | 19.254 | +0.109 |
| World → town | 62.322 / 62.968 | 63.213 | +0.245 |
| Town → world | 90.338 / 91.027 | 91.268 | +0.241 |
| World → town | 136.276 / 136.926 | 137.348 | +0.422 |
| Town → world | 164.467 / 165.174 | 165.331 | +0.157 |
| World → town | 212.254 / 212.963 | 213.246 | +0.283 |

Both visual-transition variants produce identical event times. These offsets
are not exact animation-onset latency: the reference itself has approximately
0.64–0.71 s gaps and intermediate zoom at these brackets. Two fixed rendering
scales still cannot fully explain the animated zoom, so transition-local errors
and discontinuities remain after the gate is removed.

Confidence: **high** that the hard spatial gate causes these repeated misses;
**medium** that visual confirmation without this gate generalizes beyond these
recordings; **low** that the reset combination improves overall control quality.
Run 20 was a fresh holdout for production and known-start tracking. It became
development evidence once the new gate-removal hypothesis was formed.

Matched free-roam controls on the earlier recordings use the same current
reference index and verified-start policy:

| Session | Spatial gate | P95 / worst error px | Longest lost s | Largest step px |
| --- | --- | ---: | ---: | ---: |
| 11 | Production | 2.783 / 24.175 | 0.402 | 19.308 |
| 11 | Removed | 2.789 / 24.175 | 0.402 | 19.308 |
| 14 | Production | 2.793 / 35.856 | 1.346 | 19.307 |
| 14 | Removed | 2.790 / 35.856 | 1.266 | 19.690 |

These controls show no material position-error gain on the original crossings,
and preserve the existing transition errors. Small timing/loss differences on
run 14 are inconclusive under asynchronous frame delivery. Do not compare these
loss durations directly with an older report's different reference-derived
tolerance, or compare this free-roam run 11 against an earlier route-assisted
run as if only the gate changed.

## Atlas-owned town and world evidence

Two separate candidate atlases live under the evidence root:

- `map-feature-atlas`: observed mode support learned from hash-validated runs
  11 and 14; run 20 used only for evaluation.
- `map-feature-atlas-expanded`: adds run 20 after that evaluation. This expanded
  artifact has no new independent holdout and is not used by the replay tracker.

They preserve existing artwork and localization layers, adding canonical
town/world support masks, unknown/overlap labels, source sample identities, and
directed crossing brackets in `map_features/observed_modes.json`. These are map
features, with no route order or lap-progress authority. They are not complete
town segmentation and do not label the entire mini-map footprint as town.
The expanded candidate records 1,350 stable observed samples and eight directed
crossing brackets, with 5,332 town-support and 8,632 world-support canonical
pixels. Support is a record of evidence, not a surveyed boundary.

On stable run 20 reference samples, the earlier-session support map labels 347
correctly and 11 incorrectly, leaves 650 unknown, and marks 40 ambiguous. Its
town support covers only 36 of 443 town samples unambiguously correctly. This
negative result matters: sparse paths cannot serve as an exhaustive permission
map for transitions. Unknown area must not veto strong current-image evidence.
Entry and exit observations also occur at different positions; retain direction
and uncertainty rather than forcing them into one exact boundary point.

Recommendation for later benchmarking: store an accumulating map-owned region
and crossing model as a soft prior, separate from current-image representation
verification. Learn spatial extent from diverse traversals; test withheld lanes
and directions. Do not promote the current sparse masks into a hard classifier.

## Precision and human aiming

Native frame review and the slow-reference path show separated bridge lanes and
small endpoint loops. Outbound examples at 25, 98, and 176 seconds track along
the player's left side of the bridge. Return examples at 45, 125, and 202
seconds preserve the other lane. The path is not a repeated centerline; forcing
nearest-route correspondence would erase the intended offset and circular turns.

The recorded input supports an aim-and-maintain strategy. During 45–50 seconds,
horizontal mouse travel totals only 33 raw counts; during 50–55 seconds there
are no mouse-motion events while W continues. The 35–40 second endpoint turn
uses 2,806 horizontal counts. Gate openings, lamp posts, and walkway edges remain
visible ahead on the straights. They are plausible aiming references, not proven
gaze targets. There is no eye tracking in this recording.

The current real-time path reports scene yaw as
`bypassed:route-map-correlation`; it does not currently imitate scene-feature
aiming. Cursor heading is a separate image measurement. During the no-mouse
50–55 second interval, the verified-start replay has a 2.33° cursor-heading
span, 0.55° standard deviation, and 0.80° P95 successive step across 146 fresh
samples. This is an input-idle stability diagnostic, not absolute yaw error.

The world atlas is approximately 3.86 canonical px per mini-map pixel, versus
1.30 for town. Integer template maxima visibly produce stair-step positions;
even the inferred reference cannot substantiate arbitrarily fine control
precision. Smoothness, absolute XY, camera yaw, avatar heading, and movement
direction must remain separate evaluations.

Proposed benchmark methods, in priority order:

1. **Distant-feature bearing control.** In each directed route phase, retain a
   few persistent scene features and their demonstrated image bearings. Track
   them with masked sparse optical flow, use current-image template matches to
   reacquire, and correct horizontal bearing error with bounded mouse input.
   Keep a learned offset instead of always centering a landmark: the desired
   lane can lie to its left or right. Use the atlas for coarse position and
   progress, not as the sole source of fine yaw. Sparse Lucas–Kanade tracking is
   directly supported by [OpenCV](https://docs.opencv.org/5.0/main_modules/video_track.html);
   regulating image features follows [image-based visual servoing](https://www-sop.inria.fr/members/Patrick.Rives/Publications/ITRA-espiau-chaumette-rives-92.pdf).
2. **Lookahead plus local corrections.** Learn route direction, offset lane,
   and local curvature, then select a farther image target on straights and a
   nearer sequence around circles. Blend changes in targets to avoid snapping
   the commanded yaw. Compare this with nearest-waypoint steering on exactly
   the same demonstration/held-out traversals.
3. **Feature handoff and loss handling.** Reacquire before a target leaves the
   view or becomes occluded. Treat target loss, XY loss, and heading loss as
   distinct events. Benchmark intentional wrong targets, similar gate features,
   endpoint loops, and scale changes; a smooth but wrong target is a failure.

Confidence is **medium** in the match between this design and the observed
human input pattern, **low** in closed-loop performance until tested. Begin with
replay feature persistence, bearing residuals, latency, and withheld-lap matching.
Only subsequent authorized live control can establish lane keeping and arrival
precision. No control behavior was implemented or run in this round.

## Inputs, reuse, and verification

Run 20: session `ae32d060-6b92-4311-a59e-69427d3c667d`, 240.193 s manifest,
5,931 frames, 12,422 inputs, zero manifest-reported drops. Recorded frame spacing
averages 40.456 ms (about 24.72 Hz), P95 76.057 ms, maximum 290.543 ms, with 118
gaps over 100 ms. Zero reported drops therefore does not mean a sustained 30 Hz
source. Benchmark source span is 239.903 s.

The 5 Hz slow-reference cache contains 1,056 accepted states of 1,068 attempted;
12 rejected samples are scale-out-of-range near the transitions. Cache key:
`5fc3d68210a6e9b0d88b4aa0c9d76b6b1d96c617a967c88a369363a4b6517b81`.
It retains timestamps, positions, scale/alignment, rejected samples, input/source
hashes, and output checksums for reuse. It is an SIFT-derived same-atlas proxy,
not external ground truth. Only its first state is supplied to the known-start
variants as a declared proposal, verified by the first current image.

Evidence root: `artifacts/poc/narrow-lap-20260906`. `comparison.json` and
`lap-analysis.json` contain exact aggregates, reference gaps, transition events,
and input diagnostics. Each replay retains raw/scored/source telemetry and
implementation identities. `visual-review` contains 20 native scene frames,
five contact sheets, and `inferred-path.png`. Each candidate atlas includes an
`observed_mode_overlay.png` for visual review.

Verification: 75 transfer/Workbench/UI tests passed; the real seven-file session
round trip preserved all SHA-256 hashes. Thirteen transition-ablation, replay,
and tracking-loss checks passed. Production tracking and active atlas files
remain unchanged; experimental context managers restore their patched methods.
All 54 production tracking/mapping/calibration source identities common to the
four run 20 reports match. Eight completed replays cover approximately 18.7
minutes of recorded-source time. The original atlas input hashes remain valid.
