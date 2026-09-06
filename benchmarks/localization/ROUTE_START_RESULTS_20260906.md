# Route startup integration and transition decisions

Promote explicit known-start template acquisition. Keep the transition reset and
ambiguous-scale hold experimental: their better reference-loss numbers conceal
larger correction jumps or longer gaps without fresh XY. Production steady
matching, transition behavior and physical frame capture are unchanged.

## Delivered behavior

The Route tracer has a visible Starting position selector. Its default is
"At the route beginning", matching the user's stated assumption. "Find my
position" retains global acquisition. Existing API requests that omit
`route_start_policy` still use `unknown-position`; free-roam requests cannot
select `demonstrated-start`. Invalid requests fail before opening capture.

The compiled route's first state supplies a bounded search proposal and initial
map layer, not a public pose. `TwoRateRealtimeTracker.set_route_start` stores it
without initializing fusion. Each current frame searches the existing 55px
initial window using the local refiner's existing 0.55 correlation requirement
and coverage checks. Only an accepted current-image position initializes fusion.
Steady matching then uses its existing radius and acceptance policy. The start
does not reset subsequent measurements, and global recovery remains available
after acquisition. Before acquisition, a failed match stays unavailable; the UI
reports that it is waiting for a matching view at the route beginning.

Player heading is not copied from the demonstration. XY freshness, heading
freshness and publication timing remain separate. This implementation is stricter
than the preceding eager-prior experiment, which inserted a pose before matching.

## Controlled recorded-source comparisons

These tests use the existing bounded 30-frame decode-ahead adapter, retain original
source timestamps, and release no image early to the tracker. Priming and buffering
are measured separately. This is a benchmark source change, not a physical-capture
optimization. Replays and tests did not run concurrently.

Run11 uses the actual Workbench route-assisted request and its compiled package.
Run14 isolates startup on a second recording: the benchmark explicitly supplies
that recording's own first compiled state, without full-route assistance. Neither
is a fresh repeat of an independently learned route.

| Full recording / method | First fresh XY s | Error P95 px | Longest reference-observed loss after acquisition s | Longest gap without fresh XY after acquisition s | Publication P95 ms |
|---|---:|---:|---:|---:|---:|
| Demo11, existing startup | 2.086 | 2.781 | 0.302 | 1.066 | 25.00 |
| Demo11, new verified start | 0.000 | 2.779 | 0.302 | 1.066 | 24.92 |
| Reverse14, existing startup | 2.510 | 2.800 | 0.231 | 0.154 | 21.24 |
| Reverse14, preceding eager prior | 0.000 | 2.787 | 0.231 | 0.154 | 32.34 |
| Reverse14, new verified start | 0.000 | 2.793 | 0.231 | 0.154 | 37.25 |

The reverse acquisition-only comparisons now reproduce the same 0.2314058s loss,
35.86px worst reference error and 19.69px maximum output step. The previous
apparent continuity regression did not reproduce with controlled delivery.
Demo control and verified start also preserve the same transition-loss interval
and 13.65px maximum output step. Small aggregate P95 differences reflect different
processed samples, including newly available startup positions.

Source-time acquisition of zero means the first frame, not zero wall time. First
XY publication is 44.21ms on the demo and 40.21ms on the reverse recording; neither
first frame has fresh heading. Demo verified-start output is 1737 fresh, 28 held,
0 unavailable out of 1765 processed / 1800 source frames. Reverse verified-start
output is 434 fresh, 16 held, 0 unavailable out of 450 processed / 465 source
frames. Source drops and asynchronous scheduling remain visible.

The +40px X prior control also acquires on its first frame, correcting to the same
image-measured demo position [1822.606, 1363.295], with 2.731px inferred-reference
error. It is a 9.98s startup clip, not another full route. In the +200px wrong-start
clip, the new path publishes no position: 0 fresh / 0 held / 298 unavailable.
The preceding eager-prior test had published 206 incorrect fresh positions in
its corresponding 300-frame clip. This negative control supports the initial
image gate, not a guarantee against every repeated-texture false match.

The 0.302s reference-observed forward loss understates the control interruption:
there is a 1.066s measured gap without fresh XY while the scale changes. Reference
evidence is insufficient for 3.904s of the demo and 1.001s of the reverse recording.
Unknown intervals are not proven accurate. Joint fresh XY and heading delivered
within 33.3ms reaches only 94.28% of demo source frames and 87.53% of reverse source
frames for the new path. This is not a sustained auto-cruise readiness pass.

## Transition diagnosis and ablations

In reverse, the new world layer is confirmed around 15.694s. The runtime clears
the old phase-correlation image and says the local reference was reset, but the
direct template tracker retains its old measurement timestamp and narrow search.
During zoom it has already accepted displaced old-layer matches. Correcting the
position is then rejected as a continuity jump until approximately 16.017s.

Reset-only clears the template timestamp after a confirmed layer switch. The next
image uses the existing wider first-refinement radius without comparing its step
against the old representation. It preserves the position as a proposal and any
trained transition anchor; the observer itself does not supply XY.

Hold-only suppresses XY updates while the existing controller is armed and the
normalized layer-score gap is below its existing minimum margin. It introduces
no new numeric threshold. The combination applies both interventions.

| Reverse14 candidate, verified start | Reference-observed loss s | Longest gap without fresh XY s | Worst error px | Maximum output step px | Fresh / held / unavailable |
|---|---:|---:|---:|---:|---:|
| Unchanged transition | 0.231 | 0.154 | 35.86 | 19.69 | 434 / 16 / 0 |
| Reset only | 0.000 | 0.128 | 3.80 | 32.02 | 442 / 9 / 0 |
| Hold only | 0.183 | 0.790 | 30.42 | 21.19 | 435 / 25 / 0 |
| Hold + reset | 0.000 | 0.790 | 3.80 | 26.49 | 440 / 19 / 0 |

Reset-only improves reacquisition but enlarges the correction jump. Holding
reduces some misleading measurements but lengthens the control blackout. Combining
them is not a smooth-tracking solution even though reference-observed loss is
zero. Reset-only and the combination were also replayed over full Demo11: both
retain 0.302s reference loss, 1.066s without fresh XY, 11.96px worst error and a
13.65px maximum step. Hold-only was not separately tested on the demo after its
reverse continuity tradeoff. Do not promote these transition candidates now.

Forward reference measurements explain a further limitation: inferred map scale
is 1.671 at 37.270s, 1.391 at 37.498s and 1.304 at 37.701s, while the fixed town
layer is approximately 1.303. The images undergo a transient scale change rather
than translation alone during this interval. A timestamp reset cannot supply
the missing scale model. Intermediate-scale current-image matching is the next
specific hypothesis; it has not been implemented or benchmarked here.

## Input revalidation

Final auditing detected that `calibration.json` had changed before this round's
first replay (mtime 02:36:31 UTC; first replay 02:59:16 UTC). The old reference
input SHA256 was `7f7ace6000892bdd76bb63aaad1799af9c668095cbf80d2811a34c027dd38540`;
the current file is `0385607d88f42c73e91b25b1082162553a71cb01b7751b1ae706dc3c73d460f4`.
The audit failed instead of silently accepting it.

Both evaluated references were rebuilt with the current inputs. All 265 demo
and 38 reverse retained states exactly reproduce the old scoring frame/time,
XY, layer, scale and alignment. The report builder independently checks this
equivalence, both old and new cache output hashes, and all 17 current source,
atlas and calibration input identities. Original runtime reports remain intact.
This proves unchanged scoring geometry on these recordings; it does not assert
that every calibration field is unchanged or make the proxies external truth.

New caches are `0642cd8df60e5fa1d8776355f87ba6a9be5c26d54e95b9cceec0973d2858e1e7`
(demo) and `9737b77b32a38ad845a2f73aa11147f7865470997145c5b4e9616cf67af16cb9`
(reverse). `reference_revalidation.json` preserves the comparison, and
`revalidated_references.json` selects them. Future replays validate frozen input
hashes before capture and refuse stale caches. The two benchmark CLIs accept an
explicit `--references` index so current caches can be selected without altering
the previous evidence.

## Decision and verification

Promote the explicit route-start path: high confidence in the tested startup
benefit, bounded by the stated known-start assumption. Keep transition mechanisms
unchanged in production; evidence does not support their smooth-control benefit.
Retain masked local gradient matching, and do not add XFeat or a separate global
template initializer. There is no claimed physical-capture speedup.

Twelve replays cover 379.49 recorded seconds. Seventy-five focused tests pass,
including unavailable-before-measurement, one-time start use, invalid inputs,
post-acquisition recovery, Workbench package handoff, legacy startup, cache input
drift, route tracking, loss scoring and decode-ahead ordering. Workbench JavaScript
parses successfully and existing UI structure tests pass. No fresh moving holdout,
native-game steering test or closed-loop auto-cruise validation was performed.

Evidence: `artifacts/poc/route-start-integration-20260906`, including generated
`comparison.json` / `COMPARISON.md`, raw Workbench telemetry, source scheduling,
executed experiment snapshots and rebuilt-reference provenance. Reproduce the
integrated path with `benchmarks.localization.prefetched_replay`, selecting
`--mode route-assisted --route-start-policy demonstrated-start --runs 11` and
the revalidated index through `--references`; use a new output directory.
See [ROUTE_START_PROTOCOL_20260906.md](ROUTE_START_PROTOCOL_20260906.md).
