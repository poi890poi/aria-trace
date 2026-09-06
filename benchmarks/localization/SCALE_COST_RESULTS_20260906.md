# Scale-search implementation recommendations

This round tests changes for the next live-tracker implementation. It does not
enable a new live default or rebuild the active atlas. The full 17-scale/subpixel
algorithm from `cdb671a` is the accuracy baseline; the question is whether its
cost can be reduced while preserving the measurements.

## Decisions

- **Recommend implementing the optimized combined tracker for the next normal
  live-tracking test.** Integrate bounded parallel matching into the existing
  17-scale/subpixel localizer, with six workers on this measured host. Retain
  every scale, visual confirmation, coverage/score checks and iteration order
  for tied scores, and shut workers down explicitly. This is supported by exact
  native outputs and substantial recorded-source delivery improvements.
- The separate late mode-consumption hook is not recommended from these results.
  Its isolated test proves that it can publish a completed observation sooner,
  but all six full-lap loss intervals start and recover at exactly the same
  source timestamps with and without it. It does not solve the remaining mode lag.
- **Recommend landing precise recorded-source deadline waiting independently.**
  It removes measured artificial delivery lateness. This is benchmark fidelity,
  not an estimator or physical-capture optimization. Preserve original timestamps
  and no-early-release semantics.

## Native matching screen

78 native Run 20 observations were fixed before the screen: 48 distributed
positions plus samples around six previously identified transition peaks.
Images, masks, proposal XY, radius and active mode were identical across methods.
Proposals came from the prior visual-only control, not evaluation references.
Three repetitions rotated method order. All 936 measured outputs matched the
serial implementation exactly, including all scale scores and fractional XY.

| Workers | Native wall P95 ms | Native wall mean ms | Process CPU mean ms |
|---|---:|---:|---:|
| 1 | 20.708 | 18.793 | 17.228 |
| 2 | 12.053 | 10.786 | 18.296 |
| 4 | 9.892 | 7.659 | 17.495 |
| 6 | 8.725 | 7.381 | 19.965 |

Six workers gave the lowest P95 on this host, with more CPU than four. This is
not evidence that six is universally optimal. Process CPU readings have coarse
clock resolution; per-call zero values and percentiles are not meaningful CPU
absence claims. Native screen excludes decode, other tracker workers and publication.
No hypothesis pruning, new acceptance threshold, kernel replacement or extra
atlas pyramid allocation is needed for this speedup.

## Original-source replay checks

The independent 25-second first-crossing replays gave publication P95 87.427 ms
serially, 43.719 ms with four workers and 39.930 ms with six. Each retained the
same 3.352 px worst inferred-reference disagreement and zero held outputs.
The late-consumption hook alone shortened observed loss from 0.145 to 0.109 s;
this small scheduling effect did not generalize to the full-lap longest loss.

The full six-worker lap gave 2.189/3.352 px P95/worst disagreement, no held output,
largest XY step 2.836 px, longest lost 0.422 s, and publication P95 33.732 ms.
The preceding serial service baseline was 2.191/3.352 px, longest lost 0.493 s,
and publication P95 72.994 ms. That older full baseline is historical; the fresh
25-second serial/concurrent pair above is the controlled same-round timing check.

Adding late mode consumption to six workers gave 35.033 ms publication P95 and
the same 0.422 s longest loss. All six loss episodes were wrong-map-mode intervals
with identical start/recovery timestamps. Retain that negative result rather
than adding a mechanism based only on a unit test or a one-frame clip improvement.

## Replay-source wait diagnosis

The six-worker full lap's engine P95 was 14.870 ms; source release lateness P95
was 20.012 ms. An isolated alternating-order timer screen measured 100 event waits
and 100 short-sleep deadline waits. Lateness P95 was 14.689 versus 0.709 ms.
Neither method was credited with changed source capture timestamps. Source-side
improvements are not evidence that physical GDI/camera capture is now faster.

The full six-worker replay with precise waits gives source lateness P95 1.201 ms
and publication P95 19.361 ms. It processes 5925 of 5931 source frames, all fresh
XY, versus 5874 with the original waits. Joint fresh XY and heading within 33.3 ms
increases from 93.71% to 98.62% of all source frames. Zero source frames are
released early in any evaluated run. Score timestamps are unchanged; no delay
is subtracted afterward.

| Full replay | Reference P95 / worst px | Longest lost s | Fresh / held / unavailable | Publish P95 ms | Joint <=33.3 ms / all source |
|---|---:|---:|---:|---:|---:|
| Six workers, lap 20, original waits | 2.189 / 3.352 | 0.422 | 5874 / 0 / 0 | 33.732 | 93.71% |
| Six workers, lap 20, precise waits | 2.189 / 3.352 | 0.422 | 5925 / 0 / 0 | 19.361 | 98.62% |
| Six workers, reverse 14, original waits | 1.854 / 2.982 | 0.000 | 459 / 0 / 0 | 39.780 | 87.10% |
| Six workers, route-assisted demo 11, original waits | 2.165 / 3.139 | 0.302 | 1739 / 0 / 0 | 38.444 | 87.61% |

All three full lap replays detect all six transitions and have no jumps over
8 px. The 0.422 s longest loss remains a wrong-mode interval despite accurate XY;
the precise-source lap still has 31 fresh outputs disagreeing with reference mode.
This round does not solve coherent XY/scale publication. The demo's worst error
is 3.139 px versus the prior serial 3.099 px, with different processed-frame sets;
do not claim exact E2E trajectories from exact same-input kernel parity.

### Repeat and remaining delivery tail

Do not treat the full-lap 19.361 ms P95 as a universal deadline pass. Its first
25 seconds have 40.255 ms P95 and 91.43% joint on-time delivery. Repeating that
25-second clip gives 34.535 ms P95 and 94.51% joint delivery. Both keep the same
3.352 px worst disagreement with no held output. The same precise-source serial
clip gives 63.521 ms P95 and 59.44% joint delivery. Thus parallel matching's
benefit repeats, while strict 33.3 ms P95 does not hold consistently in this window.

With the original waits, the six-worker first-25-second P95 is 39.930 ms in the
isolated clip and 38.464 ms in the full run. Better source waits reduce release
lateness and improve joint delivery, but do not monotonically reduce every short-
window publication percentile. Host scheduling and engine tails remain in the
measurements; this is not a license to omit the early interval or subtract delay.

The early-window decomposition localizes part of that variation beyond template
matching: template P95 is 15.972 ms with old waits, 17.632 ms with precise waits
in the full run, and 16.264 ms in the repeat. Engine elapsed time beyond the local
template has P95 1.664, 16.997 and 1.699 ms respectively. The extra early engine
tail does not reproduce; its internal cause is not isolated here. Keep the
measured joint delivery and full latency rather than promising a fixed deadline.

## Confidence and scope

Parallel matching: **high confidence** in preserving matching behavior and reducing
measured CPU wall time; **moderate confidence** in live delivery/generalization.
It is ready for service integration and normal live-tracking validation, not a
claim of completed autonomous cruising. The optimized combined tracker retains
the known scale-aware accuracy benefits, so a separate subpixel-only promotion
is not recommended.

Precise source waits: **high confidence** in the diagnosed benchmark lateness
and preservation of deadlines, ordering and image identity. Cancellation is
checked between sleep requests of at most 5 ms; actual wakeup latency still
depends on host scheduling. Do not describe this as a physical-capture speedup.

Late mode consumption: retain the negative experiment; **insufficient evidence
to land it** as a tracking-quality improvement. No normalization-kernel rewrite,
scale pruning, new atlas prior or threshold tuning was needed or tested here.

Verification: 119 focused tests pass, including worker completion/tie ordering,
worker failure, mode confirmation and unchanged XY, source deadlines,
cancellation, image/timestamp identity, and reordered-input rejection. This round
contains 11 E2E replays (five complete sessions and six 25-second clips), 936
measured native parity comparisons and 200 isolated deadline waits. No fresh
post-hypothesis recording is available. References are reused same-atlas inferred
proxies, not independent position/heading truth; Run 20 retains 7.440 s unknown.
No new active-atlas data, route compilation, live default or game controls were
changed. The recommendations above have not yet been enabled in normal Workbench
tracking; the next implementation should make that status explicit.

## Reproduction and evidence

Protocol: `SCALE_COST_PROTOCOL_20260906.md`. Evidence root:
`artifacts/poc/scale-cost-20260906`. It contains native screen inputs and exact
comparisons, timer measurements, raw/source/scored E2E telemetry, executed source
snapshots and configuration, and aggregate reports. Each E2E report identifies
its actual Git state and source hashes. Cached slow inferred reference outputs
are reused and input/output hashes revalidated, never injected as later runtime XY.

```powershell
.\.tools\standalone-release-py31210\Scripts\python.exe -m benchmarks.localization.scale_cost --scale-workers 6 --tracker-implementation scale-aware --representation existing --action replay --runs 20 --output artifacts/poc/NEW-OUTPUT --references artifacts/poc/narrow-lap-20260906/references.json --start-policy verified --start-route 20 --prefetch
```

Use a new output directory. `--consume-mode-after-xy` isolates the mode-delivery
hook; `--precise-release` isolates the recorded-source wait change. Neither flag
changes normal Workbench tracking. Route-assisted verification adds
`--mode route-assisted`. Original recorded timestamps and inferred-reference
unknown intervals remain in scoring; no latency is subtracted from the results.
