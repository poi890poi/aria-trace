# Tracking implementation status

This is the consolidated status record for the September 5–6 tracking work.
Historical experiment reports describe their state at the time; they are not
the current implementation plan. “Committed” alone does not mean enabled in
the live tracker.

## Latest experiment decision

Latest user-verified route completion: [19:27 milestone](route-tracking-milestone-2026-09-06-1927.md),
tag `route-tracking-known-good-20260906-1927`. The next free-roam test exposes
sustained town holds; live timing also deteriorates substantially in town.
Those are the current priorities, ahead of occasional transitions.

The subsequent [scale-cost experiments](../benchmarks/localization/SCALE_COST_RESULTS_20260906.md)
support implementing bounded parallel scale matching and a separate correction
to recorded-source deadline waits. The late mode-consumption hook did not improve
full-lap loss and is not recommended. These are implementation recommendations;
normal Workbench tracking has not been switched to these experimental methods.

## Last completed implementation round

Completed: reusable visual-transition policy (`63f3654`) and combined
scale/subpixel localizer (`cdb671a`) in service code, explicitly selectable through
the recorded-source benchmark entry point. Both are benchmark candidates; the
existing live default and known-good tag remain unchanged. No HTTP/UI candidate
flag or game steering was added.

Type: intentional tracker behavior implementation plus benchmark integration.

The tracker may use current images, previous estimated XY, atlas pixels and
existing image-confirmed startup. Cached later reference states are scoring
only. Preserve mode confirmation, image score and coverage gates, XY continuity,
heading deadlines, route assistance and legacy single-map fallback.

Verification covered the visual-only policy independently, then the combined
localizer, native parity, full Run 20 E2E, reverse Run 14, route-assisted Run 11,
and a wrong-start rejection clip. Disagreement, longest lost, freshness, jumps
and publication latency are recorded below. The 17-scale candidate remains too
slow, and references are inferred rather than external truth. This implementation
transfer does not claim an efficiency improvement.

## Already implemented in the live application before this round

| Change | Commit | Status |
|---|---|---|
| Current-image atlas XY without requiring route guidance; wider first refinement after global seed | `83c0389` | Enabled; replaces accumulated relative XY on supported atlas localizers |
| Physical continuity measured from the last accepted pose time | `274bcaf` | Enabled; held frames do not reset the catch-up interval |
| Preserve global recovery consensus across accepted route-local measurements | `69eb435` | Enabled; distinct from the relative-motion fallback candidate below |
| Observation-time position uncertainty for transition decisions | `9f1eba3` | Enabled |
| Consume ready heading results after XY work within the source deadline | `b6c6125` | Enabled |
| Observe armed map transitions at control cadence | `e7f3e04` | Enabled |
| Reject invalid feature geometry before expensive global correlation | `cd5cdca` | Enabled; acceptance checks preserved |
| Image-verified demonstrated route start, including UI start selector | `2aa258c` | Enabled; the prior itself never publishes a pose |
| Confirm route finish from visual pose; record route trace through finish | `481ae22`, `55c811c` | Enabled before the recent user-verified route milestone |
| Recorder status isolated from catalog polling | `5c5e461` | Enabled; recording test subsequently confirmed by user |
| Export/import all or selected recorded sessions | `464962f` | Enabled; independent feature commit |
| Optimized map-tile positions actually used in composition | `abe644f` | Implemented before the tracking rounds |
| Global translation loop-closure / pose-graph placement | `3e6618f` | Implemented; compositor authority corrected by `abe644f` |
| Hard-owner seam evidence; avoid redundant minimap extraction; publish capture before recording observation | `b29b4b2`, `f58b859`, `4192793` | Earlier implemented mapping/capture changes; not new scale-search optimizations |

Cursor/exterior minimap masking was already present in the local matcher. The
later XFeat input-mask ablations did not introduce that production behavior.
The successful live-route milestone is tagged `route-tracking-known-good-20260906`
at `9e8de66`; session recording is tagged `session-recording-known-good-20260906`.

## Implemented and tested as experiments before this round

| Candidate | Location / commit | Decision |
|---|---|---|
| Remove hard spatial transition vetoes | `transition_zone_ablation.py`, `b6cf818` | Transferred to reusable service code in `63f3654`; all six narrow-lap switches detected |
| Subpixel peaks + intermediate-scale image matching | `precision_candidates.py`, `eaec05d` | Transferred combined behavior to service code in `cdb671a`; benchmark-only selection because of latency |
| Atlas-owned sparse town/world observations and directed crossing brackets | `map_mode_support.py`, `b6cf818` | Stored in inactive candidate atlas; retain unknowns; no runtime soft prior yet |
| Decode-ahead preserving original frame release times | `prefetched_replay.py`, `d828f4b` | Successful benchmark transport improvement; already reused; not a physical capture optimization |
| Cached slow inferred references, longest-loss and joint-delivery scoring | `4d9b125`, `1386101` and follow-ups | Implemented benchmark infrastructure and reused; not external ground truth |
| Scene optical flow, template tracking, reacquisition and HUD/avatar masking | `scene_bearing_probe.py`, `b43e74e` | Research prototype; no learned target handoff or steering controller |

## Candidate review: retain versus reject

Retain subpixel only as part of the scale-aware candidate: alone it improves
ordinary precision but worsens some transition errors. Retain visual mode
confirmation; removing the spatial veto is not removing confirmation or broad
XY holds. Retain the observed map features without turning sparse support into
a complete town polygon or hard gate.

Rejected or unsupported: score-0.55-triggered scale expansion; endpoint-only
matching as sufficient zoom handling; broad transition hold removal; reverse
route anchors; unconditional scene-processing removal; latest-consensus XY as
a standalone estimator fix; XFeat and tested pixel fills; separate global
template startup; post-switch timestamp reset and ambiguous-scale holds as a
smooth-tracking solution. The full scale sweep remains an accuracy reference,
not an acceptable low-latency implementation.

Earlier K/M candidates also remain experimental: wider post-switch refinement
had mixed reverse-crossing outcomes, and bounding startup from the first valid
global hypothesis did not reliably improve unknown-start acquisition. Their
exclusion from implementation is deliberate, not a missing successful change.

One candidate deserves explicit follow-up rather than being forgotten: experiment
D reproduced recovery consensus being erased by local updates. Its state-machine
evidence was strong, but it did not improve the old estimator's overall tracking,
and the default atlas path was subsequently replaced. The current relative-motion
fallback still clears hypotheses after accepted phase correlation; the separate
route-local version was already fixed in `69eb435`. Review the fallback's relevance
separately before implementing it. This round does not silently add that
independent behavior change.

## Remaining work

Two older challengers were also missing from the short plans: Laplacian CCORR
reduced one reference outlier but lost fresh coverage and had a larger compute
spike; color graph-cut seams improved seam metrics but created visible tonality
blocks. Keep both as lower-priority research, not confident implementation wins.
The existing hard source owner plus corrected global placement remains preferred.

The cost experiments now support parallel matching; that integration is the
next recommended change. Test coherent XY/scale publication and atlas-owned soft scheduling priors;
learn repeatable scene landmarks and demonstrated screen bearings; implement
and evaluate target handoff before any steering. Acquire fresh holdouts when
available. These are remaining tasks, not completed features.

## Verification and commits from this round

The visual-transition service candidate is implemented in `63f3654`. It is
selected programmatically through the Workbench localizer factory and the
recorded-source CLI; the ordinary Workbench still constructs the existing
localizer. The policy preserves learned zones as diagnostics and retains visual
confirmation, without using those zones to veto observations or arm route holds.
107 tests passed for that slice. Full Run 20 detected all six changes: inferred
reference P95 2.914 px, worst 38.011 px, longest lost 1.229 s, 88 held frames,
publication P95 22.641 ms. It is not sufficient alone for smooth transitions.

The combined service candidate (`cdb671a`) passes exact output parity on six frozen native
transition images against the earlier experiment, excluding execution time.
All 115 focused unit/integration tests passed on the final rerun. An earlier
suite attempt encountered a Workbench temporary-file access error on shutdown;
the isolated test and complete rerun both passed.

### Recorded-source integration results

These are full Workbench tracking-loop replays, with the original source release
times and a bounded decode-ahead adapter. Current images must verify the declared
first-state startup proposal. Later cached reference states are scoring only.
The new localizer and transition policy use ordinary service dispatch, with no
algorithm monkey patches; the existing benchmark source and startup-input
adapters remain. No physical capture, new live play, or game steering is claimed.

| Implementation / session | Reference P95 / worst px | Longest lost s | Fresh / held / unavailable | Largest XY step px | Publication P95 ms |
|---|---:|---:|---:|---:|---:|
| Visual transitions, Run 20 narrow lap | 2.914 / 38.011 | 1.229 | 5822 / 88 / 0 | 24.534 | 22.641 |
| Combined scale/subpixel, Run 20 narrow lap | 2.191 / 3.352 | 0.493 | 5669 / 0 / 0 | 2.836 | 72.994 |
| Combined scale/subpixel, Run 14 reverse transition | 1.855 / 2.982 | 0.000 | 457 / 0 / 0 | 2.184 | 59.007 |
| Combined scale/subpixel, Run 11 route-assisted demo | 2.169 / 3.099 | 0.302 | 1710 / 0 / 0 | 2.572 | 68.077 |

The negative startup check offsets the Run 20 proposal by +200 canonical X
pixels. Over a five-second clip it publishes zero fresh or held poses and 114
unavailable outputs. It retains rejection rather than treating the prior as
an observation. This is a bounded rejection check, not full-session recovery.

Both Run 20 implementations detect all six changes. Combined matching removes
the large XY excursions and has no jumps over 8 px. The remaining six loss
episodes are wrong discrete-map-mode intervals: 32 fresh outputs have accurate
XY but disagree with the inferred reference mode. Correct coherent XY/scale
publication remains work to do. Inferred-reference coverage leaves 7.440 s
unknown on Run 20; zero loss in a different session is not proof of full truth.

The combined run processes 5669 of 5931 source frames (95.58% fresh over all
source frames), versus 5822 fresh of 5931 (98.16%) for visual transitions alone.
Joint fresh XY and heading published within 33.3 ms falls from 97.22% to 41.93%
of source frames. Local-template P95 rises from 3.755 to 42.783 ms. Preserve
these costs: “all processed frames fresh” does not mean all source frames were
processed or that updates arrive in time for steering.

The previous patched combined Run 20 experiment had P95/worst 2.194/3.352 px,
longest lost 0.422 s and publication P95 66.743 ms. This service implementation
reproduces the six crossing-window worst errors exactly, but asynchronous mode
publication and processed-frame counts vary; its longest loss is 0.493 s and
publication is slower in this replay. This is implementation transfer evidence,
not a new speedup or statistically exact E2E equivalence claim.

Confidence: high that the reusable matcher preserves the tested image-matching
behavior, supported by native parity and full recorded-source runs; moderate
for generalization. Low confidence in live control readiness because of latency,
discrete-mode lag, no new holdout, and same-atlas inferred references rather than
external position/heading truth. Keep both implementations explicitly selected
for benchmarking; the successful live tag and default remain unchanged.

Evidence: `artifacts/poc/tracker-implementation-20260906/comparison.json`,
`COMPARISON.md`, `precision-analysis.json`, `native-parity.json`, and each
cohort's raw/scored/source telemetry and report. Reference input/output hashes
were revalidated by the report builder; cached slow references were reused.
The historical `experiment.base=f607fa6` label belongs to the old benchmark;
each report's actual `git_revision` and implementation hashes identify this run.

Reproduce with the pinned Python runtime:

```powershell
.\.tools\standalone-release-py31210\Scripts\python.exe -m benchmarks.localization.template_cpu --tracker-implementation scale-aware --representation existing --action replay --runs 20 --output artifacts/poc/NEW-OUTPUT --references artifacts/poc/narrow-lap-20260906/references.json --start-policy verified --start-route 20 --prefetch
```

Use a new output directory. Select `visual-transitions` for the separate policy
candidate, `default` for the current application implementation. The combined
service retains fixed 17-scale search and subpixel fitting together; no selective
score trigger, reduced-scale approximation, or live default promotion is included.
