# Known-start template tracking

Supplying the demonstrated start lets the existing masked gradient template
tracker measure XY on the first frame. It removes global acquisition delay on
the demo and a second recording with a town start. It does not remove the
remaining representation-transition loss, position jumps, or source-delivery
delays. This is a supported startup direction, not autonomous-cruise validation.

## What was changed and compared

Production already matches the current minimap against a bounded atlas window
for steady XY. The experiment initializes its search from the explicitly supplied
demonstrated position, map alignment and world/town layer. The first visual
refinement searches the existing 55 canonical-pixel radius; subsequent matching
uses the existing local radius. Cursor and minimap exterior pixels retain their
existing exclusion mask. No new model or motion accumulation is needed.

The start is recorded as a prior, not a fresh measurement or player heading.
Fresh XY is still set only by the runtime's current-image measurement result.
Only the first demonstration state supplies this startup prior; it is not reset
on later frames. Tests also verify that injecting a prior does not itself mark
an unavailable visual measurement fresh.

The primary free-roam harness comparisons isolate this start-only intervention
from full-route guidance. They are explicitly known-start tests despite the
harness mode name. The route-assisted pair additionally uses the existing
Run11 route package and transition anchors, unchanged between its two runs.
Run14 uses its own declared first-state prior for a separate development test;
it is not represented as another traversal of Run11's demonstrated route.

## Full demo replay

All rows below replay the same 59.98-second Run11 recording at source cadence.
First fresh XY is source time from the first recorded frame, not zero execution
time. Reference errors are canonical atlas pixels against the cached inferred
reference, not independently measured game-world coordinates.

| Method | First fresh XY s | Error P95 px | Longest lost s | Longest lost after acquisition s | Fresh / held / unavailable frames | Publication P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Existing unbounded SIFT startup | 2.201 | 2.783 | 2.201 | 0.402 | 1714 / 20 / 64 | 29.13 |
| SIFT bounded to known start | 2.033 | 2.782 | 2.033 | 0.402 | 1720 / 20 / 57 | 26.14 |
| Translation-only gradient startup, bounded | 4.061 | 2.774 | 4.061 | 0.402 | 1660 / 20 / 118 | 21.61 |
| Seed existing local template tracker | 0.000 | 2.782 | 0.402 | 0.402 | 1779 / 20 / 0 | 26.19 |
| Same seed, prior shifted +40px X | 0.000 | 2.782 | 0.402 | 0.402 | 1780 / 20 / 0 | 22.66 |
| Same seed, prior shifted -40px Y | 0.000 | 2.782 | 0.402 | 0.402 | 1780 / 20 / 0 | 33.71 |
| Existing route-assisted startup | 2.058 | 2.762 | 2.058 | 0.302 | 1711 / 28 / 59 | 27.17 |
| Route assistance + known-start seed | 0.000 | 2.762 | 0.302 | 0.302 | 1772 / 28 / 0 | 29.88 |

Source count is 1800; processing drops differ slightly and are retained in the
raw reports. The first three seeded start-only runs all produce exactly the
same first measured XY, [1822.606, 1363.295], including both perturbed priors.
Its inferred-reference error is 2.731px. The first unperturbed seeded frame takes
48.29ms capture-to-publication and has no fresh heading yet. The source-time
acquisition of zero therefore does not mean immediate complete control output.
The unperturbed seeded full run performs no global localization queries.

Startup changes leave the same 0.402s transition loss near 37.3s, six output
steps above 8px, and 19.31px maximum step in the start-only demo tests. Full-route
assistance has one step above 8px, maximum 13.65px, and 0.302s longest transition
loss with or without the known-start seed. These jump cutoffs are descriptive,
not newly chosen gameplay tolerances. Known-start seeding composes with the
existing assistance; it does not explain or fix those remaining discontinuities.
The demo has 3.904s without sufficient reference evidence; these are not counted
as demonstrated accurate tracking.

## Why the separate template initializer is slower

The bounded translation matcher finds the first position, but it still enters
the original global consensus mechanism. The first two measured positions are
31.14px apart after roughly two seconds of movement, exceeding the existing
30px consensus displacement limit. Consensus resets and the third query at
about four seconds initializes. Query time itself is only about 3ms after the
first 12.72ms call. Bounded SIFT keeps geometry checks and initializes at 2.033s
in its paired full run. Replacing global feature proposals alone is not the
startup improvement under the user's known-start assumption.

The first gradient run also exposed a diagnostic inherited from the global
peak-selection code: when suppression covers the whole bounded window, its -1
sentinel was emitted as a fictitious alternative and inflated the reported
margin. The experimental implementation now omits suppressed alternatives and
uses zero as the no-alternative comparison score. Original evidence is preserved;
the first three poses remain above all applicable acceptance gates after this
correction. A repeated 9.98s startup clip with corrected diagnostics reproduces
4.061s acquisition. A focused compact-window test checks this behavior. Neither version
provides a competing-peak check inside the suppression neighborhood, and neither
manufactures feature inliers or feature/correlation agreement.

Grayscale translation matching has synthetic correctness tests but no recorded
E2E measurement in this evaluation. Unknown-position global template probes and
template-only global recovery were not evaluated. Earlier XFeat experiments
addressed unknown-position global initialization, so their failures are not a
like-for-like comparison with a tracker supplied the correct start.

## Town-start reverse recording and timing uncertainty

Run14 is 19.94s, with 465 source frames. The two seeded runs each produce a fresh
position on their first frame, at 54.03ms and 92.85ms publication latency. They
have no global queries. The first position coincides with the inferred start;
this same-frame agreement is not independent evidence of absolute accuracy.

| Run14 comparison | First fresh XY s | Error P95 px | Longest lost / after acquisition s | Fresh / held / unavailable | Publication P95 ms |
|---|---:|---:|---:|---:|---:|
| Control | 3.578 | 2.886 | 3.578 / 0.311 | 367 / 16 / 74 | 81.52 |
| Known start | 0.000 | 2.828 | 0.474 / 0.474 | 445 / 13 / 0 | 289.15 |
| Control repeat | 2.983 | 2.858 | 2.983 / 0.231 | 369 / 16 / 62 | 144.45 |
| Known start repeat | 0.000 | 2.797 | 0.311 / 0.311 | 447 / 16 / 0 | 85.88 |

The seeded runs have longer post-acquisition loss in both pairs. They do not
establish a steady-quality-neutral change across recordings. Source scheduling,
frame drops and transition confirmation also vary, so attributing the difference
to the start prior alone is unresolved. Run14 has 1.001s unknown-reference time.

In the first seeded reverse run, the worst frame is published after 513.89ms:
339.36ms release lateness, 172.45ms decode and only 1.64ms tracker update. Its
source decode worst case is 446.99ms and release-lateness P95 is 266.17ms. The
baseline repeat also degrades to 144.45ms publication P95. One focused test run
overlapped the first seeded reverse replay; that pair is not an isolated resource
comparison. The repeat pair ran without concurrent benchmark/test workloads
initiated by this task. These observations establish substantial source-delivery
variation, not an exhaustive attribution of host load or a tracking speedup.

Even among the three start-only seeded demo runs, joint fresh XY and heading
delivered within 33.3ms varies from 93.72% to 97.00% of source frames. A single
passing percentage would not demonstrate sustained control readiness. Before
production integration, repeat the reverse transition with controlled delivery
and obtain fresh same-route recordings with a separately recorded start prior.

## Failure control and limits

A deliberately wrong +200px X prior fails to acquire a reference-verified pose
throughout its 9.980s clip. Error P95 is 243.47px even though 206 of 300 processed
frames are marked fresh. The loss remains unresolved at the clip end and includes
2.930s of unknown-reference intervals. This demonstrates why fresh updates and
smooth output alone cannot establish location correctness. Existing local
acceptance is permissive and continuity cannot validate an arbitrary first seed.
The supplied known-start condition must actually hold; these tests establish
neither a universal start-error tolerance nor arbitrary-start recovery.

All reference, source-recording, atlas and calibration input hashes are checked
by the report builder; cached reference outputs are also checked and reused.
References share the atlas and SIFT assumptions, and the demonstrated first
state is explicitly reused as a runtime prior. Position corrections and subsequent
tracking are measured from images, but these are development replays, not fresh
cross-session repetitions or external truth. There is no independent heading
reference or active closed-loop steering validation.

## Decision

- Known-start local template seeding: retain as the preferred route-start design.
  Confidence is high for removing observed acquisition delay on the tested
  recordings and offsets. Overall continuity confidence is limited by the reverse
  post-acquisition regressions and delivery variation; generalization to fresh
  route traversals remains unverified. Retain experimentally rather than changing
  production defaults now.
- Bounded SIFT alone: measured modest acquisition reduction on the demo, while
  retaining its consensus delay; not the main solution to known-start tracing.
- Bounded translation global initializer with existing consensus: reject as a
  startup improvement in this configuration. It finds XY but waits longer.
- Known start plus existing route assistance: supported composition on the demo;
  remaining transition errors persist. It does not establish autonomous-control
  readiness or a reliable publication-latency improvement.

Production startup defaults, dependencies and route storage are unchanged.
The delivered changes are the reproducible experimental harness, behavioral tests
and this report. All 28 focused tests passed (template/start semantics, production
integration, loss scoring and replay scoring). Fourteen replay evaluations cover
579.51 recorded seconds across two recordings, including two 9.98s clips; these
are correlated development observations, not fourteen independent routes.
The next production step needs an explicit route-start path
that obtains a current-image measurement before reporting verified acquisition;
it must not silently apply a demonstrated start to unknown-position free roaming.

Evidence root: `artifacts/poc/tracking-template-20260906`. Each cohort retains
raw/scored telemetry, actual Workbench output and input/source identities.
`comparison.json` and `COMPARISON.md` are regenerated with
`python -m benchmarks.localization.build_template_report <evidence-root>`.
See [TEMPLATE_PROTOCOL_20260906.md](TEMPLATE_PROTOCOL_20260906.md).
