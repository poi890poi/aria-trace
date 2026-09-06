# Scale-search cost experiments

Type: performance experiments and recorded-source verification. Baseline is
`cdb671a`'s reusable 17-scale/subpixel matcher, with visual transition confirmation.
The live default, active atlas, route compilation and game controls are unchanged.

First isolate parallel execution of the same 17 independent image matches, using
1/2/4/6 workers with six existing OpenCV threads. Preserve hypothesis order for
ties, score/coverage acceptance, subpixel fitting, current-image pose authority,
discrete mode confirmation and startup rejection. No scale may be omitted.
Measure native paired output equality, wall/CPU time, then complete recorded-
source Workbench replay. Worker lifecycle must close cleanly; oversubscription
or higher joint-delivery latency disqualifies an apparent matcher-only speedup.

If necessary, separately screen algebraic masked normalized correlation using
two unnormalized correlations and explicit normalization. Binary and weighted
masks, finite/zero-energy responses, coverage, fractional peak and winning scale
must be checked. Floating-point differences are reported, not called exact.
Only combine independently supported changes.

Use frozen Run 20 current-frame images and the prior visual-only control's
proposals for the paired screen; later reference states are never search input.
Sample the six previously selected transition peaks and evenly spaced stable
frames, with no tuning on reverse/demo recordings. Reuse the validated inferred
reference index for E2E scoring and declared first-state startup only. Preserve
unknown intervals and report longest loss, fresh/held/unavailable output, jumps,
wrong fresh mode, source-frame coverage, and joint XY/heading deadline delivery.

Screening equality is transfer evidence, not position truth. Run 20 is development
data; Runs 14 and 11 are historical direction/route integration checks, not new
holdouts. No recording newer than Run 20 currently exists. Repeat surviving E2E
timing comparisons; do not promote a change from one isolated kernel speedup.
The target is to preserve the 3.35 px measured worst lap disagreement while
substantially improving delivery. Do not invent a gameplay pixel tolerance.

Evaluate synchronized reuse of scale evidence only as a separate behavior
candidate if cost results justify it. Preserve confirmation and explicit map
mode ownership; do not relabel accurate XY as proof of correct published scale.

Amendment after the independent 25-second concurrency screens: six workers give
19.16 ms engine P95 but 39.93 ms publication P95; source-release lateness is
25.23 ms P95. Separately test recorded-source absolute-deadline waits using
short interruptible sleep steps instead of one event timeout. Preserve scheduled
timestamps, images, order, release-not-before-deadline and prompt cancellation.
Measure lateness and CPU cost. This is benchmark transport fidelity, not a
physical-capture or estimator speedup. Report unchanged-source comparisons
separately from improved-source comparisons and never subtract latency in scoring.
