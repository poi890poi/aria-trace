# Cursor center noise audit, 2026-09-08

Type: noise bug fix in the shared cursor estimator, followed by an evidence-rendering feature.

## Contract and cause

An incomplete cold disc with diverse directions must locate the cursor pivot
without balanced direction counts. A direction observed once must remain useful
even after a long dwell in another direction. Static or empty noisy frames must
not manufacture a rotation center. Acceptance target for the named synthetic
rotating inputs is less than 1 px error against the independently drawn pivot.

The cold-circle estimator introduced in `a46767f` retains per-pixel extrema to
avoid dwell weighting. At baseline `d869428`, those extrema also retain noise
spikes. A fixed intensity-contrast threshold of 2 permits noise to create edges;
the small-circle Hough search can then fit accidental arcs. Good normal
conditioning and a small radial residual do not independently prove rotation.
This audit reproduces that code flaw; it does not diagnose an inaccessible
deployment recording.

The correction belongs in `calibration/cursor/center.py`:

* Apply a small linear spatial Gaussian filter before temporal extrema. This
  attenuates isolated spikes without temporal trimming or direction weighting.
* Compare temporal contrast against measured spatial range noise, frame noise,
  and the high tail of range noise. The tail is necessary for sparse impulses
  that median-based noise estimates miss. Report measured values on rejection.
* Keep Hough fitting, incomplete-arc conditioning, search bounds, and diagnostic
  confidence. The initial mini-map center constrains the search, never replaces
  an unobservable fit. Exact duplicates and frame order do not change the fit.
* Retain the raw Hough circle separately from the refined center for evidence.

No profile migration, capture timing, color requirement, boundary refit, adapter
behavior, or new dependency is introduced. Existing profiles remain unchanged
until recalibration. Filtering can affect very small or weak cursor features;
synthetic tests cannot establish performance on all phones or moving map textures.

## Experiments and decisions

Truth comes from supersampled raster drawing in `tests/test_cursor_cold_circle.py`.
The estimator sees only frames, initial `(64, 64)`, and search radius 12. Actual
pivots are used only to draw inputs and score returned centers. There is no held
pose or fallback center; each outcome is fresh acceptance or unavailable.

| Cohort | Baseline | Final correction |
| --- | --- | --- |
| Primary: 16 rotating, Gaussian sigma 0/2/6/12 | 16 accepted; max error 0.526 px | 16 accepted; max error 0.432 px |
| Primary: 32 static/empty controls | 16 false acceptances | 0 false acceptances |
| Development fresh set: 24 rotating, dwell 30/180/350 plus six single frames | 24 accepted; max error 8.563 px | 24 accepted; max error 0.290 px |
| Development fresh set: 48 static/empty controls | 36 false acceptances | 0 false acceptances |
| Final unseen validation: 16 rotating, dwell 90/240 plus six single frames | 16 accepted; max error 1.201 px | 16 accepted; max error 0.279 px |
| Final unseen validation: 32 static/empty controls | 24 false acceptances | 0 false acceptances |

The two larger cohorts include dots and triangles, different pivots and sizes,
Gaussian sigma 2/12, and 0/0.003 random-pixel impulse probability per frame. Empty
controls repeat across the shape loop and are not independent across shapes.
The final validation seeds are 55217 and 55313; they were fixed after proposing
the final noise-tail test, and no parameters were changed after opening them.

Rejected intermediate candidates are relevant:

* Range-MAD gating alone fixed the primary Gaussian false positives but retained
  three static impulse failures and a 1.967 px rotating error on the first
  unbalanced impulse holdout (120 dwell frames plus six single directions).
* Full median filtering and selective spike replacement improved rotating
  accuracy but produced false circles at stationary edges. Low-noise impulse
  stress cases still failed. These filters are not in the final implementation.
* Linear filtering plus frame/range-MAD gates passed that stress set, but four
  of 48 static/empty controls in the next set still passed. The noise-tail test
  addresses these non-Gaussian tails. That set consequently became development
  data; only the subsequent validation set is the final unseen holdout.

Offline fitting becomes slower: final validation rotating median/p95 was
111/169 ms versus baseline 9/24 ms, measured around the estimator only. These
runs shared the machine with other checks, so they are not a controlled speed
benchmark. Capture, synthetic generation, evidence rendering and disk I/O are
excluded. Filtering is restricted to the local search ROI; no per-frame adapter
work is added.

## Reproduction and evidence

Run the checked-in harness from the repository root, using the project Python:

```powershell
python -B -m benchmarks.cursor_pose.calibration_noise --suite validation --baseline --output artifacts/cursor-noise-validation-baseline.json
python -B -m benchmarks.cursor_pose.calibration_noise --suite validation --output artifacts/cursor-noise-validation.json
```

Other suites are `primary`, `stress`, and `fresh`. `--baseline` reads the pinned
estimator with `git show`; it does not change the checkout. The final estimator
can be exercised without Git. Raw output records seeds, pivot, size, directions,
noise, outcomes, fit diagnostics, independent errors, and estimator timing.
Additional local development results are under `artifacts/cursor-noise-*.json`.

Both combined mini-map and standalone cursor calibration now overlay a dashed
cyan Hough circle, green refined circle, and red pivot in the existing temporal,
Hough-vote and mean-image PNGs. `cursor_center_fit.png` adds a magnified comparison
with an unannotated mean capture, temporal range, Hough votes, legend, coordinates
and frame count. Geometry uses the calibration crop's pixel-center coordinates.

The inspected example is
`artifacts/cursor-unbalanced-evidence/cursor_center_fit.png`: triangle pivot
`(59.8, 65.4)`, 120 dwell frames, six single directions, Gaussian sigma 6 and
0.003 impulse probability. Returned pivot `(59.728, 65.562)` is 0.177 px away.
The evidence is generated through the public standalone calibration path.

Normal unavailable conditions still include absent rotation/temporal signal,
signal submerged in measured noise, too few observable cold-core edges, an arc
that cannot constrain two coordinates, or a circle outside the local search.
The public entry point also requires enough same-sized frames and matching
coordinate-space metadata. Unbalanced counts alone are not a rejection reason.

Verification covers cursor, combined mini-map, partial shape failure, profile
publication and recalibration/adapter contracts. Physical-phone capture,
deployment execution and packaged release were not run. The results establish
the named noise regressions, not immunity to arbitrary noise or scene motion.
