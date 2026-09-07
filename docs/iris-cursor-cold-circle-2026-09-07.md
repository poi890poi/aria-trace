# Cursor pivot with uneven direction samples

Type: bug fix. The old rotation-center owner compared the aggregate temporal
heatmap with its 180-degree reflection, including a reflection registration
refinement. Both require an approximately balanced observed envelope. Balanced
joystick commands do not guarantee balanced cursor directions or dwell times.
On an independently rendered incomplete dot orbit, that mechanism misplaced the
pivot by 4.91 pixels.

The replacement fits a small circle on the inward edge of the temporal cold
region, using cold-facing gradient Hough votes near the supplied mini-map
center. Circle radii are searched from 2 to 15 pixels, bounded by the local
search scale; the existing center search radius still limits the allowed
offset. Missing sectors supply no votes and incur no balance penalty. A local
normal intersection refines the center below the half-pixel vote grid.
Per-pixel temporal range makes the fit invariant to duplication and ordering
of identical frames. Angular occupancy is diagnostic only.

The estimator receives only captured frames, the mini-map center, and the
configured search radius. Known synthetic pivots are withheld from fitting and
used only to calculate Euclidean error. The mini-map center bounds the search;
it is never silently substituted for an unobservable pivot.

## Independent comparison

Baseline: `b9be825e0f3bbb3494674e13d4f4e5d1f2803514`.
Reproduce from the repository root:

```powershell
./.tools/standalone-release-py31210/Scripts/python.exe -B -m benchmarks.cursor_pose.calibration_cold_circle
```

The harness emits each input's SHA-256, raw fitted coordinates, quality
diagnostics, availability, and isolated fit time. Local run output is
`artifacts/cursor-cold-circle-comparison.jsonl`. Synthetic frames use 4x
supersampling with pixel-center compensation, static textured backgrounds,
and a cursor outside the legacy cyan HSV interval. Both methods receive
identical frames and the same initial center `[64, 64]` and 12-pixel search.

| Input | Triangle old/new error (px) | Dot old/new error (px) |
|---|---:|---:|
| Balanced 12 directions | 0.40 / 0.02 | 0.36 / 0.02 |
| One direction repeated 35 times | 0.23 / 0.02 | 0.58 / 0.14 |
| Incomplete disc, 0–205 degrees | 0.46 / 0.03 | 4.91 / 0.48 |
| Short arc, 0–90 degrees | 0.82 / 0.39 | 6.30 / 1.25 |
| Fresh pivot `[66.6, 61.2]`, scale 0.85, noise 0.6 DN | 0.98 / 0.06 | 2.16 / 0.22 |
| Fresh pivot `[59.8, 65.4]`, scale 1.3, noise 0.6 DN | 0.88 / 0.08 | 4.49 / 0.19 |

The final two rows were introduced after the cold-facing vote method was
selected on the development inputs. They use new direction sets, pivots,
scales, background seed, and modest independent pixel noise. No threshold was
tuned on these holdouts. The development prototype's original rasterization
had a 0.375-pixel sampling offset; the comparison above corrects the fixture
coordinate convention for both methods.

All 12 observable candidate cases produced fresh fits; none reused a previous
center. Static frames, uniform flicker, and a translating dot were unavailable
with explicit reasons. A straight moving edge is also covered by a rejection
regression. This is a batch calibration estimator: held-pose continuity, loss
episode counts, and tracking discontinuities do not apply.

Single-run candidate fit times on the observable cases were 4.00–6.35 ms;
baseline times were 31.06–100.05 ms. These exclude synthesis, capture, decoding,
evidence rendering, and file I/O. This is diagnostic timing, not a repeatable
performance or device-latency claim.

## Limits and impact

The short dot arc still has 1.25-pixel error and only moderate reported
confidence. Partial-circle geometry can be poorly conditioned; the score is
not calibrated absolute accuracy. No real-phone recording was available for
this change, and physical capture and packaged execution remain unrun. Moving
map clutter, a noncircular cold region, very small or large cursors outside the
searched radius range, and severe image noise need recording-based validation.
The estimator does not promise a valid center from every diverse sequence.

Normal unavailable conditions include no temporal contrast, too few inward
edges, no supported circle in the allowed search, and an arc whose normals
cannot constrain both center coordinates. Those produce reasons instead of
an invented mini-map-center result. No angular-balance or complete-disc gate
is introduced.

Both cursor-only and full mini-map calibration use the shared estimator.
Evidence now names and displays Hough votes (`cursor_center_hough.png`) and
the fitted cold-core circle. Published center-quality metadata carries the
new radius, support, residual, conditioning, and vote score. Existing stored
profiles are not rewritten; legacy quality metadata remains readable.
Boundary fitting, coordinate-space checks, balanced input capture plans,
optional HSV shape fitting, color fallback, and active-profile policy are
unchanged. No new dependency is required.

Decision: land the cold-circle replacement for this demonstrated bias, retaining
the short-arc and physical-data limitations above. Regression coverage includes
known offset pivots, duplicate/order invariance, noisy fresh holdouts, public
calibration and saved geometry, no-motion and translation rejection, balanced
center/shape calibration, profile publication, portable profiles, and game
calibration. Validation log: `artifacts/cursor-cold-circle-regressions.log`.
All 51 tests passed in 37.763 seconds on the bundled Python 3.12 environment.
