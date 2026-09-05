# Full-map stitching benchmark — 2026-09-05

## Decision

Recommend global translation pose-graph adjustment followed by the existing
subpixel highest-feather source owner. This is the only candidate that improves
independently held-out geometric consistency on both available sessions without
introducing a new visible artifact. It still averages no map images: every
covered output pixel has exactly one source owner.

Do not land a seam finder yet. Color graph cut lowers measured seam jumps and
crosses fewer protected line features, but visual review exposes broad source
tile/tonality blocks. DP color-gradient and graph-cut color-gradient are clear
negative results.

The recommended placement has since landed. A post-land audit found that the
first integration used optimized positions for canvas bounds and metadata but
still used sequential positions for composition. Commit `abe644f` repairs that
authority break and adds a regression proving pixels use the optimized pose.
Post-build owner and seam evidence landed separately in `b29b4b2`.

## Evidence contract

- Input is the original session video and a completed `map_stitch.json`.
- The recorded production placement and compositor are the baseline.
- Loop constraints are found only among non-neighbouring, overlapping source
  viewports.
- A deterministic fifth of accepted loop constraints is withheld from fitting.
- All compositions use identical decoded frames and positions for each seam
  comparison.
- Output pixels select one subpixel-warped source pixel; mixed-source count is
  zero for every method.
- The protected-feature mask dilates stable Canny edges. It covers roads,
  streets, labels, shorelines, and other line detail; it is intentionally not a
  game-specific semantic road classifier.
- Seam metrics do not supersede visual evidence. In particular, local boundary
  scores can miss a broad low-frequency tile change.

## Placement result

| Session | Selected views | Held-out loops | Placement | Median | P95 | Worst |
|---|---:|---:|---|---:|---:|---:|
| Run 19, high-detail map | 437 | 169 | sequential baseline | 2.52 px | 5.12 px | 6.13 px |
| Run 19, high-detail map | 437 | 169 | robust pose graph | 0.26 px | 0.67 px | 1.07 px |
| Run 05, independent older map | 71 | 9 | sequential baseline | 1.01 px | 1.45 px | 1.51 px |
| Run 05, independent older map | 71 | 9 | robust pose graph | 0.18 px | 0.72 px | 0.84 px |

The held-out constraints never participate in fitting. Run 19 supplies the
strong decision evidence; Run 05 is a small independent confirmation, not a
second large truth set. Candidate discovery does use the baseline footprints to
find likely overlap, so this is held-out closure evidence rather than external
ground truth.

## One-change composition ladder

Session 19, committed revision `b158f35959c1edff2589454443cda7110c680cb7`:

| Placement + owner | Seam jump P95 | Protected-feature seam | Seam length | Offline composition | Visual decision |
|---|---:|---:|---:|---:|---|
| sequential + current feather | 24 | 46.2% | 190,720 px | 6.91 s | baseline |
| pose graph + current feather | 20 | 43.8% | 190,772 px | 6.90 s | **recommend** |
| pose graph + DP color-gradient | 94 | 37.5% | 131,724 px | 15.88 s | reject |
| pose graph + graph-cut color | 13 | 29.6% | 194,807 px | 19.67 s | hold; tonal blocks |
| pose graph + graph-cut color-gradient | 43 | 44.9% | 287,134 px | 32.95 s | reject |

Session 05 repeats the ranking: pose-graph/current-feather preserves P95 22,
color graph cut reaches 19, DP worsens to 41, and gradient graph cut worsens to
33. Color graph cut changes valid-image mean luma from 77.88 to 71.77 on Run 05
and from 100.28 to 97.53 on Run 19. That source-selection shift and the visible
large tiles prevent it from replacing the current owner.

The maximum seam jump remains about 205–219 luma levels for all plausible
methods. Those isolated dynamic/UI or high-contrast cases are not solved by seam
placement. A seam method can hide small residual disagreement; it cannot make a
bad or changing source frame geometrically consistent.

OpenCV's all-at-once Voronoi finder failed natively on the 437-view overlap
graph. The failure is retained as an implementation negative, not represented
as an algorithm score. The corrected bounded DP and graph-cut wrappers produced
distinct ownership masks with zero finder failures.

## Tracking change recommendations

1. **Fix full live E2E latency before changing the local matcher.** The offline
   gradient core has 4.88 ms serial decode-to-XY P95, but the latest live trace
   delivers only 22.27 updates/s and 123.97 ms capture-to-publication P95. Use
   latest-frame bounded queues and isolate capture, local estimation,
   global/recovery work, recording, overlay, and publication so slow consumers
   cannot age the control frame. Preserve stage timestamps and stale-frame drops
   in telemetry.
2. **Keep the verified simple local pipeline:** oriented-gradient
   `TM_CCORR_NORMED`, 12 px radius, no 0.50 final gate, accepted-time continuity,
   active layer outside learned transition zones, both layers inside, two target
   confirmations, and hold during an unconfirmed representation transition.
3. **Run global/route proposal work asynchronously and only for initialization
   or genuine recovery.** It must not interrupt a valid continuous local track
   or snap output onto the demonstrated route.
4. **Keep Laplacian CCORR as an A/B challenger, not the default.** It lowers the
   largest measured reference outlier from 16.70 to 12.52 px, but has slightly
   worse fresh coverage and a 55.48 ms serial worst case versus 32.81 ms for the
   gradient baseline.
5. **Do not revive rejected candidates:** intensity CCORR, gradient phase
   correlation, Canny CCORR, the 0.50 hard acceptance gate, or frame-clock
   continuity.

The tracker is not yet a 30 FPS E2E pass. Offline matcher success is necessary
but not evidence that capture-to-control latency passes.

## Mapping change recommendations

1. **Add robust translation pose-graph adjustment after keyframe registration.**
   Retain pairwise chain edges, add validated non-neighbour overlap closures,
   solve both axes with robust weighting, and preserve the original gauge. Write
   every accepted/rejected loop, residual, and final tile transform to the map
   artifact for review.
2. **Keep the current hard highest-feather source owner as default.** Apply the
   optimized positions, but do not average, multiband blend, or Poisson blend map
   artwork. The benchmark's center-area review shows the Windrise road split is
   repaired by placement alone.
3. **Add post-build quality evidence, not an invented pass threshold:** held-out
   loop residual distribution, seam overlay, protected-feature crossing rate,
   low-frequency tile-step map, source-owner map, holes, and coverage. More fresh
   map sessions are needed before choosing universal accept/reject limits.
4. **Keep color graph cut as research code only.** Before retesting it, exclude
   fading, animated, overlay-obscured, or tonally inconsistent source tiles and
   add an independent low-frequency step metric. Then test an explicit seam cost
   combining photometric disagreement with a penalty around stable line
   features, so unavoidable seams prefer water/terrain interiors over roads and
   streets.
5. **Reject DP color-gradient and graph-cut color-gradient for this source.**
   They are slower and visibly worse. Do not add a semantic road model until the
   generic protected-feature cost has failed on wider data.

## Traceability

- Benchmark implementation: commits `413733b`, `60d4e3b`, `8812fd6`, `b158f35`.
- Production placement integration: `3e6618f`; compositor authority correction:
  `abe644f`; post-build quality evidence: `b29b4b2`.
- Run 19 machine result:
  `artifacts/poc/map-stitch-session19-20260905-final/results.json`
- Run 19 comparison:
  `artifacts/poc/map-stitch-session19-20260905-final/comparison.jpg`
- Run 19 per-method full mosaics and seam overlays are in sibling method
  directories.
- Run 05 machine result:
  `artifacts/poc/map-stitch-run05-20260905/results.json`
- Run 05 comparison:
  `artifacts/poc/map-stitch-run05-20260905/comparison.jpg`
- Tracking report: `benchmarks/localization/LIVE_E2E_BENCHMARK_20260905.md`.
- Run 19 artifact SHA-256:
  `20547b49e9c4ef86f1bdb4248d72400bce1b4d541d5c0b5e49f717539783478f`.
- Run 19 video SHA-256:
  `a26c3faaadbadf964a59f071c390ceaeaf7f7dd2416320f8ebd007cfb77c6a10`.
- Run 05 artifact SHA-256:
  `b2bca768c3b715dcc6c7f9e28457eb7e94645d4fb254e4e63fa670e48904556a`.
- Run 05 video SHA-256:
  `6cd058189ec470f542cc51fe338078b4eb598b50d0b96ddc5d5dca3f37022039`.

The tracked worktree was clean for both final runs. Unrelated untracked user
files were not read, modified, or included.
