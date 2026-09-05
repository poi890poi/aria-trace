# Map-stitching benchmark

This benchmark asks two separate questions:

1. are the source viewports placed consistently; and
2. given unavoidable residual disagreement, where should a hard source seam go?

It never averages map images. Every covered output pixel is owned by exactly one
subpixel-warped source viewport. The current production composition is retained
as the baseline.

## Contract

Input is an existing reviewed `map_stitch.json` plus its original session video.
The recorded selected frame indices and registrations define the baseline. No
calibration or live-tracking result is used.

Loop constraints are discovered only between non-neighbouring viewports whose
baseline footprints overlap. A deterministic fifth is held out before global
translation fitting. Held-out residuals therefore evaluate placement but never
steer it.

Seam methods are compared on identical frames and positions. Evaluation reports:

- held-out loop residual in pixels (placement; lower is better);
- seam RGB jump (visible discontinuity; lower is better);
- fraction of seams crossing dilated stable map edges (roads, streets, labels,
  shorelines, and other line detail; lower is better);
- covered area, holes, seam length, runtime, and output sharpness; and
- ownership completeness. A value of zero mixed pixels proves that no source
  images were averaged.

The edge-protection metric is deliberately broader than a game-specific road
classifier. Review the rendered seam overlay to distinguish roads/streets from
other protected details.

OpenCV's batch Voronoi seam finder is not a runnable candidate for the large
session-19 overlap graph: it raises an opaque native exception. The benchmark
records this negative implementation result rather than disguising a fallback
as Voronoi output. DP and graph-cut candidates use bounded pairwise updates.

## Run

```powershell
.\.tools\standalone-release-py31210\Scripts\python.exe -m benchmarks.mapping.run_stitch_benchmark `
  --artifact artifacts\workbench\map_stitches\genshin-impact-pc\<id>\map_stitch.json `
  --output artifacts\poc\map-stitch-<name>
```

Commit the benchmark implementation before a decision run. A dirty-source run
is diagnostic only under the repository benchmark policy.
