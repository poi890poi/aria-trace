# Map atlas contract

The atlas starts with one sharp, highest-detail master stitch in canonical map
pixel coordinates. Rendered mini-map scales are derived views of that master,
not independently stitched maps and never enlarged substitutes.

## Build modes

- With no transition session, `build_map_atlas` emits one localization layer
  from the master stitch's reviewed localization derivative.
- With a transition session, Workbench extracts stable before/after mini-map
  references and derives both scale-specific localization rasters from the same
  master artwork. Downsampling is allowed; image enlargement is rejected.
- Explicit multi-stitch `layers` and legacy `town_stitch_id` requests remain
  readable for old artifacts and developer-authored coverage extensions. They
  are not the default scale-building workflow.

`source_map_stitch_id` identifies the master when every layer shares it.
`source_layers` preserves the exact per-layer provenance.

## Transition behavior

Transition calibration records one or more `transition_zones` in canonical map
pixels. Each zone has an ID, center, observed radius, and sample count. The live
representation observer is read-only: its correlation scores may change the
active scale only when the current canonical position lies inside a learned
zone. A switch holds continuous XY, changes the scale, and resets only the local
image reference.

Models without spatial evidence retain likelihood-only behavior solely for
legacy compatibility. New learned models always write `transition_zones` and
also retain `canonical_boundary` while old readers exist.

## Rebuild order

1. In Workbench, run **Map stitching** for the highest-detail full-map session.
   This calls `WorkbenchAnalysisMixin.run_map_stitch` and must report bounded
   gradient correlation in `[-1, 1]`.
2. Under **Canonical multi-scale atlas**, choose that master stitch.
3. Select no transition for a single-scale atlas, or select a reviewed
   transition recording to derive the observed scales and spatial switch zone.
4. Recompile route packages against the new atlas before live route tracking.

Existing atlas and route artifacts are immutable evidence. Rebuilding creates
new atlas/route IDs; compare their manifests before changing the live selection.
