# Map atlas contract

The atlas starts with one sharp, highest-detail master stitch in canonical map
pixel coordinates. Workbench selects the current master automatically using
comparable observed world coverage, then source detail. Rendered mini-map
scales are derived views of that master, not independently stitched maps and
never enlarged substitutes.

Generated history remains on disk for traceability. Routine selectors hide a
stitch when its source session UUID no longer matches the current recording in
that run folder, and show only the newest ready atlas for each master plus
transition provenance pair. Explicit artifact IDs remain accepted by the API.

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

In route-assisted mode, entering a learned transition zone also arms the
matching demonstrated route transition. The tracker holds its last verified XY
while the representation is ambiguous. After the target layer is confirmed,
the first trained target state supplies a bounded search center; it is not a
pose measurement. Tracking resumes only when current-frame target-layer map
correlation validates a position near that center. Existing route packages use
their transition center state as the compatible target anchor; newly compiled
packages record the last source and first target states explicitly.

Models without spatial evidence retain likelihood-only behavior solely for
legacy compatibility. New learned models always write `transition_zones` and
also retain `canonical_boundary` while old readers exist.

## Rebuild order

1. In Workbench, run **Map stitching** for the highest-detail full-map session.
   This calls `WorkbenchAnalysisMixin.run_map_stitch` and must report bounded
   gradient correlation in `[-1, 1]`.
2. Under **Canonical multi-scale atlas**, review the automatically selected
   master summary.
3. Select no transition for a single-scale atlas, or select a reviewed
   transition recording to derive the observed scales and spatial switch zone.
4. Recompile route packages against the new atlas before live route tracking.

Existing atlas and route artifacts are immutable evidence. Rebuilding creates
new atlas/route IDs; compare their manifests before changing the live selection.
