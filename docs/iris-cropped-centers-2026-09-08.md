# Cropped mini-map center audit

## Observed defect and cause

The demo previously drew the boundary ellipse and cursor pivot but did not mark
the boundary center. Adding a distinct marker is a requested display feature.
The accompanying coordinate audit found a separate runtime defect for non-unit
normalization scales.

The legacy standalone mini-map homography translated camera coordinates directly
into a phone-pixel crop, while full/dual output and annotation getters used the
rig's normalized output raster. Its output dimensions therefore disagreed with
the dimensions assumed by the center coordinates. Dense-map output already
sampled the normalized raster, but both paths reported a crop-to-phone transform
containing only translation and rotation, omitting normalization scale. Mini-map
rotation bookkeeping also used the unscaled crop dimensions.

The differing standalone/full sampling paths are already present in the code
moved by `e6a8c288` (2026-08-31); this does not establish that the refactor
introduced them. Translation/rotation-only metadata is present in `daaf0705`
(2026-09-01).
This audit demonstrates the defect in the current code with controlled input;
it does not establish whether a particular user profile has a non-unit scale.

## Reproduction and correction

A fake camera emits pixel values encoding sensor X/Y, so expectations come from
actual sampled pixels, independently of the annotation conversion helpers.
The phone boundary is at (26, 31), the rotation center at (28, 34), and the
recentered phone crop starts at (12, 22). Normalization starts at (4, 4).

At scale (2, 3), the normalized full boundary center is (11, 9), its crop starts
at (4, 6), and the mini-map boundary center must be (7, 3). The rotation center
must be (8, 4). The two centers keep their physical offset after cropping.

Before correction, all 16 non-unit-scale combinations failed: two rendering
paths (homography/dense map), two modes (minimap/dual), and four output rotations.
The corresponding 16 unit-scale combinations passed. Failures included an
actual 32x24 image where the normalized contract required 16x8, and metadata
mapping a displayed center to the wrong phone coordinate.

The correction uses the normalized crop dimensions and composes the legacy
mini-map transform from the full-output homography. Crop-to-phone metadata is
precomputed from the inverse output rotation, actual rounded crop origin,
normalization scale, and normalization origin. Both center getters continue
using the same phone-to-full-to-crop calculation. Unit-scale pixel output is
preserved; scaled legacy mini-map dimensions intentionally now agree with dual
mode and the saved rig normalization.

## Calibration-side cropping and publication

Cursor calibration crops Android frames before measurement. `_logical_profile_crop`
converts the boundary to that local crop space; `_geometry_to_canonical_phone`
adds the crop offset and converts both fitted centers back to phone space during
publication. Runtime cropping is a later, separate transformation. No change to
the fitted center estimator or publication activation policy was needed.

Successful components publish independently. Usable partial cursor results can
activate; a later failure does not undo an earlier active profile. Candidates
remain inactive, orientation requires acceptance, and optional color requires
its explicit activation option. Content-identical publication reuses the existing
revision. Geometry-only game calibration publishes portable phone-game data;
local rig composition and adapter initialization consume that data separately.
An already-open adapter does not automatically reload changed profiles.

## Verification and limits

The 32 pixel/geometry combinations pass after correction, checking exact output
images, distinct center positions, and mapping back to phone coordinates. GUI
tests cover separate and coincident centers, boundary toggling, unavailable
cursor calibration, and unchanged input images. A rendered preview was inspected
with both a small offset and coincident centers.

All 112 tests in the selected suites pass. Adjacent tests cover existing camera modes, demo CLI, recalibration contracts,
profile publication, game calibration, and standalone source export. Tests use
temporary files and a simulated camera; no live phone/camera session or packaged
executable build was run. Raw unrectified projective output still reports that
canonical circle/center annotations are unavailable in stream space.
