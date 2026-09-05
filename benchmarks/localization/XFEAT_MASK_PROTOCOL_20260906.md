# XFeat input masking follow-up

The preceding evaluation filtered feature locations with the calibrated minimap
mask but did not hide masked pixels from the learned extractor. This tests that
missing preprocessing factor. Base production remains f607fa6; previous
experiment/report commit d6e78bd. Use the same official model, CPU environment,
atlas, calibration, 46 development samples each from Run11/17, frozen references
and unchanged final pose checks.

The input is 138x138, with an outer valid radius of 67 and a central exclusion
radius of 14 pixels. Inspected frames Run11 frame0/800 and Run17 frame749 show
the central cursor within that exclusion. Fog and other icons extend inside the
circle and are not removed by it. Diagnostic mask overlays are saved under
artifacts/poc/tracking-xfeat-mask-20260906.

First change query feature-extraction pixels only: fill the excluded region with
the rounded mean of valid grayscale pixels. Leave atlas extraction, the feature
location mask, original image used for correlation, geometry and all gates
unchanged. Compare native XFeat ratio/mutual variants and the previously partial
2x ratio variant. Repeat unmasked controls on the same host; previous raw outputs
remain immutable. Test zero fill as a distinct appearance intervention if needed.
If a combined cursor/exterior mask helps, ablate the two components separately.

The combined mean and zero masks instead reduce the partial 2x ratio demo result
from 11/46 to 3/46 and 2/46. Also test cursor-only and exterior-only mean filling
on that configuration: a harmful combination could hide opposing component
effects. This is a bounded attribution check, with no new thresholds or tuning.

Acceptance still requires better acquisition/continuity without false accepted
poses or worse output age, not merely more matches. Stop wider replays if global
query quality remains clearly worse than SIFT. If a candidate survives, run the
actual Workbench at original source cadence and score longest loss with acquisition
shown separately. Same-atlas SIFT references are evaluation proxies only; no
reference-centered crop is permitted in inference. No fresh holdout or autonomous
cruising claim is available from the existing recordings.
