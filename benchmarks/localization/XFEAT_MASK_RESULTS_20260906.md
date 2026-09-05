# Cursor and minimap exterior masking for XFeat

The previous XFeat comparison masked keypoint locations, not pixels entering
the network. This follow-up explicitly masks those query pixels before feature
extraction. Neither tested fill nor the separate cursor/exterior masks improves
the global initializer. Keep production unchanged.

## What is masked

The calibrated 138x138 crop uses a valid circle of radius67 pixels and a central
exclusion disk of radius14. The retained area is 70.68% of the crop. The cursor
fits inside the central disk in the three inspected frames (Run11 frames0/800,
Run17 frame749). Fog, the north indicator and other icons can remain inside the
circle; this is not semantic removal of every game overlay.

Previously, feature centers in excluded locations were removed after XFeat had
processed the image. Descriptors could therefore depend on those pixels. Here,
only the grayscale query image supplied to feature extraction changes. The
atlas descriptors, feature-location mask, original correlation image, geometry,
coverage and acceptance checks remain unchanged. Fill is either zero or the
rounded mean brightness of the valid map pixels. At 2x, filling happens before
resizing and feature coordinates return to the original image space.

The actual original/mean/zero input visualization was opened and inspected:
`artifacts/poc/tracking-xfeat-mask-20260906/input-mask-example.png`.
The mask-contour views are in the same directory.

## Same-frame comparisons

Each row below uses the same 46 samples from each session's first 45 seconds.
These are independent unrestricted two-layer queries, not initialization times
or full live-tracker runs.

| Method / query input | Demo11 accepted / 46 | Outdoor17 accepted / 46 |
|---|---:|---:|
| SIFT, repeated unchanged control | 44 | 21 |
| XFeat ratio at native resolution, mean fill both | 0 | 0 |
| XFeat ratio at native resolution, zero fill both | 0 | 0 |
| XFeat mutual matching at native resolution, mean fill both | 0 | 0 |
| XFeat mutual matching at native resolution, zero fill both | 0 | 0 |
| XFeat ratio at 2x, repeated unmasked control | 11 | 0 |
| XFeat ratio at 2x, mean fill cursor only | 6 | 0 |
| XFeat ratio at 2x, mean fill exterior only | 5 | 0 |
| XFeat ratio at 2x, mean fill both | 3 | 0 |
| XFeat ratio at 2x, zero fill both | 2 | 0 |

The SIFT and 2x unmasked controls reproduce the preceding evaluation's accepted
counts. The masked 2x variants also start accepting later in the sampled demo:
38.75 seconds for either individual mean mask, 43.82 for both mean masks and
44.82 for both zero masks, compared with 25.55 for the unmasked query control.
These are first accepted independent queries, not the stateful runtime's first
pose. Do not substitute them for E2E acquisition measurements.

For 2x XFeat on the demo, whole-query mean/P95 times are 221/660 ms unmasked,
175/602 ms cursor-only, 175/614 ms exterior-only, 147/490 ms both mean-filled,
and 121/118 ms both zero-filled (the zero-fill worst query is 610 ms). Lower
times mostly reflect more rejected queries skipping expensive correlation; they
are not successful localization speedups. The mean exceeding P95 for zero fill
comes from two expensive queries in 46 samples. Detailed CPU distributions and
per-sample results are retained in the generated comparison.

The apparent lower accepted-position P95 after masking is again selection:
only 2–6 late demo observations remain. It does not establish better accuracy,
continuity, longest-lost duration or control quality.

## Decision and limits

Reject these input-mask variants for integration. Confidence is high for the
observed lack of benefit on these recordings and settings. This does not prove
that masked pixels never matter or that all mask-aware feature methods fail.
Constant filling changes image appearance and removes surrounding context as
well as overlays; the experiments do not isolate the unique cause of degradation.

The component tests prevent blaming only the combined intervention: neither
cursor-only nor exterior-only mean masking improves the best partial XFeat
configuration. We did not tune mask radii, add semantic icon segmentation,
alter atlas extraction or weaken pose acceptance to obtain a positive result.
Masked LighterGlue and masked 2x mutual matching were not evaluated.

There are 920 probe evaluations across 92 unique source observations, with
matching observation hashes checked. These are correlated samples, not 920
independent scenes. The atlas/calibration and reused frozen reference outputs
are hash checked. The references share SIFT and atlas assumptions and are
inferred proxies, not external truth. No reference position enters inference.
No new E2E or full-lap replay was justified after the query-stage regressions.

26 focused tests passed, including input immutability and separate cursor/exterior
mask semantics, coordinate conversion, mutual matching, replay scoring, loss
evaluation and production integration. Experimental tooling only changed;
production tracking and dependencies remain unchanged.

Evidence: `artifacts/poc/tracking-xfeat-mask-20260906`. `comparison.json` and
`COMPARISON.md` are generated from the preserved probe rows; each run contains
executed source snapshots, model identities, setup costs and per-layer queries.
The protocol is [XFEAT_MASK_PROTOCOL_20260906.md](XFEAT_MASK_PROTOCOL_20260906.md).
The preceding unmasked evaluation is [XFEAT_RESULTS_20260905.md](XFEAT_RESULTS_20260905.md).
