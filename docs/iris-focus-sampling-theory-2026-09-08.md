# Focus sampling reference

## Change contract (before implementation)

Type: diagnostic feature. The HIK focusing assistant currently displays measured
e-SFR MTF50 and a session maximum. Add a separately labelled ideal MTF50 reference
for the panel, camera, and their combination, including Diamond PenTile and Bayer
channel lattices. Rename the session maximum to `best` to avoid ambiguity.

The reference uses only the configured layout and calibrated local geometry,
including measured lens distortion. It must not depend on measured sharpness,
alter camera controls, change e-SFR curves, or gate calibration/profile activation.
Unknown layouts show named alternatives, never an inferred phone model. Existing
profiles remain readable; optional CLI selectors affect only this diagnostic.

MTF50 cannot be inferred uniquely from sampling density. The proposed reference
is the limiting MTF50 of an ideal independent-channel, flat-passband low-pass
system with cutoff at each lattice's first Nyquist boundary. It ignores aperture,
optics, noise, gamma and ISP. A real processed edge can exceed this reference;
it is not an absolute hardware bound or an acceptance threshold. A universal
`0.5 cy/pixel`, `half Bayer resolution`, or fixed PenTile multiplier is rejected.

Verification: analytic axis/diagonal examples; reciprocal-lattice checks at fresh
angles with anisotropic scale, rotation and shear; unchanged e-SFR on fixed input;
missing/invalid metadata and unusable edge cases; CLI and focus UI integration.
No physical phone/camera is available, so actual aperture/ISP performance and
device layout identification cannot be validated here.

Baseline: e95af4f, bundled Python 3.12, OpenCV from standalone release runtime.
`unittest tests.test_hik_rig_calibration tests.test_rig_calibration`: 118 tests,
117 pass; pre-existing unrelated `test_ground_truth_feature_metrics_and_mma`
fails (repeatability 0.7397899649941657, expected >0.90). Do not change that metric.

## Model and provenance

Spatial frequency vectors transform as `q = J.T @ n`, where J maps raw camera
pixels to display pixels locally, and n is the unit display edge normal. Keeping
both components preserves CFA orientation under rotation, shear and perspective.

For vector v, a unit square lattice reaches Nyquist at `0.5/max(abs(v))`;
the checkerboard lattice at `0.5/sum(abs(v))`. Bayer green uses the checkerboard;
red/blue use a square lattice of pitch 2, hence `0.25/max(abs(q))`.
The Diamond PenTile model uses one green site per nominal pixel on a square
lattice and half-density red/blue checkerboards. It does not represent every
PenTile family or proprietary subpixel renderer. RGB stripe has one site of
each color per nominal pixel; subpixel aperture widths are not sample pitches.
The combined independent-channel cutoff is the minimum of panel and camera.
With the existing Rec.709 luminance weights, green alone exceeds 50%, so the
ideal white-edge luminance MTF50 in this model is the green cutoff. Report R/B
separately, and keep the existing 0.500 cy/display-pixel measurement range distinct.

Primary sources:

- [Kodak / Palum, Image Sampling with the Bayer Color Filter Array](https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2001/PICS-0-251/4631.pdf): channel-specific directional Nyquist domains and interpolation.
- [Samsung Display, Diamond Pixel](https://global.samsungdisplay.com/31139?category=68&page=1&type=list): stripe versus diamond subpixel structure and greater green density.
- [Imatest, Sharpening](https://www.imatest.com/docs/sharpening.html): processing changes measured MTF50; sampling alone is insufficient.

## Implementation and verification

Decision: land the diagnostic model; do not claim an absolute hardware MTF50
maximum. The pure model owns channel lattices; the existing e-SFR geometry helper
supplies its local Jacobian independently of image contrast. The focus view
shows the panel, camera and combined reference by default; T restores pose and
physical-scale details. Movement/save-blocked status remains on both views.
The panel grows vertically to fit readable text within the desktop work area.

`--focus-panel-layout` accepts unknown (default), rgb-stripe or diamond-pentile.
`--focus-camera-sampling` accepts auto (default), bayer or full-grid. Auto detects
the four PFNC Bayer8 numeric codes and named standard Bayer formats. Other numeric
formats and RGB/BGR/Mono transport formats remain unknown: transport format alone
does not prove the physical sensor layout. Full-grid is a spatial sampling model
for mono or co-sited RGB, not a claim that a monochrome sensor resolves color.
Selected layouts and per-edge reference are retained once per focus-loop invocation
in focus history, avoiding repeated model tables in every captured-frame record.

| Check | Result |
| --- | --- |
| Axis / diagonal lattices, magnification, rotated camera | Analytic expectations pass |
| 24 fresh angles with anisotropy and shear | Matches independently enumerated reciprocal-lattice boundaries |
| Bayer green sampled sinusoids | Explicit pair aliases beyond the diagonal green boundary |
| Projective geometry / ROI-origin shift | Same local frequency mapping |
| Lens correction handoff | Independent affine distortion stand-in gives expected sampling scale |
| Unusable flat image | Measured MTF50 unavailable; theoretical reference still available |
| Frozen e95af4f e-SFR versus candidate | Entire result and evidence image exactly equal; MTF50 0.2885547333420806 |
| Theory / pose toggle / save integration | Pass; model recorded once |
| 1920x1080 and 1280x720 desktop previews | Reference text visible at native panel font size |
| Four-edge theory timing, 100 warm iterations | Median 0.733 ms; p95 2.551 ms (local, no hardware/ISP claim) |

Final focused command:

```powershell
.tools/standalone-release-py31210/Scripts/python.exe -B -m unittest tests.test_focus_sampling tests.test_hik_rig_calibration tests.test_rig_calibration.RigImageQualityTests.test_display_referred_esfr_uses_prewarp_oversampled_camera_pixels tests.test_rig_calibrator_gui tests.test_rig_presentation tests.test_standalone_release_export
```

128 tests: 125 passed, 3 Qt GUI checks skipped because their optional dependencies
are absent. The HIK/OpenCV focus UI integration ran with mocked acquisition/window
calls and real chart measurement/rendering. CLI help and diff whitespace checks
also pass. The separate baseline feature-matching failure above remains unmodified.
No physical rig, panel aperture measurement or hardware demosaic validation ran.

Local evidence: `artifacts/focus-sampling/verification.json`,
`artifacts/focus-sampling/focus-1280.jpg`, and
`artifacts/focus-sampling/focus-1920.jpg`. Samsung's downloaded microscope diagram
was inspected at `artifacts/focus-sampling/samsung-diamond.png`; it is reference
material, not a newly generated calibration input or a redistributed source file.
