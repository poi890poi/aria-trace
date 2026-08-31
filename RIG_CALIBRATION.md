# AriaTrace Camera-to-Phone Rig Calibration

The standalone HIK MVS camera/Android-display workflow and rectified stream are
documented in [`aria_trace/services/calibration/rig/hik/README.md`](aria_trace/services/calibration/rig/hik/README.md).

## 1. Purpose

This document specifies how AriaTrace calibrates a fixed USB camera that observes an Android phone screen. The process must work when the camera sees the entire display, only the task-relevant part of the display, or additional area outside the display.

The calibration has five outputs:

1. A geometric measurement of the camera view relative to the phone screen.
2. Live guidance for positioning, rotation, distance, exposure, and focus.
3. Standard ISO/IEC 15415 Data Matrix Decode grades at declared logical
   display-pixel module widths.
4. An end-to-end control-to-perception latency distribution measured with alternating visual signals.
5. A small, commented YAML contract that other agents can use to normalize frames without understanding the calibration procedure.

The immediate task is reliable mini-map translation and cursor-pose estimation. Maximizing full-screen coverage is secondary to preserving the complete mini-map, a safety margin around it, and sufficient matchable detail.

## 2. Consumer Contract First

Calibration complexity stays behind the capture boundary. Lens undistortion is performed by the calibrated UVC capture layer. Downstream agents receive an image in the declared `camera_undistorted_px` input space and normalize it with one 3x3 projective matrix.

For an undistorted input pixel `(x, y)`:

```text
[u' v' w']^T = matrix_3x3 * [x y 1]^T
u = u' / w'
v = v' / w'
```

`(u, v)` is an output-image pixel. Its canonical phone-screen coordinate is:

```text
screen_x = origin_screen_xy[0] + u * screen_units_per_output_pixel_xy[0]
screen_y = origin_screen_xy[1] + v * screen_units_per_output_pixel_xy[1]
```

The canonical origin is the centre of the phone screen's top-left pixel. Integer coordinates address pixel centres, positive X points right, and positive Y points down; the continuous raster extent is `[-0.5, width-0.5] x [-0.5, height-0.5]`. The preferred full-screen normalization uses origin `[0, 0]` and scale `[1, 1]`, so output pixels are directly comparable with ADB screenshot pixels.

A consumer therefore needs only:

- `normalization.input_space`
- `normalization.matrix_3x3`
- `normalization.output_size_px`
- `normalization.origin_screen_xy`
- `normalization.screen_units_per_output_pixel_xy`
- `normalization.valid_mask_file`

Frame IDs in the same block let an optional external spatial-registry adapter publish the relationship without adding any registry dependency to rig calibration.

OpenCV consumption is intentionally simple:

```python
normalized = cv2.warpPerspective(
    undistorted_camera_frame,
    np.asarray(calibration["normalization"]["matrix_3x3"], dtype=np.float64),
    tuple(calibration["normalization"]["output_size_px"]),
)
```

The consumer must reject an input whose resolution, orientation, or `input_space` does not match the YAML. An agent opening raw UVC frames directly must first use the declared lens model or precomputed remap; it must not apply the projective matrix to distorted pixels and silently claim canonical geometry.

## 3. Coordinate Spaces

The process uses these named spaces:

- `camera_raw_px`: pixels delivered by the selected UVC mode.
- `camera_undistorted_px`: the same raster extent after the capture layer applies lens rectification.
- `normalized_output_px`: the raster produced by the saved 3x3 matrix.
- `phone_screen_px`: canonical phone coordinates matching the declared ADB screenshot orientation.

Every image, point set, polygon, and matrix in evidence must declare its space. Matrix direction must never be inferred from a filename.

When only a cropped part of the phone is normalized, `origin_screen_xy` records the canonical phone coordinate represented by output pixel `(0, 0)`. `screen_units_per_output_pixel_xy` records scaling explicitly. This allows a close mini-map-only view to remain interoperable with full-screen ADB references.

## 4. Inputs and Defaults

The wizard attempts to discover:

- UVC device identity, actual frame size, frame rate, FourCC, and supported optical controls.
- Phone serial, model, orientation, logical screenshot dimensions, and reported density through ADB.
- Physical display dimensions from a known device description or operator measurement.
- An optional required-region polygon, label, guard band, and acceptance policy supplied by an external adapter or drawn by the operator.

Missing information must not block relative optimization. Initial defaults are:

- 1920x1080 at 30 FPS when the camera offers that mode.
- Phone brightness at approximately 70%, with adaptive brightness disabled.
- A nominal 20 mm square resolving-power patch.
- The full screen as the required ROI when no game ROI is known.
- A 10% mini-map-diameter guard band when a mini-map ROI is known.

Values are tagged `measured`, `reported`, `manual`, `assumed`, or `unknown`. Physical distance and millimetre-based resolution remain nominal when physical screen scale is assumed, but ratios and before/after comparisons remain valid.

The calibration core treats the required region as geometry plus an opaque caller label. It does not import a game profile, mini-map extractor, map, route, controller, or dataset component. An adapter may label the region `minimap`, but that label does not change calibration algorithms.

The UI should offer an optional on-phone ruler. The user adjusts it to a physical ruler once, replacing the assumed scale without invalidating pixel geometry.

## 5. Process Flow

```text
Discover camera and phone
        |
        v
Collect intrinsic-calibration poses, or choose homography-only mode
        |
        v
Lock the phone and camera in their final bracket positions
        |
        v
Detect the full-screen ChArUco coordinate system from a frame burst
        |
        v
Measure screen coverage, camera utilization, IoU, pose, and uncertainty
        |
        v
Optimize required-ROI framing and perspective
        |
        v
Inspect raw pixels at 1:1 and grade alternating fixed-patch Data Matrix targets
        |
        v
Measure control-to-perception latency with alternating signals
        |
        v
Run held-out static and motion validation
        |
        v
Save calibration.yaml, remap data, masks, metrics, and review images
```

### 5.1 Camera and phone discovery

The wizard opens the requested UVC mode, then records the actual mode returned by the device. It samples focus, autofocus, exposure, auto-exposure, gain, white balance, and power-line frequency when the driver exposes them. Unsupported controls remain `unknown` rather than being filled with invented values.

The default HIK workflow serves an Aria-owned exact-pixel target surface on a
loopback port forwarded with ADB reverse. It reports target revision, drawing
surface, natural target raster, fullscreen state, and host-received paint time.
The former Android Gallery presenter remains available only as the explicit
`legacy_gallery` compatibility option. The phone displays targets full-screen
with system bars hidden and fixed orientation. The captured phone resolution
and orientation become part of calibration applicability.

### 5.2 Intrinsic calibration

A single fixed view of a planar display supports a screen homography but does not robustly determine full camera intrinsics. The preferred flow temporarily moves or tilts the phone through 10 to 15 ChArUco views before the bracket is locked:

- near and far;
- left, centre, and right image regions;
- positive and negative pitch;
- positive and negative yaw;
- corners reaching the outer camera-image regions.

The system estimates the camera matrix, radial/tangential distortion, per-view reprojection error, and uncertainty. It rejects blurry views, poorly distributed corners, and redundant poses.

If trusted intrinsics already exist for the exact camera mode, they may be reused. If the user skips this stage, the wizard continues in `homography_only` mode and lowers confidence near image edges. A projective matrix must not be represented as having corrected unmeasured radial distortion.

### 5.3 Final ChArUco screen registration

After the rig is locked, the phone displays a ChArUco board whose marker IDs and margins locate every board point in canonical phone-screen coordinates. The layout is generated for the phone aspect ratio so that useful identified corners remain visible under partial cropping.

The board is treated as a coordinate atlas, not merely as a picture whose
outer boundary must be visible. Detected IDs locate the camera viewport within
the complete logical display even when the camera sees only a partial screen.
The wizard must fit this atlas and report screen coverage, camera utilization,
screen-view IoU, required-ROI coverage, and extrapolation risk before it may
present any e-SFR or feature target. Quality trials use only a conservative
patch wholly inside the camera-visible intersection with the required ROI.

The system captures a burst and:

1. Undistorts each accepted frame.
2. Detects marker IDs and interpolated ChArUco corners.
3. Fits the camera-to-screen homography with robust outlier rejection.
4. Aggregates accepted frames rather than trusting one detection.
5. Bootstraps detections across frames to estimate transform and boundary uncertainty.
6. Projects the complete screen rectangle, including inferred portions outside the camera frame.

Four non-collinear points are a mathematical minimum, not an operational quality target. Acceptance should normally require at least 12 well-distributed ChArUco corners and sufficient corner-hull coverage around the required ROI.

## 6. Geometry Measurements

Let `S` be the complete phone-screen polygon and `V` be the camera viewport projected onto the phone-screen plane.

### Screen coverage

```text
screen_coverage = area(S intersection V) / area(S)
```

This answers: "How much of the phone screen can the camera see?"

### Camera utilization

```text
camera_utilization = visible phone-screen pixels / camera-frame pixels
```

This answers: "How much of the camera image is useful phone screen?"

### Screen-view IoU

```text
screen_view_iou = area(S intersection V) / area(S union V)
```

Both polygons are evaluated in normalized phone-plane coordinates. IoU penalizes both cropping and wasted surrounding view, but it is a diagnostic rather than the sole optimization objective.

### Task-region coverage

The required mini-map ROI and its guard band are evaluated independently:

- mini-map coverage;
- guard-band coverage;
- minimum distance from required ROI to every camera edge;
- detected-corner convex-hull coverage;
- transform uncertainty inside the ROI;
- local camera pixels per canonical screen pixel;
- perspective scale ratio across the ROI.

A partial-screen view may be excellent for AriaTrace even when full-screen IoU is low. The preferred pose is the closest view that keeps 100% of the required ROI and guard band while satisfying matchability and uncertainty limits.

The target presenter must report its physical drawing-buffer dimensions and a
paint acknowledgement for every controlled target revision. The dimensions
must match the declared canonical display raster. A quality observation is
eligible only when its host-receive timestamp is later than the acknowledgement
for the exact painted revision; a fixed sleep is not evidence that the intended
target reached the camera.

### Pose and positioning

When physical scale and intrinsics are available, report camera-to-screen distance and XYZ displacement. Always report:

- screen roll;
- pitch and yaw of the optical axis relative to the screen normal;
- horizontal and vertical ROI displacement;
- left/right and top/bottom scale imbalance;
- suggested correction in physical units when trustworthy and screen-relative units otherwise.

Example guidance:

- `Move the camera 8% closer.`
- `Shift right by 4% of screen width.`
- `Rotate clockwise 1.6 degrees.`
- `Tilt the top edge away by 3.2 degrees.`
- `Mini-map is visible, but the left guard band is only 4%; target is 10%.`

## 7. 1:1 Image-Quality and Focus Inspector

The geometry preview necessarily scales the whole camera image to fit the page and therefore cannot be used to judge focus. The calibration UI must include a separate magnified inspector centred on the mini-map ROI or the active quality patch.

Its default mode is **sample 1:1**:

- one source camera sample maps to one canvas sample;
- no bilinear, bicubic, browser smoothing, or fit-to-window resampling is permitted;
- the panel scrolls or crops rather than shrinking the image;
- its label always names the source space and zoom, for example `RAW CAMERA - 1:1 - NO INTERPOLATION`;
- a nearest-neighbour 2x, 4x, and 8x mode may be offered for Bayer, display-grid, ringing, and moire inspection, but 1:1 remains the reference.

The inspector provides:

- raw-camera and undistorted-camera tabs;
- current, frozen-best, side-by-side, and blink comparison modes;
- an ROI reticle and canonical mini-map outline;
- clipping indicators and luminance histogram;
- current and best ISO 12233 e-SFR/MTF result at the inspected ROI;
- focus/exposure stability over the recent burst;
- a warning when browser or canvas scaling would make the view non-1:1.

The raw tab is authoritative for optical focus because rectification interpolation can hide or introduce apparent sharpness. The undistorted tab verifies the actual pixels consumed by normalization.

If manual focus is used, every settled adjustment becomes one sample on a focus sweep. The UI shows `better`, `worse`, or `unchanged within uncertainty` relative to the best retained sample. If UVC focus control is writable, the wizard may sweep supported values automatically, select the best reviewed e-SFR/MTF result, and then lock focus. Task matching is evaluated separately after optical focus is fixed.

## 8. Standard Data Matrix Decode Grade

The one operator-facing feature-readability measurement is the **Decode
parameter from ISO/IEC 15415:2024**, applied to an **ISO/IEC 16022:2024 Data
Matrix**. Its globally defined result is deliberately simple:

- `4.0 / A`: the symbol is readable by the reference decode procedure and its
  codewords are valid;
- `0.0 / F`: reference decode fails or the encodation is invalid.

There is no project-defined percentage threshold, weighted score, average
grade, or renamed resolution unit. The displayed Data Matrix module width is
recorded as a test condition in logical display pixels per module
(`dpx/module`), not folded into the grade. The report therefore says, for
example, `ISO/IEC 15415 Decode grade 4/A at 1 dpx/module`.

The implementation is explicitly the ISO Decode parameter only. It must not
claim to be a complete ISO/IEC 15415 symbol verifier or emit an ISO overall
symbol grade; complete verification additionally requires standardized image
formation/calibration and grading of contrast, modulation, fixed-pattern
damage, nonuniformity, unused error correction, and other parameters.

The former `MR95-20` name and calculation were project-defined. They are
deprecated and must not appear as an accepted calibration metric, quality
gate, UI headline, or newly generated YAML field. Existing artifacts retain
their original value only as explicitly labeled legacy data; they must not be
silently translated into MTF, repeatability, matching score, or MMA.

### 8.1 Alternating fixed-patch procedure

Use short, unique, fixed-length payloads that all produce the same square Data
Matrix dimensions. Render each module as an exact integer square of logical
display pixels with no interpolation and retain at least the symbology-required
quiet zone. Only one symbol is shown at a time.

Successive targets reuse the same camera-visible screen patch:

```text
fixed visible patch
      |
      +-- payload A, 1 dpx/module --> capture --> Decode grade 4/A or 0/F
      +-- payload B, 1 dpx/module --> capture --> Decode grade 4/A or 0/F
      +-- payload C, 2 dpx/module --> capture --> Decode grade 4/A or 0/F
      +-- payload D, 2 dpx/module --> capture --> Decode grade 4/A or 0/F
      +-- ...
```

Temporal alternation lets a small display or partial camera/display
intersection test many payloads and module widths without fitting them all at
once. Use the smallest square symbol version that holds the trial token. If a
particular symbol/module combination cannot fit the fixed patch, record it as
`untestable`; it is not a Decode grade 0/F.

The payload is also synchronization evidence. A capture is eligible only
after the paint acknowledgement for that target revision and configured
settling. If a different valid payload is decoded, that observed symbol has a
Decode grade of 4/A but the image is marked `wrong_target_or_stale_frame` and
excluded from the requested target's evidence. A burst of correlated frames
from one unchanged presentation must not be represented as independent tests.

The review UI shows one row per observation and groups—not averages—the
standard grades by module width:

```text
Module width      ISO/IEC 15415 Decode observations
1 dpx/module      A, A, F, A, F
2 dpx/module      A, A, A, A, A
4 dpx/module      A, A, A, A, A
```

No automatic boundary is inferred between those rows. If an integrating
application needs a pass policy, that policy belongs to the application and
must remain visibly separate from the ISO grade.

The measurement library is image-in/grade-out and keeps its encoder and
decoder replaceable:

```python
from aria_trace.services.calibration.rig import grade_data_matrix_decode

result = grade_data_matrix_decode(camera_patch, expected_payload="A7K2")
print(result["grade"], result["grade_letter"])  # 4.0 A, or 0.0 F
```

The built-in adapter uses ZXing-C++ for Data Matrix creation and decoding.
ZXing-C++ is not represented as a certified ISO verifier; the saved result is
labeled `Decode_parameter_only_not_complete_ISO_IEC_15415_verifier`.

Finder patterns and error correction mean that Decode grade demonstrates
recovery of structured information, not independent localization of every
one-pixel game feature. Actual mini-map position and pose error remain a
separate downstream validation.

### 8.2 Supporting ISO 12233 e-SFR and MTF

Physical imaging resolution is measured using the slanted-edge electronic
spatial frequency response procedure from **ISO 12233:2024, Digital cameras —
Resolution and spatial frequency responses**. The authoritative result is the
normalized e-SFR curve. The following conventional crossing-frequency
summaries are derived from that curve and labeled as derived values:

- `MTF50`: spatial frequency where normalized response first reaches 0.50;
- `MTF10`: spatial frequency where normalized response first reaches 0.10.

The display and camera are two different sampled grids, so every frequency
must name its grid. Do not use the unqualified term `lines/pixel`, which is
ambiguous by a factor of two. One spatial cycle (one line pair) contains one
dark and one bright line width. Use:

- `cycles_per_display_pixel` (`cy/dpx`) for a target rendered on the phone and
  for the primary display-referred system e-SFR/MTF result;
- `cycles_per_camera_pixel` (`cy/cpx`) for the native camera-analysis curve and
  reproducibility diagnostics;
- `cycles_per_mm_on_phone_plane` (`cy/mm`, equivalently `lp/mm`) only when the
  physical display scale is measured.

A raster whose dark and bright lines are each one logical display pixel wide
has a two-display-pixel period and is therefore `0.5 cy/dpx`, the display-grid
Nyquist limit. It may be described as one line width per display pixel only
when that convention is stated explicitly; it must not be recorded simply as
`1 line/pixel`. The one-pixel raster is an upper-end stress condition, not a
claim that the camera resolves that frequency.

The authoritative samples are taken from the unwarped camera analysis raster,
but the headline curve and MTF crossings are transformed back and reported in
`cy/dpx`. This measures the finest logical display features preserved by the
complete display-to-camera chain, which is the rig's task-relevant quantity.
The native `cy/cpx` curve is retained as supporting evidence, not used as the
primary resolving-power score. A homography-normalized phone image is not the
measurement input because interpolation changes its spatial frequency
response; normalized images remain appropriate for downstream matching and
pose tests.

In the locally axis-aligned, isotropic case, if the sampling ratio is `s`
camera pixels per display pixel, `f_c = f_d / s` and therefore
`f_d = s * f_c`. In the general projective case, transform the oriented
spatial-frequency vector with the transpose of the local camera-to-display
homography Jacobian and retain the ROI position and direction used. This makes
camera oversampling beneficial without allowing high camera pixel density to
inflate the display-referred result. If the measured display pitch is
`p_d mm/dpx`, a display-domain frequency converts as
`f_mm = f_d / p_d cy/mm`.

Preserve both frequency axes, the local conversion/Jacobian, full curve, edge
image, edge angle, linearization/OECF procedure, sampling frequency, ROI,
channel, and crossing method so the summaries remain auditable. If physical
display pitch is assumed rather than measured, retain `cy/dpx` and `cy/cpx` and
omit the absolute `cy/mm` result rather than presenting a nominal conversion as
measured.

The target suite includes a frequency sweep below and through the practical
cutoff, including but not relying on the `0.5 cy/dpx` one-pixel endpoint. It
also includes multiple edge/pattern orientations, screen positions, luminance
levels, contrast levels, and luminance/colour channels. Multiple subpixel
phases are required because logical display pixels, physical phone subpixels,
the camera Bayer grid, demosaicing, sharpening, and resampling can produce
phase-dependent aliasing and moire. Results are reported per condition plus a
declared conservative aggregate; one favorable edge or one-pixel raster cannot
represent the complete mini-map ROI.

Laplacian variance and Tenengrad may remain responsive focus aids, but they are
relative diagnostics. e-SFR/MTF remains supporting engineering evidence when
Decode grades contradict apparent focus; it is not a second operator-facing
feature-readability grade.

### 8.3 Task-validation feature-matching measurements

Planar target pairs use the known camera-to-screen homography as ground truth.
Report established homography-ground-truth measurements without inventing a
combined name. The baseline implementation uses keypoint centres rather than
affine-region overlap and labels that distinction explicitly:

- **Point repeatability at threshold `t`:** mutually nearest detected keypoint
  centres within `t` display pixels under the ground-truth homography, divided
  by the smaller detected-keypoint count in the common visible area. This is
  not mislabeled as Oxford affine-region repeatability.
- **Matching score:** correct descriptor matches divided by the smaller
  detected-region count within the common visible area.
- **Mean Matching Accuracy at threshold `t`, `MMA@t`:** fraction of evaluated
  matches whose ground-truth reprojection error is no greater than `t`
  normalized-screen pixels, averaged over image pairs.

Always report `MMA@1px`, `MMA@2px`, `MMA@3px`, and the complete `MMA@1..10px`
curve, together with detected-feature count, evaluated-match count,
correct-match count, spatial coverage, and failure count. MMA alone can look
excellent when a matcher returns very few correspondences, so counts,
repeatability, and matching score are mandatory companions.

The system-level result is the downstream homography, translation, rotation,
or pose error produced from those correspondences. Report median, P95, and the
accuracy/AUC curve at declared application thresholds. Matcher confidence or
correlation response is diagnostic evidence only; it is not ground truth.

### 8.4 References and conditions

Measure reference modes separately:

- `generated_to_camera`: the exact generated target is matched directly to the
  pre-warp physical-camera observation;
- `adb_to_camera`: an ADB screenshot is matched directly to the pre-warp
  physical-camera observation;
- `camera_to_camera`: a retained physical-camera reference is matched to a
  later physical-camera observation.

Use held-out trials spanning real mini-map content and controlled targets,
static and moving capture, luminance and colour, orientation, subpixel phase,
focus, exposure, phone brightness, and expected rig variation. Detail size in
display pixels, `cy/dpx`, `cy/cpx`, or measured `cy/mm` is a trial condition—not
a new metric. Report every condition or a declared worst-case aggregate so a
single headline number cannot hide a systematic failure.

### 8.5 Ground truth and acceptance

A correct match is determined by homography reprojection error, not by the
matcher accepting itself. A pose trial is accepted only when the downstream
translation and rotation errors satisfy caller-declared mini-map requirements.
Report catastrophic false matches separately from rejected/no-result trials.

Calibration acceptance uses all of the following:

- reviewed ISO/IEC 15415 Decode grades at every declared Data Matrix module
  width, with wrong/stale targets excluded rather than counted as failures;
- reviewed ISO 12233 e-SFR curve and derived MTF50/MTF10 for the required ROI;
- repeatability and matching score;
- MMA curve at declared normalized-screen-pixel thresholds;
- sufficient correct-match count and spatial coverage;
- downstream translation, rotation, or pose-error distributions;
- results stratified by reference mode and capture condition;
- confidence intervals with the sampling unit and trial count stated.

The Decode grade is standard and has no project threshold. Any application
policy across repeated grades, plus task matching and pose thresholds, comes
from the mini-map estimator's permitted position and direction error and must
be stored separately from the ISO measurement.

### 8.6 Control-to-perception latency

Rig calibration also measures the delay from a timestamped control request to the first observation that reliably contains the requested visual state. The physical-camera endpoint is required for the rig result; an ADB/screenshot observation endpoint and device presentation endpoint are retained when available. These are generic display/capture-pipeline baselines and must not be mislabeled as a particular game's complete input-response latency.

The calibration core knows only two small interfaces:

```text
AlternatingStimulus.set_state(state, token) -> timestamped control event
SignalObserver.observe(frame) -> state probabilities and observation timestamp
```

An adapter may implement the stimulus through a calibration app, ADB, touch injection, another control backend, or a test display. The calibration algorithm does not import or understand any of those components.

The recommended target alternates between two complementary, equal-average-luminance patterns at the required ROI. Complementary patterns produce a strong signed correlation while reducing auto-exposure bias. States alternate, but dwell lengths follow a known pseudorandom schedule so periodic phase aliases cannot produce a convincing latency estimate.

For every transition, retain:

- control token and requested state;
- control issue, adapter acceptance, and device presentation timestamps when available;
- source and host timestamps for every candidate camera frame;
- observed state probabilities;
- first threshold crossing and first stable-state frame;
- rising/falling transition type;
- rejection reason for missed, ambiguous, or partially exposed transitions.

The primary reported end-to-end value is:

```text
control_to_camera_perception_latency
    = first_stable_observation_time - control_issue_time
```

Clock mapping and causal latency are separate models. A clock offset or drift converts timestamps into a common clock; it must not absorb display, exposure, transfer, or processing delay. When all events use the same host monotonic clock, the clock transform is identity while latency remains non-zero.

When ADB and camera observations are both available, measure `control_to_adb_perception` and `control_to_camera_perception` from the same accepted transition tokens. A paired incremental ADB-to-camera delay may then be reported with its own uncertainty. Do not subtract unrelated aggregate medians or compare endpoints before mapping their timestamps into a common clock.

Use at least 64 accepted transitions when practical. Report:

- accepted, missed, and ambiguous transition counts;
- median, P05, P95, and maximum latency;
- robust jitter, such as half the P95-P05 interval;
- rising-versus-falling bias;
- frame-interval quantization and timestamp uncertainty;
- camera receive delay when both capture and receive timestamps exist;
- row-dependent transition time when rolling-shutter evidence is visible;
- latency stability over the run;
- cross-correlation lag as a global consistency check.

Sub-frame interpolation may be estimated from a partially transitioned exposure, but the un-interpolated frame-bound result must remain available. The YAML declares whether the latency endpoint is `first_detected`, `first_stable`, or another explicit criterion.

Evidence includes a control/observation timeline, transition-latency distribution, cross-correlation curve, representative alternating frames, and rejected-transition examples. The 1:1 inspector can follow the stimulus ROI so the operator can see blur, rolling bands, or ambiguous state mixtures directly.

## 9. Calibration UI

The workbench flow contains four pages or progressive panels.

### Camera setup

- Camera mode and actual returned mode.
- Phone resolution, orientation, and scale source.
- Optical controls and whether each value can be locked.
- Assumption badges with an edit action.

### Geometry

- Live frame with detected corners.
- Detected ChArUco corner support and the projected complete screen boundary.
- Full-plane inset comparing the complete screen, camera viewport, and selected
  camera-visible quality patch when boundaries extend outside the raw frame.
- Required mini-map ROI and guard band.
- Green safe area, amber extrapolated area, and red cropped required area.
- Coverage, utilization, IoU, roll, pitch/yaw, and positioning guidance.

### Focus, Decode grade, timing, and task validation

- The 1:1 inspector described above.
- ISO/IEC 15415 Decode grade `4/A` or `0/F` for the current Data Matrix,
  grouped as unaveraged observations at each declared `dpx/module` condition.
- Fixed-patch target revision, expected/decoded payload, and an explicit stale
  target warning when they differ.
- Current versus best ISO 12233 e-SFR curve and derived MTF50/MTF10.
- Repeatability, matching score, MMA curves, match counts, spatial coverage,
  and reviewed correct/incorrect match examples.
- Exposure, clipping, motion-blur, and temporal-stability warnings.
- Alternating-signal state, accepted-transition count, latency distribution, and command/observation timeline.

### Review and save

- Consumer normalization contract at the top.
- Geometry and task validation summary.
- Every measured and assumed input.
- Confidence and applicability conditions.
- Links to diagnostic images.
- `Save and activate` and `Save with warning` actions.

The user may retain a marginal calibration for experiments. Its status must be `warning` or `failed_task_requirement`; it must not silently become an accepted runtime profile.

## 10. Artifact Layout

```text
artifacts/rig_calibrations/<rig-id>/<calibration-id>/
|-- calibration.yaml
|-- rectification_maps.npz
|-- valid_screen_mask.png
|-- charuco_detection.png
|-- screen_overlap.png
|-- focus_best_raw_1x.png
|-- focus_best_undistorted_1x.png
|-- data_matrix_decode_observations.png
|-- slanted_edge_roi.png
|-- esfr_mtf_curve.png
|-- mma_curve.png
|-- match_examples.png
|-- latency_timeline.png
|-- latency_distribution.png
|-- latency_correlation.png
`-- evidence/
```

`calibration.yaml` is the authoritative, human-readable contract. Large dense arrays such as precomputed undistortion maps remain referenced binary assets. YAML comments explain conventions and fields for both humans and agents. Writers should preserve the leading contract comments when updating a reviewed file.

## 11. Commented YAML Contract

```yaml
# AriaTrace camera-to-phone rig calibration.
# Consumer fast path:
#   1. Obtain a frame in normalization.input_space from the calibrated source.
#   2. warpPerspective(frame, normalization.matrix_3x3,
#                      normalization.output_size_px).
#   3. Interpret output (0, 0) as normalization.origin_screen_xy and apply
#      normalization.screen_units_per_output_pixel_xy to recover phone coordinates.
schema_version: "1.0"
calibration_id: "rig-logitech-c920-pixel7-20260826T120000Z"
status: accepted              # accepted | warning | failed_task_requirement
created_utc: "2026-08-26T12:00:00Z"

rig:
  rig_id: "logitech-c920-pixel7"
  camera:
    device_id: 0
    hardware_id: "usb-vid_pid_serial-if-available"
    width_px: 1920             # Actual mode returned by the camera, not only requested mode.
    height_px: 1080
    fps: 30.0
    pixel_format: "MJPG"
  phone:
    adb_serial: "serial-if-available"
    model: "Pixel 7"
    orientation: portrait
    screen_size_px: [1080, 2400]
    physical_size_mm: [64.7, 143.8]
    physical_size_source: reported  # measured | reported | manual | assumed | unknown

optics:
  focus:
    mode: manual
    value: 118
  exposure:
    mode: manual
    value: -6
  gain: 32
  white_balance:
    mode: manual
    kelvin: 5000
  # The calibrated source applies this model before exposing camera_undistorted_px.
  lens_model:
    model: opencv_radtan
    camera_matrix_3x3:
      - [1500.0, 0.0, 959.5]
      - [0.0, 1502.0, 539.5]
      - [0.0, 0.0, 1.0]
    distortion_coefficients: [-0.12, 0.03, 0.0, 0.0, -0.01]
    precomputed_maps_file: "rectification_maps.npz"
    source: measured           # measured | reused | assumed | unavailable

normalization:
  # Downstream agents must use this exact input space and resolution.
  input_frame_id: "aria://rig/logitech-c920-pixel7/camera/undistorted"
  input_space: camera_undistorted_px
  input_size_px: [1920, 1080]
  input_origin: top_left_pixel_center
  input_axes: [right, down]

  # matrix_3x3 maps input pixels to output pixels.
  transform_direction: input_pixel_to_output_pixel
  matrix_3x3:
    - [1.234, 0.012, -322.1]
    - [-0.008, 1.229, -104.7]
    - [0.00001, -0.00002, 1.0]

  output_size_px: [1080, 2400]
  output_frame_id: "aria://artifact/rig-logitech-c920-pixel7-20260826T120000Z/normalized"
  output_origin: top_left_pixel_center
  output_axes: [right, down]

  # Canonical phone coordinate represented by output pixel (0, 0).
  canonical_screen_frame_id: "aria://device/pixel7/screen/portrait/layout-1080x2400"
  origin_screen_xy: [0.0, 0.0]
  # Canonical phone-screen units represented by one output pixel.
  screen_units_per_output_pixel_xy: [1.0, 1.0]
  valid_mask_file: "valid_screen_mask.png"
  border_mode: constant
  border_value_bgr: [0, 0, 0]

geometry:
  mode: intrinsics_and_homography  # or homography_only
  screen_polygon_input_xy:
    - [324.2, 102.1]
    - [1507.8, 111.4]
    - [1660.1, 1031.0]
    - [171.0, 1024.2]
  screen_coverage: 0.98
  camera_utilization: 0.71
  screen_view_iou: 0.70
  required_roi_coverage: 1.0
  guard_band_coverage: 1.0
  reprojection_rmse_px: 0.31
  transform_p95_error_px_at_required_roi: 0.58
  roll_deg: 0.4
  pitch_deg: -2.1
  yaw_deg: 1.7
  distance_mm: 248.0
  distance_source: measured
  charuco_atlas:
    fit_precedes_quality_measurement: true
    detected_corner_count: 54
    detected_marker_count: 38
    screen_view_iou: 0.70
  quality_region:
    status: available
    # Entirely inside the camera-visible required ROI and supported atlas hull.
    xywh: [26, 50, 294, 294]
    space: canonical_phone_screen_px
    requires_detected_atlas_hull_support: true

required_roi:
  # Canonical phone-screen coordinates; example values only.
  kind: minimap                 # Opaque caller label; the rig core does not interpret it.
  xywh: [18.0, 42.0, 310.0, 310.0]
  guard_band_px: 31.0
  source: game_profile

data_matrix_decode:
  standard: "ISO/IEC 15415:2024"
  symbology_standard: "ISO/IEC 16022:2024"
  parameter: Decode
  grade_scale: "4/A or 0/F"
  meaning: "4/A = reference decode succeeds; 0/F = reference decode fails"
  implementation: ZXing-C++ Data Matrix decoder
  implementation_conformance: Decode_parameter_only_not_complete_ISO_IEC_15415_verifier
  aggregation: none_standard_grades_reported_as_counts
  fixed_target_rect_screen_xywh: [26, 50, 128, 128]
  alternating_same_patch: true
  tested_module_widths_display_px: [1, 2, 4]
  module_width_results:
    - module_width_display_px: 1
      observation_count: 5
      eligible_observation_count: 5
      decode_grade_4_A_count: 3
      decode_grade_0_F_count: 2
      ineligible_wrong_target_count: 0
    - module_width_display_px: 2
      observation_count: 5
      eligible_observation_count: 5
      decode_grade_4_A_count: 5
      decode_grade_0_F_count: 0
      ineligible_wrong_target_count: 0
  # Each trial retains the standard grade, module width, expected and decoded
  # payloads, target revision/paint acknowledgement, and capture timestamp.
  trials: [] # abbreviated example; no mean or project pass threshold is stored

image_quality:
  standard: "ISO 12233:2024"
  method: slanted_edge_e_sfr
  implementation_conformance: non_certified
  measurement_input_space: camera_pre_homography_px
  # Never write an unqualified "lines/pixel": one cycle/line pair contains
  # one dark and one bright line width, so the term is factor-of-two ambiguous.
  display_target:
    spatial_frequency_unit: cycles_per_display_pixel
    # One-pixel dark + one-pixel bright lines have a two-pixel period.
    maximum_test_frequency: 0.5
    minimum_line_width_display_px: 1
    logical_pixel_definition: canonical_phone_screen_pixel
  # Samples come from the pre-warp camera raster. The frequency axis is then
  # transformed to display pixels; a warped image is not used to measure MTF.
  primary_spatial_frequency_unit: cycles_per_display_pixel
  native_analysis_frequency_unit: cycles_per_camera_pixel
  display_nyquist_cycles_per_display_pixel: 0.5
  condition_count: 16
  display_referred:
    aggregation: minimum_response_curve_across_declared_conditions
    spatial_frequency_unit: cycles_per_display_pixel
    mtf50_conservative: 0.184
    mtf10_conservative: 0.327
    frequency: [0.0, 0.002, 0.004, 0.006] # abbreviated example; artifact retains full curve
    mtf_conservative: [1.0, 0.998, 0.994, 0.989]
    mtf_median: [1.0, 0.999, 0.996, 0.992]
  conditions:
    - channel: luminance
      edge_angle_display_deg: 5.0
      phase_display_px: 0.0
      mtf50: 0.201
      mtf10: 0.351
      confidence: 0.93
  # Each retained measurement also contains its full cy/dpx curve, native
  # cy/cpx axis, local homography Jacobian, sample/bin support, OECF status,
  # target/capture timestamps, confidence components, and warnings.
  measurements: [] # abbreviated here
  failed_conditions: []

feature_matching:
  protocol: planar_homography_ground_truth
  threshold_space: canonical_display_px
  reference_modes: [generated_to_camera, adb_to_camera, camera_to_camera]
  primary_threshold_display_px: 3
  repeatability_by_threshold_px:
    1: 0.710
    2: 0.814
    3: 0.842
  matching_score_by_threshold_px:
    1: 0.662
    2: 0.748
    3: 0.781
  mma_by_threshold_px:
    1: 0.746
    2: 0.889
    3: 0.934
    4: 0.951
    5: 0.963
    6: 0.971
    7: 0.976
    8: 0.980
    9: 0.983
    10: 0.985
  denominator_feature_count: 614
  evaluated_match_count: 482
  correct_match_count_by_threshold_px: {1: 360, 2: 429, 3: 451}
  spatial_coverage_min: 0.72
  downstream_reprojection_p95_display_px: 0.84
  catastrophic_mismatch_rate: 0.004
  detector_descriptors: [SIFT]
  trials: [] # retains per-trial counts, match errors/examples, and downstream solve

timing:
  # Clock conversion and causal latency are intentionally separate.
  clocks:
    control_event: host_monotonic_ns
    camera_observation: host_monotonic_ns
  clock_transform:
    from_clock: host_monotonic_ns
    to_clock: host_monotonic_ns
    model: affine
    scale: 1.0
    offset_ns: 0
    uncertainty_ns: 0
  control_to_camera_perception:
    stimulus: complementary_binary_patch
    schedule: alternating_pseudorandom_dwell
    roi_screen_xywh: [18.0, 42.0, 310.0, 310.0]
    endpoint: first_stable
    issued_transitions: 72
    accepted_transitions: 68
    missed_transitions: 2
    ambiguous_transitions: 2
    median_ns: 81600000
    p05_ns: 65300000
    p95_ns: 112400000
    maximum_ns: 129100000
    robust_jitter_ns: 23550000  # Half of P95-P05.
    rising_falling_bias_ns: 3200000
    frame_interval_median_ns: 33333333
    timestamp_uncertainty_ns: 16666667
    source: measured
    scope: display_and_camera_pipeline_baseline
  control_to_adb_perception:    # Optional endpoint measured from the same transition tokens.
    endpoint: first_stable
    accepted_transitions: 68
    median_ns: 46200000
    p05_ns: 33100000
    p95_ns: 70500000
    source: measured
  paired_adb_to_camera_delay:
    median_ns: 35400000
    p05_ns: 21100000
    p95_ns: 52700000
    source: derived_from_paired_transitions
  evidence:
    timeline: "latency_timeline.png"
    distribution: "latency_distribution.png"
    correlation: "latency_correlation.png"

confidence:
  geometry: 0.96
  image_quality: 0.94
  feature_matching: 0.91
  timing: 0.93
  overall: 0.91
  assumptions: []
  warnings: []

applicability:
  # A mismatch requires validation or recalibration instead of silent reuse.
  require_same_camera_hardware_id: true
  require_same_camera_mode: true
  require_same_phone_orientation: true
  require_same_phone_screen_size_px: true
  require_locked_focus: true
  require_locked_exposure: true
  maximum_startup_reprojection_error_px: 1.5

evidence:
  charuco_detection: "charuco_detection.png"
  screen_overlap: "screen_overlap.png"
  focus_best_raw_1x: "focus_best_raw_1x.png"
  focus_best_undistorted_1x: "focus_best_undistorted_1x.png"
  data_matrix_decode_observations: "data_matrix_decode_observations.png"
  slanted_edge_roi: "slanted_edge_roi.png"
  esfr_mtf_curve: "esfr_mtf_curve.png"
  mma_curve: "mma_curve.png"
  match_examples: "match_examples.png"
```

Numeric values above illustrate the contract only and are not acceptance thresholds.

## 12. Runtime Use and Validation

The calibrated frame source:

1. Verifies camera identity and actual mode.
2. Applies the declared lens remap.
3. Publishes `camera_undistorted_px` with `calibration_id` in frame metadata.
4. Optionally performs the saved perspective warp itself for consumers that request canonical screen frames.

Other agents either request an already normalized frame or apply the YAML matrix. They do not refit screen geometry, infer matrix direction, or guess crop offsets.

The rig artifact may be registered in the spatial-unification graph described by `SPATIAL_UNIFICATION.md`, but the rig-calibration module does not depend on that registry. It exports self-describing frame IDs, a directed spatial transform, clock information, and latency measurements; an external adapter decides whether and where to register them.

At startup, a short validation target may verify reprojection and matching without replacing the saved calibration. Recalibration is required when the camera mode, physical mount, focus mode, phone orientation, logical screen size, or display scaling changes. Exposure or brightness changes may permit quick matchability revalidation when geometry remains stable.

## 13. Acceptance and Confidence

Geometry confidence includes:

- accepted-corner count and distribution;
- detected convex-hull coverage;
- robust reprojection residuals;
- burst-to-burst transform stability;
- extrapolation distance at the required ROI;
- intrinsic-calibration quality when available.

Image-quality and matching confidence include:

- Data Matrix Decode grade counts per declared module width, rejected stale
  target count, and exact target/capture provenance (without averaging grades);
- slanted-edge count, position, angle, channel, phase, and OECF provenance;
- e-SFR curve stability and derived MTF crossing uncertainty;
- held-out target count;
- static and moving trials;
- reference-mode agreement;
- colour/orientation worst-case performance;
- repeatability, matching score, MMA, correct-match count, and spatial coverage;
- confidence-interval method and width;
- temporal stability after focus and exposure are locked.

Timing confidence includes accepted transition count, state separability, missed/ambiguous rate, clock uncertainty, camera-frame quantization, rising/falling agreement, and latency stability over the run.

Initial task gates should require:

- 100% required mini-map coverage;
- 100% configured guard-band coverage;
- no required-ROI pixels outside the valid mask;
- transform uncertainty below the caller-supplied permitted position error;
- reviewed ISO/IEC 15415 Data Matrix Decode grades in the camera-visible
  required ROI, with any application policy stated separately;
- reviewed e-SFR/MTF performance throughout the required ROI;
- repeatability, matching score, MMA, correct-match count, and feature coverage
  above caller-supplied requirements;
- P95 translation and rotation errors below the caller-supplied estimator limits;
- catastrophic mismatch rate below the declared route-safety limit;
- enough accepted alternating transitions to characterize latency tails, with missed and ambiguous rates reported rather than hidden.

The project should derive numeric limits from recorded target-game evidence rather than treating the illustrative YAML values as universal thresholds.

## 14. Implementation Boundaries

The implementation should add:

- a rig-calibration service responsible for targets, corner fitting, geometry, quality trials, and YAML generation;
- a calibrated UVC source responsible for lens correction and calibration metadata;
- a workbench calibration UI with the geometry overlay and 1:1 inspector;
- a small YAML loader/validator shared by all consumers;
- evidence rendering and repeatable tests for matrix direction, origin,
  scaling, partial-screen geometry, alternating fixed-patch ISO/IEC 15415
  Decode grading, ISO 12233 e-SFR/MTF processing, and established
  feature-matching measurements;
- repeatable timing tests for alternating-signal detection, clock/latency separation, transition rejection, and tail statistics;
- a dependency-free spatial export adapter conforming to `SPATIAL_UNIFICATION.md`.

The core interfaces use generic frames, targets, required regions, stimuli, and observers. Game profiles, ADB commands, UVC drivers, UI controls, map artifacts, and dataset importers remain replaceable adapters outside the module.

Rig calibration remains separate from game-specific mini-map calibration. Rig calibration converts physical-camera pixels into canonical phone-screen observations. Mini-map calibration then finds and models the game UI within that normalized coordinate system.

## 15. Current implementation

The independent core is implemented under `aria_trace/services/calibration/rig/`; its API and dependency boundary are documented in `aria_trace/services/calibration/rig/README.md`. It includes exact-pixel Data Matrix target generation, alternating fixed-patch sequencing, image-in/Decode-grade-out evaluation, stale-payload rejection, and unaveraged sweep summaries. Synthetic verification is in `tests/test_rig_calibration.py`. The package includes spatial-fragment export but does not implement the external spatial registry/resolver.

The optional standalone Windows application is implemented under
`aria_trace/apps/rig_calibrator/`. It provides a PySide6 guided UI, opt-in
OpenCV camera capture, a fullscreen phone target service, exact-pixel review,
ChArUco-atlas IoU/coverage fitting, display-referred slanted-edge e-SFR/MTF,
homography-ground-truth feature matching, alternating-signal camera latency,
optional ADB reference capture, and reviewed bundle export. The Data Matrix
measurement library is implemented but is not yet wired into this GUI workflow.
Camera, ADB, and phone target
implementations are public replaceable adapters; hardware-specific controls
and alternative transports remain outside the calibration algorithms. The
PyInstaller build is isolated beneath `.tools/` and emits its distribution
beneath ignored `artifacts/` storage.

The public API, UI, and new YAML output no longer expose the former
project-defined resolving-power result. e-SFR output declares the implementation
`non_certified`, records whether measured OECF linearization was applied, and
retains per-condition failures. Calibration remains `warning` until the caller
provides task-specific acceptance thresholds. The previously packaged Windows
distribution predates this source revision and must be rebuilt before hardware
review; no camera, phone, ADB, or GUI was exercised while implementing it.

## 16. Normative and Benchmark References

- [ISO/IEC 15415:2024 — Bar code symbol print quality test specification — Two-dimensional symbols](https://www.iso.org/standard/76876.html)
- [ISO/IEC 16022:2024 — Data Matrix bar code symbology specification](https://www.iso.org/standard/80926.html)
- [ISO 12233:2024 — Digital cameras — Resolution and spatial frequency responses](https://www.iso.org/standard/88626.html)
- [Oxford VGG affine-feature detector repeatability protocol](https://www.robots.ox.ac.uk/~vgg/research/affine/evaluation.html)
- [Oxford VGG region-descriptor matching-score protocol](https://www.robots.ox.ac.uk/~vgg/research/affine/desc_evaluation.html)
- [HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors](https://openaccess.thecvf.com/content_cvpr_2017/html/Balntas_HPatches_A_Benchmark_CVPR_2017_paper.html)
- [Image Matching across Wide Baselines: From Paper to Practice](https://research.google/pubs/image-matching-across-wide-baselines-from-paper-to-practice/)
- [IEEE 2020-2024 — Standard for Automotive System Image Quality](https://standards.ieee.org/ieee/2020/11960/)
