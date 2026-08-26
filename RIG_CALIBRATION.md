# AriaTrace Camera-to-Phone Rig Calibration

## 1. Purpose

This document specifies how AriaTrace calibrates a fixed USB camera that observes an Android phone screen. The process must work when the camera sees the entire display, only the task-relevant part of the display, or additional area outside the display.

The calibration has four outputs:

1. A geometric measurement of the camera view relative to the phone screen.
2. Live guidance for positioning, rotation, distance, exposure, and focus.
3. An end-to-end resolving-power measurement based on image matching rather than nominal camera pixels.
4. A small, commented YAML contract that other agents can use to normalize frames without understanding the calibration procedure.

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

The canonical origin is the phone screen's top-left corner. Positive X points right and positive Y points down. The preferred full-screen normalization uses origin `[0, 0]` and scale `[1, 1]`, so output pixels are directly comparable with ADB screenshot pixels.

A consumer therefore needs only:

- `normalization.input_space`
- `normalization.matrix_3x3`
- `normalization.output_size_px`
- `normalization.origin_screen_xy`
- `normalization.screen_units_per_output_pixel_xy`
- `normalization.valid_mask_file`

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
- Expected mini-map ROI and guard band from the selected game profile.

Missing information must not block relative optimization. Initial defaults are:

- 1920x1080 at 30 FPS when the camera offers that mode.
- Phone brightness at approximately 70%, with adaptive brightness disabled.
- A nominal 20 mm square resolving-power patch.
- The full screen as the required ROI when no game ROI is known.
- A 10% mini-map-diameter guard band when a mini-map ROI is known.

Values are tagged `measured`, `reported`, `manual`, `assumed`, or `unknown`. Physical distance and millimetre-based resolution remain nominal when physical screen scale is assumed, but ratios and before/after comparisons remain valid.

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
Inspect raw pixels at 1:1 and optimize end-to-end matchability
        |
        v
Run held-out static and motion validation
        |
        v
Save calibration.yaml, remap data, masks, metrics, and review images
```

### 5.1 Camera and phone discovery

The wizard opens the requested UVC mode, then records the actual mode returned by the device. It samples focus, autofocus, exposure, auto-exposure, gain, white balance, and power-line frequency when the driver exposes them. Unsupported controls remain `unknown` rather than being filled with invented values.

The phone displays targets full-screen with system bars hidden and fixed orientation. The captured phone resolution and orientation become part of calibration applicability.

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
- current and best MR95-20 score;
- focus/exposure stability over the recent burst;
- a warning when browser or canvas scaling would make the view non-1:1.

The raw tab is authoritative for optical focus because rectification interpolation can hide or introduce apparent sharpness. The undistorted tab verifies the actual pixels consumed by normalization.

If manual focus is used, every settled adjustment becomes one sample on a focus sweep. The UI shows `better`, `worse`, or `unchanged within uncertainty` relative to the best retained sample. If UVC focus control is writable, the wizard may sweep supported values automatically, select the best held-out matchability result, and then lock focus.

## 8. Match-Based Resolving Power

Nominal camera pixels, Laplacian variance, and ordinary MTF do not measure whether AriaTrace can match a pre-captured reference to a live physical-camera frame. The primary global metric is **MR95-20: Matchable Resolution at 95% reliability over a 20 mm patch**.

MR95-20 is the largest number of independent band-limited detail cells that fit across a nominal 20 mm display patch while the production matcher recovers the correct translation and rotation in at least 95% of held-out trials.

Example interpretation:

```text
MR95-20: 64 cells across 20 mm
Smallest reliably matchable detail: 20 / 64 = 0.31 mm
Estimated matchable cells across a 28 mm mini-map: 90
```

Higher is better. When physical patch size is assumed, the metric is labeled `nominal`; its relative value remains useful for focus and distance optimization.

### 8.1 Targets

The phone displays seeded, band-limited targets at progressively finer scales. The suite includes:

- luminance textures;
- red/green and blue/yellow chromatic textures;
- multiple pattern orientations and phases;
- mini-map-like contours, paths, landmarks, and icon clutter;
- a directional cursor-like shape.

Using the complete capture-and-match path automatically includes phone pixel layout, camera Bayer pattern, lens blur, demosaicing, sharpening, noise reduction, moire, exposure, compression, and rectification effects.

### 8.2 Reference modes

Two modes are measured separately:

- `adb_to_camera`: an ADB screenshot or known rendered target is matched to a live camera frame.
- `camera_to_camera`: a retained physical-camera reference is matched to a later live camera frame.

The primary score is the conservative supported-mode result. Per-mode, luminance, chromatic, orientation, and motion scores remain visible so a single headline value cannot hide a systematic failure.

### 8.3 Trial definition

Targets receive known translations and rotations. A held-out trial succeeds only when:

- the matcher selects the correct correspondence;
- translation error is no more than 1% of patch width;
- rotation error is no more than 1 degree;
- the result passes the production matcher's ambiguity test, such as correlation-peak margin or geometric-inlier support.

Report:

- static and moving-target MR95-20;
- median and P95 translation error as a percentage of mini-map diameter;
- median and P95 rotation error in degrees;
- catastrophic mismatch rate;
- match-confidence separation;
- score stability across the capture burst;
- bootstrap 95% confidence interval.

Slanted-edge MTF50, Laplacian sharpness, signal level, clipping, and noise may be retained as diagnostic evidence. They do not determine task acceptance.

## 9. Calibration UI

The workbench flow contains four pages or progressive panels.

### Camera setup

- Camera mode and actual returned mode.
- Phone resolution, orientation, and scale source.
- Optical controls and whether each value can be locked.
- Assumption badges with an edit action.

### Geometry

- Live frame with detected corners.
- Solid detected screen boundary.
- Dashed inferred off-camera boundary.
- Required mini-map ROI and guard band.
- Green safe area, amber extrapolated area, and red cropped required area.
- Coverage, utilization, IoU, roll, pitch/yaw, and positioning guidance.

### Focus and matchability

- The 1:1 inspector described above.
- Current versus best focus position and MR95-20.
- Match examples at the passing and failing resolution boundary.
- Exposure, clipping, motion-blur, and temporal-stability warnings.

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
|-- matchability_curve.png
|-- match_examples.png
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
  input_space: camera_undistorted_px
  input_size_px: [1920, 1080]
  input_origin: top_left
  input_axes: [right, down]

  # matrix_3x3 maps input pixels to output pixels.
  transform_direction: input_pixel_to_output_pixel
  matrix_3x3:
    - [1.234, 0.012, -322.1]
    - [-0.008, 1.229, -104.7]
    - [0.00001, -0.00002, 1.0]

  output_size_px: [1080, 2400]
  output_origin: top_left
  output_axes: [right, down]

  # Canonical phone coordinate represented by output pixel (0, 0).
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

required_roi:
  # Canonical phone-screen coordinates; example values only.
  kind: minimap
  xywh: [18.0, 42.0, 310.0, 310.0]
  guard_band_px: 31.0
  source: game_profile

matchability:
  metric: MR95-20
  patch_size_mm: 20.0
  patch_size_source: measured
  primary_cells_across_patch: 64
  smallest_matchable_detail_mm: 0.3125
  adb_to_camera_cells: 64
  camera_to_camera_cells: 71
  static_cells: 71
  moving_cells: 64
  bootstrap_95_ci_cells: [60, 67]
  translation_error_p95_minimap_fraction: 0.008
  rotation_error_p95_deg: 0.72
  catastrophic_mismatch_rate: 0.004
  matcher:
    name: "aria_minimap_matcher"
    version: "1.0"
    config_sha256: "sha256-of-matcher-configuration"

confidence:
  geometry: 0.96
  matchability: 0.91
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
  matchability_curve: "matchability_curve.png"
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

At startup, a short validation target may verify reprojection and matching without replacing the saved calibration. Recalibration is required when the camera mode, physical mount, focus mode, phone orientation, logical screen size, or display scaling changes. Exposure or brightness changes may permit quick matchability revalidation when geometry remains stable.

## 13. Acceptance and Confidence

Geometry confidence includes:

- accepted-corner count and distribution;
- detected convex-hull coverage;
- robust reprojection residuals;
- burst-to-burst transform stability;
- extrapolation distance at the required ROI;
- intrinsic-calibration quality when available.

Matchability confidence includes:

- held-out target count;
- static and moving trials;
- reference-mode agreement;
- colour/orientation worst-case performance;
- bootstrap interval width;
- temporal stability after focus and exposure are locked.

Initial task gates should require:

- 100% required mini-map coverage;
- 100% configured guard-band coverage;
- no required-ROI pixels outside the valid mask;
- transform uncertainty below the game profile's permitted position error;
- P95 translation and rotation errors below the mini-map estimator's declared limits;
- catastrophic mismatch rate below the declared route-safety limit.

The project should derive numeric limits from recorded target-game evidence rather than treating the illustrative YAML values as universal thresholds.

## 14. Implementation Boundaries

The implementation should add:

- a rig-calibration service responsible for targets, corner fitting, geometry, quality trials, and YAML generation;
- a calibrated UVC source responsible for lens correction and calibration metadata;
- a workbench calibration UI with the geometry overlay and 1:1 inspector;
- a small YAML loader/validator shared by all consumers;
- evidence rendering and repeatable synthetic tests for matrix direction, origin, scaling, partial-screen geometry, and matchability scoring.

Rig calibration remains separate from game-specific mini-map calibration. Rig calibration converts physical-camera pixels into canonical phone-screen observations. Mini-map calibration then finds and models the game UI within that normalized coordinate system.
