# Rig calibration package

`aria_trace.services.calibration.rig` is the dependency-light implementation of the design in [`RIG_CALIBRATION.md`](../../../../RIG_CALIBRATION.md). It contains calibration math and artifacts, not camera, ADB, game, map, Workbench, or dataset integrations.

The package provides:

- generic frame/stimulus/observer adapter interfaces;
- camera-to-screen homography, coverage, utilization, IoU, required-region quality, and confidence;
- multi-view planar intrinsic calibration;
- optional ChArUco target generation and detection;
- lens rectification and one-matrix frame normalization;
- no-interpolation 1:1 inspection crops and review-image renderers;
- alternating fixed-patch ISO/IEC 15415 Decode grading for ISO/IEC 16022 Data
  Matrix symbols, with exact integer display-pixel modules;
- display-referred ISO 12233 slanted-edge e-SFR/MTF analysis from native,
  pre-homography camera samples;
- homography-ground-truth feature repeatability, matching score, MMA, spatial
  coverage, match counts, and downstream geometry error;
- alternating-signal control-to-perception latency and paired-endpoint delay;
- commented `calibration.yaml` persistence and validation;
- dependency-free spatial-fragment export.

`FrameSample`, `ControlEvent`, and `SignalObservation` are package-owned values. An external UVC, ADB, workbench, or dataset adapter converts its native packets into these values; the calibration package never imports those implementations.

## Standalone Windows application

The optional [PySide6 application](../../../apps/rig_calibrator/README.md) supplies the guided USB-camera
workflow while retaining this package boundary. Run it from source with:

```powershell
python -m aria_trace.apps.rig_calibrator
```

Camera, ADB, and phone presentation use public adapter classes and optional
`module:function` factories. Importing or launching the app does not claim
hardware, run ADB, start the phone server, or write into recorder sessions.
The isolated PyInstaller build is documented in the app guide.

The public calibration API and GUI no longer use the former project-defined
`MR95` evaluator. [`RIG_CALIBRATION.md`](../../../../RIG_CALIBRATION.md) selects the
standard ISO/IEC 15415 Decode parameter as the one feature-readability grade:
`4/A` for a successful reference decode or `0/F` for failure. Module width in
display pixels is a test condition, not a new metric. The code does not average
grades, infer a custom boundary, or claim a complete ISO/IEC 15415 overall
symbol grade. ISO 12233 e-SFR/MTF and ground-truth feature matching remain
supporting diagnostics and task validation.

Resolution frequencies retain their sampling domain. The primary
display-referred e-SFR/MTF result uses cycles per display pixel (`cy/dpx`), while
cycles per camera pixel (`cy/cpx`) is retained as the native analysis/audit
axis. Cycles per millimetre (`cy/mm` or `lp/mm`) is emitted only when physical
display pitch is measured or explicitly reported with provenance; logical UI
density is never treated as physical pitch. One-pixel-wide alternating phone lines are
`0.5 cy/dpx`, not an unqualified `1 line/pixel`; they are a Nyquist stress
target rather than a standalone resolution result. Samples are measured before
homography normalization and their frequency axis is transformed using the
local geometry, so resampling does not become part of the MTF measurement.

The ChArUco target is an atlas: marker/corner IDs locate the camera viewport in
the complete canonical display even when the camera sees neither the full
screen nor its outer boundary. Geometry, screen coverage, camera utilization,
and IoU are fitted first. Quality trials then use a conservative patch wholly
inside the camera-visible intersection with the required task ROI. A partial
screen can therefore be measured honestly, while a cropped required ROI still
prevents calibration acceptance.

## Dependency boundary

Ordinary geometry, normalization, e-SFR, feature matching, latency, YAML, and spatial export require NumPy, OpenCV, and PyYAML. Live ChArUco functions require the ArUco module from `opencv-contrib-python-headless`. The built-in Data Matrix encoder/decoder requires optional ZXing-C++; both current APIs and the legacy 2.x writer/decoder signatures are supported. Callers can inject another compatible encoder/decoder. Importing the package requires neither optional component. [`requirements-rig-calibration.txt`](../../requirements-rig-calibration.txt) describes an isolated environment with them enabled on supported Python versions.

## Grade Data Matrix Decode

```python
from aria_trace.services.calibration.rig import (
    DataMatrixObservation,
    grade_data_matrix_decode,
    summarize_data_matrix_decode_sweep,
)

grade = grade_data_matrix_decode(camera_patch, expected_payload="A7K2")
print(grade["grade"], grade["grade_letter"])  # 4.0 A, or 0.0 F

# Group standard observations by their declared display module width without
# averaging them or applying a project threshold.
report = summarize_data_matrix_decode_sweep(observations)
```

`alternating_data_matrix_targets(...)` renders one target at a time in the
same fixed visible patch. It therefore supports small displays and partial
camera coverage. A valid but unexpected payload is retained as stale/wrong
target evidence and excluded from the requested target rather than mislabeled
as an ISO Decode failure.

The current project environment may use ordinary `opencv-python-headless`. In that environment, calling a ChArUco function fails with an explicit installation message while every other package function remains available.

## Build a calibration from correspondences

```python
from pathlib import Path

from aria_trace.services.calibration.rig import (
    build_calibration,
    write_calibration_bundle,
)

calibration, geometry, valid_mask = build_calibration(
    calibration_id="camera-phone-20260826T120000Z",
    camera_points_xy=detected_camera_points,
    screen_points_xy=known_phone_screen_points,
    camera_size_px=(1920, 1080),
    screen_size_px=(1080, 2400),
    input_frame_id="aria://rig/camera-phone/camera/undistorted",
    canonical_screen_frame_id="aria://device/phone/screen/portrait/layout-1080x2400",
    required_region_screen_xy=required_region_polygon,
    required_roi={"kind": "caller-label", "polygon_xy": required_region_polygon},
)
write_calibration_bundle(Path("artifacts/rig_calibrations/example"), calibration, valid_mask)
```

`geometry.matrix_3x3` maps undistorted camera pixels directly to phone-screen coordinates. The matrix stored under `calibration["normalization"]` additionally applies the selected output origin and scale.

## Normalize a frame

```python
from aria_trace.services.calibration.rig import FrameNormalizer, load_calibration_yaml

root = Path("artifacts/rig_calibrations/example")
calibration = load_calibration_yaml(root / "calibration.yaml")
normalizer = FrameNormalizer(calibration, root)

# The calibrated capture layer normally supplies this undistorted frame.
normalized = normalizer.normalize(undistorted_camera_frame)
```

`FrameNormalizer.normalize_raw` is available to an adapter that intentionally owns both raw capture and the lens model. Consumers should normally request an undistorted or already normalized frame rather than repeat lens handling.

## Measure alternating-signal latency

```python
from aria_trace.services.calibration.rig import estimate_latency, estimate_paired_delay

adb_latency = estimate_latency(control_events, adb_state_observations)
camera_latency = estimate_latency(control_events, camera_state_observations)
paired_delay = estimate_paired_delay(adb_latency, camera_latency)
```

Events and observations name their clocks. Different clocks require an explicit affine clock transform; measured causal delay is never absorbed into that transform.

## Verify

```powershell
python -m unittest tests.test_rig_calibration -v
```
