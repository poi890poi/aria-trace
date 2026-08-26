# Rig calibration package

`acquisition.rig_calibration` is the dependency-light implementation of the design in [`RIG_CALIBRATION.md`](../../RIG_CALIBRATION.md). It contains calibration math and artifacts, not camera, ADB, game, map, workbench, or dataset integrations.

The package provides:

- generic frame/stimulus/observer adapter interfaces;
- camera-to-screen homography, coverage, utilization, IoU, required-region quality, and confidence;
- multi-view planar intrinsic calibration;
- optional ChArUco target generation and detection;
- lens rectification and one-matrix frame normalization;
- no-interpolation 1:1 inspection crops and review-image renderers;
- matcher-independent MR95 evaluation plus a phase-correlation baseline;
- alternating-signal control-to-perception latency and paired-endpoint delay;
- commented `calibration.yaml` persistence and validation;
- dependency-free spatial-fragment export.

`FrameSample`, `ControlEvent`, and `SignalObservation` are package-owned values. An external UVC, ADB, workbench, or dataset adapter converts its native packets into these values; the calibration package never imports those implementations.

## Dependency boundary

Ordinary geometry, normalization, matchability, latency, YAML, and spatial export require NumPy, OpenCV, and PyYAML. Live ChArUco functions require the ArUco module from `opencv-contrib-python-headless`; importing the package does not require it. [`requirements-rig-calibration.txt`](../../requirements-rig-calibration.txt) describes an isolated environment with that feature enabled.

The current project environment may use ordinary `opencv-python-headless`. In that environment, calling a ChArUco function fails with an explicit installation message while every other package function remains available.

## Build a calibration from correspondences

```python
from pathlib import Path

from acquisition.rig_calibration import (
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
from acquisition.rig_calibration import FrameNormalizer, load_calibration_yaml

root = Path("artifacts/rig_calibrations/example")
calibration = load_calibration_yaml(root / "calibration.yaml")
normalizer = FrameNormalizer(calibration, root)

# The calibrated capture layer normally supplies this undistorted frame.
normalized = normalizer.normalize(undistorted_camera_frame)
```

`FrameNormalizer.normalize_raw` is available to an adapter that intentionally owns both raw capture and the lens model. Consumers should normally request an undistorted or already normalized frame rather than repeat lens handling.

## Measure alternating-signal latency

```python
from acquisition.rig_calibration import estimate_latency, estimate_paired_delay

adb_latency = estimate_latency(control_events, adb_state_observations)
camera_latency = estimate_latency(control_events, camera_state_observations)
paired_delay = estimate_paired_delay(adb_latency, camera_latency)
```

Events and observations name their clocks. Different clocks require an explicit affine clock transform; measured causal delay is never absorbed into that transform.

## Verify

```powershell
python -m unittest tests.test_rig_calibration -v
```
