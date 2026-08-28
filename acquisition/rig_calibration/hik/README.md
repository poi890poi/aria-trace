# HIK camera / Android display rig calibration

This folder is an isolated HIK MVS camera plugin plus two standalone commands.
It reuses the repository's ChArUco geometry, ISO 12233 e-SFR, calibration bundle,
and Data Matrix Decode grading rather than defining replacement algorithms.

## Dependencies

- Hikrobot/HIKROBOT MVS, including `MvCameraControl_class.py` and native DLLs.
- ADB with one phone selected by an explicit serial.
- OpenCV contrib (`cv2.aruco`).
- `zxing-cpp` for built-in Data Matrix Decode grading. The setup installs the
  compatible 2.x writer on Python 3.7/3.8 and 3.x on newer Python.

Set `HIK_MVS_PYTHON_PATH`, or pass `--mvs-python-path`, to the MVS
`Samples/Python/MvImport` directory. No camera or phone is accessed merely by
importing this package.

The loader does not depend on the MVS installer updating the system `PATH`.
It discovers the matching 32/64-bit Common Files runtime from the supplied SDK
location and from standard MVS install locations. Nonstandard installs can set
`MVS_RUNTIME_PATH` explicitly.

The existing desktop adapter loader can use this camera plugin factory:

```text
acquisition.rig_calibration.hik.driver:create_camera_adapter
```

List cameras without claiming one:

```powershell
python -m acquisition.rig_calibration.hik.calibrate --list-cameras --mvs-python-path C:\path\to\MvImport
```

Interactive calibration:

```powershell
.\calibrate-hik-rig.bat
```

With exactly one HIK camera and one Android phone attached, the launcher detects
both devices, MVS, ADB, repository dependencies, and a timestamped output
directory. If multiple cameras or phones are connected, it asks which one to
use. Run `.\calibrate-hik-rig.bat --test` for a headless observation run that
never saves a calibration bundle. Advanced flags remain available for automation
and unusual installations.

The interactive launcher uses an isolated GUI-capable OpenCV contrib runtime;
the repository's existing headless OpenCV environments remain unchanged. If
that runtime is absent, run `.\setup-hik-rig.bat` once. The launcher checks for
it before calibration instead of failing at the focus stage.

Targets are pushed as full-resolution PNG files and opened with Android's
built-in Gallery/Display activity. The presenter captures an Android screenshot
after each change. The initial ChArUco square counts follow the display aspect
ratio, filling the raster without stretching cells (`9x20` on a `1080x2400`
display). A target is accepted only after the physical display reports `ON` and
three consecutive screenshots correlate with the target while at least 99.95%
of consecutive screenshot pixels are stable and at least 99.5% match the target.
The small target tolerance permits fixed display cutouts and rounded corners;
the consecutive-frame check and minimum post-tap delay reject transient system
bars. The accepted presentation also records any viewer quarter-turn. Use
`--display-component package/.Activity` only when automatic built-in-viewer
selection is ambiguous. Chrome or another browser is not part of this flow.

ADB measures and locks the current Android orientation only to keep the target
stable. Orientation calibration itself is optical: the ChArUco IDs establish
app-up and app-right relative to the camera sensor, including any rotation
introduced by the phone or image viewer. The final adapter applies that fitted
transform so output-image up is always app-up.

The calibration JSON records the natural panel raster, Android rotation, actual
fullscreen viewer activity/canvas, and the ChArUco-derived app axes in camera
coordinates. Orientation drift or loss of fullscreen during a target paint
causes an explicit failure rather than silently corrupting the geometry.

Before ChArUco detection, the camera uses a short hardware auto-exposure/gain
bootstrap and discards those frames. This is only to make geometry detectable;
the final exposure and gain are still selected manually from the measured white
mask under the refresh-period and clipping constraints.

After geometry is known, the phone shows a neutral-gray target only inside the
camera-visible display region. On the verified MV-CS016-10UC feature tree, HIK
`AOI1` is set to that projected camera region for one-shot exposure/gain and
`AOI2` is set to it for one-shot white balance. The raw one-shot exposure is
allowed to reach the longest permitted panel-refresh-safe duration (16667 us,
or two complete display periods, at 120 Hz with the default cap) instead of inheriting a stale camera auto-exposure
ceiling. It is then quantized to its nearest permitted panel-refresh multiple. Other permitted
multiples are still measured as safety fallbacks when the nearest value cannot
meet clipping/noise constraints. HIK's one-shot WB is retained as the seed and
the quarter-exposure blurred-white measurement applies only a residual channel
correction.

Interactive calibration opens a live camera overview with a compact side panel
showing the current stage, exposure mode/time, shutter rate, gain, black level,
white balance, frame rate, image statistics, and ChArUco corner count. The image
is fit within the usable desktop area and is never covered by the text. Closing
this overview hides it without cancelling calibration.

Exposure endpoints are constrained to panel-period boundaries. The default
search evaluates 0.5, 1, and 2 times the panel-refresh shutter rate: at 120 Hz,
these are 16667, 8333, and 4167 us. This preserves complete display cycles while
allowing lower gain. Pass `--max-exposure-periods 3` to allow 25000 us; the
independent `--max-shutter-multiplier 3` option permits a faster 2778 us candidate.
The workflow seeks 90% code value in the known white patch with no channel
having 5% or more clipped samples, preferring lower gain and then shorter
exposure. White balance is measured at one-quarter exposure after strong blur,
then the selected exposure is restored. Before residual WB, exposure selection
uses the brightest channel as the predicted balanced-white level because the WB
method raises the dim channels to that channel. After WB is locked, a new camera
burst verifies the actual three-channel mean and clipping. If the residual
correction misses 88-92% or clips more than 5% in any channel, the calibrator
automatically restores the HIK one-shot WB. Remaining target misses are recorded
as warnings and evidence; they do not prevent a functioning camera from continuing.

The focus window shows four camera crops at native 1:1 sampling in a 2x2
viewport, a separate fit-to-view image of the complete camera frame for
positioning, and metrics in a side panel. The overview is never cropped; the
four evidence crops remain unscaled native pixels. The combined window is sized
to the usable desktop area rather than extending under the taskbar.
The chart is centered at 62% of the camera-visible phone region, leaving a 19%
reacquisition margin on every side while the rig is moved. It contains one
complete high-contrast rectangular frame. Primitive
Otsu thresholding and quadrilateral contour fitting recover it in each camera
frame. Its known physical dimensions and orthogonal vanishing points provide a
camera-only pitch, yaw, and perpendicular lens-to-panel distance estimate; no
game image or game-pose estimator participates. A focal estimate is retained
once observable because an almost front-on rectangle has vanishing points too
far away for stable focal recovery. The UI reports that case as unavailable
instead of fabricating a distance.
The phone target contains four ISO 12233-style slanted edges distributed across
the visible field: two near-horizontal and two near-vertical. It reports the
conservative four-edge current/history maximum for Laplacian sharpness and
e-SFR MTF50/MTF10. Cycles/display-pixel are always retained. When Android reports
physical `xdpi/ydpi`, the same result is converted along each edge normal to
`lp/mm`, with `android_active_display_mode_xdpi_ydpi` recorded as its source;
logical UI density is never used as physical scale. Keys:

- `S`: save the complete calibration bundle.
- `R`: rerun geometry and locked imaging after physically moving the rig.
- `D`: run the Data Matrix size/condition sweep, then return to focus.
- `Q` or Escape: exit without saving.

Headless mode skips the focus window. Saving remains explicit:

```powershell
python -m acquisition.rig_calibration.hik.calibrate ... --headless --save
```

Use `--grade-data-matrix` to run the sweep before saving. At least 1000 trials
per size are needed to resolve an observed 99.9% rate; this is the default.
The live camera overview remains open and identifies the current module size and
trial. Poor sizes stop early as soon as even perfect remaining trials could not
reach 99.9%; with 1000 planned trials, the second failure skips directly to the
next doubled size. A potentially qualifying size still runs all required trials.
The built-in implementation grades only ISO/IEC 15415's binary Decode parameter
and is **not a complete or certified ISO/IEC 15415 verifier**. Supply a complete
verifier as `--complete-grader-plugin package.module:callable`. The callable is
passed `(camera_bgr, expected_payload, condition_metadata)` and must return a
mapping containing at least numeric `grade` and payload-match evidence.
ZXing 2.x and 3.x encoder/decoder signatures are both supported. If an optional
decoder or external grader still fails, grading is marked unavailable and the
calibrator returns to the interactive session instead of discarding the camera
calibration.

## Saved result and stream

Saving is atomic and creates `calibration.yaml`, `valid_screen_mask.png`,
`rectification_maps.npz`, evidence, and `hik_camera_calibration.json`. The JSON
records exact IDs and modes, exposure/gain/WB, hardware ROI, camera-visible phone
`(x,y,w,h)`, phone/viewer orientation, full-sensor homographies, and output geometry.
It also records which WB path was retained, every attempted WB measurement, and
non-fatal imaging-quality warnings.
It also records measured ROI payload reduction, adapter read/frame-interval
distributions, and a reference host-display-request to first stable camera-frame
latency. This reference is explicitly not labeled as game input latency.

```powershell
python -m acquisition.rig_calibration.hik.stream calibration-output\hik_camera_calibration.json --gui
```

```python
from acquisition.rig_calibration.hik.stream import open_camera

camera = open_camera("calibration-output/hik_camera_calibration.json")
ok, rectified_phone_bgr = camera.read()
camera.release()
```

### HIK-compatible import surface

Downstream code that normally imports a high-level HIK camera module can alias
the calibrated adapter directly:

```python
import acquisition.rig_calibration.hik.camera as hikcam

with hikcam.HikCamera(
    config={"calibration": "calibration-output/hik_camera_calibration.json"}
) as camera:
    rgb = camera.get_frame()
```

Alternatively set `ARIA_HIK_CALIBRATION` and use `hikcam.HikCamera()` with no
arguments. The facade implements `HikCamera`/`Camera`, context management,
`get_frame`, `robust_get_frame`, `read`, `get_shape`, `reset`, `get_all_ips`,
exposure/gain/WB controls, RGB/BGR selection, and item-style common GenICam
settings. It returns RGB by default; call `set_bgr()` for OpenCV order.

This is high-level source compatibility, not an emulation of Hikrobot's ctypes
`MvCamera` structures or integer-returning `MV_CC_*` ABI. Calibrated ROI and
output dimensions are immutable because changing them would invalidate the
saved rectification map.

The loader rejects an effective hardware ROI that differs from the calibrated
one. Frames use the saved dense map, with homography warp as a fallback.

## Phone lifecycle

The calibrator snapshots screen timeout, brightness mode/value, stay-awake,
immersive-mode, auto-rotate, and user-rotation settings; wakes/dismisses the
keyguard; locks brightness to manual 255/255; holds the display awake; pushes
targets to the built-in image viewer; then restores every setting and sends `KEYCODE_SLEEP`. Cleanup
runs on normal exit and exceptions. It does not unlock a credential-protected
phone.

MVS's Beginner/Guru selector filters only its GUI. The SDK backend can describe
GenICam nodes for diagnostics, but automatic calibration deliberately tunes only
controls with a direct measured effect: shutter multiplier, gain, white balance,
black level, and hardware ROI. It does not search the whole Guru feature tree.
The rectified facade locks the calibrated imaging controls and keeps ROI/output
shape immutable because changing them would invalidate the evidence or maps.

## Hardware verification status

Offline tests use simulated boundaries. Actual MVS node names/ranges, Android
Display presentation, panel refresh reporting, exposure/WB response, focus metrics, and
Data Matrix rates must be verified with the exact camera, lens, phone, MVS
release, and Android build before a bundle is treated as production calibration.
