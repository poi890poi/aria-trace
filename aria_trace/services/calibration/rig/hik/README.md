# HIK camera / Android display rig calibration

This folder contains device-independent HIK calibration algorithms, patterns,
color matching and coordinate-space conversion. The MVS camera integration lives
under `aria_trace.adapters.hik`; Android presentation lives under
`aria_trace.adapters.android`; orchestration and commands live under
`aria_trace.workflows` and `aria_trace.apps`. They reuse the repository's
ChArUco geometry, ISO 12233 e-SFR, calibration bundle and Data Matrix
encoding/decoding rather than defining replacement algorithms.

## Dependencies

- Hikrobot/HIKROBOT MVS, including `MvCameraControl_class.py` and native DLLs.
- ADB with one phone selected by an explicit serial.
- OpenCV contrib (`cv2.aruco`).
- `zxing-cpp` for built-in Data Matrix exact-payload decode testing. The setup installs the
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
aria_trace.adapters.hik.driver:create_camera_adapter
```

The production command also accepts another HIK-compatible adapter factory.
The same adapter instance is used by the reuse precheck and, when reuse is not
accepted, by the complete calibration workflow:

```powershell
python -m aria_trace.apps.hik_rig_calibration `
  --camera-adapter package.module:create_camera_adapter `
  --camera-id CAMERA_MODEL_ID `
  --phone-serial DISPLAY_SERIAL `
  --reuse-if-unchanged `
  --headless --save
```

The factory takes no arguments and returns `CameraAdapter`. Its reported
`adapter_id` and `device_id` are saved as profile identity, so different
cameras, lenses, resolutions, frame rates, and zoom modes can remain separate
models. A fresh saved run writes `run_timing.json`; reuse writes its stage
breakdown in `precheck.json`. Those receipts are produced by the production
paths rather than a benchmark-only calibration wrapper.

`virtualhikcam.driver:create_camera_adapter` is the repository's Android
Camera2 development implementation of this contract. It has persistent ROI
and control state and rejects concurrent opens, but makes no claims about HIK
hardware controls, sensor ROI bandwidth, or latency. See
`virtualhikcam/README.md`.

List cameras without claiming one:

```powershell
python -m aria_trace.apps.hik_rig_calibration --list-cameras --mvs-python-path C:\path\to\MvImport
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
display). Android's power-state report is best-effort telemetry and never gates
calibration. A target is accepted only after three consecutive ADB screenshots
correlate with the target while at least 99.95% of consecutive screenshot pixels
are stable. Exact pixel-match coverage remains review evidence rather than a gate.
The small target tolerance permits fixed display cutouts and rounded corners;
the consecutive-frame check and minimum post-tap delay reject transient system
bars. HIK ChArUco detection then proves that the composed target is physically
visible to the camera. The accepted presentation also records any viewer
quarter-turn. Use
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
the final exposure and gain come from a separate bounded HIK one-shot operation.

After geometry is known, the phone shows a neutral-gray target only inside the
camera-visible display region. On the verified MV-CS016-10UC feature tree, HIK
`AOI1` is set to that projected camera region for one-shot exposure/gain and
`AOI2` is set to it for one-shot white balance. The raw one-shot exposure is
limited to one complete display period (16667 us at 60 Hz), and auto gain is
limited to 12 dB. After HIK finishes, exposure and gain are read back and locked;
the workflow does not replace them with a manual candidate search. HIK's
one-shot WB is retained as the seed and
the quarter-exposure blurred-white measurement applies only a residual channel
correction.

Interactive calibration opens a live camera overview with a compact side panel
showing the current stage, exposure mode/time, shutter rate, gain, black level,
white balance, frame rate, image statistics, and ChArUco corner count. The image
is fit within the usable desktop area and is never covered by the text. Closing
this overview hides it without cancelling calibration.

The default HIK auto limits are one complete panel period and 12 dB. Override
them with `--max-exposure-periods` and `--max-auto-gain-db` when required by a
different rig. The known white patch measures and records clipping/noise after
the auto result is locked, but does not replace that result. White balance is
measured at one-quarter exposure after strong blur, then the locked exposure is
restored. Before residual WB, verification uses the brightest channel as the
predicted balanced-white level because the WB
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
The same primitive rectangle reports phone in-plane rotation clockwise from
camera-up so roll can be adjusted without invoking game pose estimation.
The phone target contains four ISO 12233-style slanted edges distributed across
the visible field: two near-horizontal and two near-vertical. It reports the
conservative four-edge current/history maximum for Laplacian sharpness and
e-SFR MTF50/MTF10. Cycles/display-pixel are always retained. When Android reports
physical `xdpi/ydpi`, the same result is converted along each edge normal to
`lp/mm`, with `android_active_display_mode_xdpi_ydpi` recorded as its source;
logical UI density is never used as physical scale. Keys:

- `S`: save the complete calibration bundle.
- `R`: rerun geometry and locked imaging after physically moving the rig.
- `D`: run the Data Matrix decode-success sweep, then return to focus.
- `Q` or Escape: exit without saving.

Headless mode skips the focus window. Saving remains explicit:

```powershell
python -m aria_trace.apps.hik_rig_calibration ... --headless --save
```

Use `--test-data-matrix-decode` (legacy alias `--grade-data-matrix`) to run the
sweep before saving. The test requires at
least 20 patterns per module size and defaults to 40; a size passes when at least
95% decode to their exact expected payload. The live camera overview identifies
the current module size, phone screen, and pattern range. Poor sizes stop early
once even perfect remaining patterns cannot reach 95%.

Several independently identified patterns share one phone image, each with its
own angle, color, intensity, cell, and rectified decode crop. The batch planner
uses up to eight patterns when geometry permits. On the connected 721x977-pixel
visible phone region it fits eight patterns per screen through 8 px/module and
four at 16 px/module, reducing the default 16-pixel test from 40 phone operations
to 10. Each cell remains white around its symbol; the raster contains the required
one-module Data Matrix quiet zone and the decoder crop retains another module.

The displayed percentage is simple exact-payload decode success. A successful
decode corresponds only to ISO/IEC 15415's binary Decode parameter; the UI does
not label it `4/A` and does **not** claim a complete or certified ISO/IEC 15415
grade. Supply a complete verifier as `--complete-grader-plugin package.module:callable`. The callable is
passed `(camera_bgr, expected_payload, condition_metadata)` and must return a
mapping containing at least numeric `grade` and payload-match evidence.
Every failed pattern is saved immediately in a sibling
`*-data-matrix-evidence-*` directory, even if the operator later exits without
saving calibration. Its original HIK frame marks the failed decoder crop with a
red polygon and label; raw and rectified crops, the displayed target, and a JSON
index preserve the exact angle, color, intensity, payload, and failure reason.
ZXing 2.x and 3.x encoder/decoder signatures are both supported. If an optional
decoder or external grader still fails, grading is marked unavailable and the
calibrator returns to the interactive session instead of discarding the camera
calibration.
Built-in testing drains buffered camera frames after each batched target change and
decodes each rectified, rotation-aware symbol crop. It uses a bounded set of ZXing
downscale and threshold fallbacks; this avoids treating one binarizer's miss as a
camera failure while retaining the exact-payload synchronization check.

## Saved result and stream

Saving is atomic and creates `calibration.yaml`, `valid_screen_mask.png`,
`rectification_maps.npz`, evidence, and `hik_camera_calibration.json`. The JSON
records exact IDs and modes, exposure/gain/WB, hardware ROI, camera-visible phone
`(x,y,w,h)`, phone/viewer orientation, full-sensor homographies, and output geometry.
It also records which WB path was retained, every attempted WB measurement, and
non-fatal imaging-quality warnings.
The hardware ROI and normalized output cover the raster bounding box of every
phone-display pixel visible in the full-sensor image; this coverage is never
eroded. Calibration adds an outward 8-display-pixel allowance by default for
fit and sampling tolerance. Override it when needed:

```powershell
.\calibrate-hik-rig.bat --visible-screen-margin-px 12
```

Exposure, focus, and Data Matrix targets use a separately recorded, fully
visible safe rectangle. Rectangular output corners outside the measured phone
footprint remain explicit in `valid_screen_mask.png`.
It also records measured ROI payload reduction, adapter read/frame-interval
distributions, and a reference host-display-request to first stable camera-frame
latency. It establishes solid-black/solid-white baselines, then measures three
alternating transitions (white-to-black, black-to-white, white-to-black). This
reference includes phone presentation and camera settling, and is explicitly
not labeled as game input latency or image-transform cost.

```powershell
python -m aria_trace.apps.hik_stream calibration-output\hik_camera_calibration.json --gui
```

For the normal operator demo, run this from the repository root:

```powershell
.\demo-hik-camera.bat
```

It selects the newest saved HIK calibration under `artifacts`, turns on only the
calibrated phone's display power, and opens the live rectified output. Press
`Q`, Escape, or close the window to stop. Camera release and display sleep run
from `finally`, including camera/stream failures. Display wake is a single
best-effort `KEYCODE_WAKEUP`: it is not polled or validated and an ADB failure
cannot block camera startup. No ADB work occurs in the frame loop. The demo does
not change phone settings, start an app, rotate the display, change brightness,
or send touch input. Pass a calibration JSON or its containing directory as the
first batch argument to override newest-calibration selection.

For minimum processing latency, pass `--no-rectify` after the calibration. This
returns the hardware-ROI camera raster directly and skips both dense
`cv2.remap` and homography `cv2.warpPerspective`:

```powershell
.\demo-hik-camera.bat ".\artifacts\hik-calibration-20260829-083137" --no-rectify
```

The low-latency output remains in camera ROI coordinates, so it is not
orthogonalized, app-up, or cropped to the canonical phone rectangle.

```python
from aria_trace.apps.hik_stream import open_camera

camera = open_camera("calibration-output/hik_camera_calibration.json")
ok, rectified_phone_bgr = camera.read()
camera.release()
```

### HIK-compatible import surface

Downstream code that normally imports a high-level HIK camera module can alias
the calibrated adapter directly:

```python
import aria_trace.adapters.hik.compat as hikcam

with hikcam.HikCamera(
    config={"calibration": "calibration-output/hik_camera_calibration.json"}
) as camera:
    rgb = camera.get_frame()
```

Production code can request the same zero-transform mode:

```python
with hikcam.HikCamera(
    config={
        "calibration": "calibration-output/hik_camera_calibration.json",
        "rectify": False,
    }
) as camera:
    hardware_roi_rgb = camera.get_frame()
```

The production reader trusts the saved calibration instead of repeating
calibration-time checks. It applies the locked controls, accepts the effective
hardware-aligned ROI returned by the camera, adjusts the rectification origin,
and enters the capture loop. Strict identity, geometry, orientation, exposure,
and evidence checks remain in calibration only.

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

This MV-CS016-10UC exposes no writable camera-side Gamma node. MVS does expose
host conversion gamma and CCM, and `MV_CC_GetImageForBGR` honors both inside its
existing Bayer conversion. A reviewed rig-game profile may therefore set them
once at adapter open to match synchronized ADB colors. The stream does not run
a separate contrast function, LUT, or color-conversion pass.

## Hardware verification status

Offline tests use simulated boundaries. Actual MVS node names/ranges, Android
Display presentation, panel refresh reporting, exposure/WB response, focus metrics, and
Data Matrix rates must be verified with the exact camera, lens, phone, MVS
release, and Android build before a bundle is treated as production calibration.
