# HIK rig-calibration change impact audit

## Classification and intended outcome

Primary type: new feature. Secondary types: hardware adapter, calibration
workflow, command-line tooling, documentation, and deterministic tests.

The new path calibrates a selected Hikrobot/HIK MVS camera against one selected
ADB phone, saves a reviewed partial-screen geometry and imaging configuration,
and exposes a small OpenCV/UVC-like rectified stream interface.

## Affected boundaries

- New code is isolated under `acquisition/rig_calibration/hik/`.
- The camera plugin implements the existing `CameraAdapter` contract and loads
  the vendor MVS Python wrapper only when the HIK path is invoked.
- The existing ChArUco geometry, ISO 12233 e-SFR/MTF, ISO/IEC 15415 Decode
  parameter, bundle, and `FrameNormalizer` implementations remain authoritative.
- The HIK workflow adds an ADB built-in Gallery/Display presenter. Each pushed
  target is screenshot-confirmed before camera evidence is accepted. The
  existing local HTTP target remains available to other calibration paths.

## Observable behavior

Previously there was no bundled HIK MVS driver or unattended ADB display
lifecycle. The new standalone calibration CLI explicitly opens one camera and
one phone, keeps the display awake, presents targets only inside the measured
camera-visible screen region, and powers the display off during cleanup. A
separate stream CLI reapplies the saved HIK controls and ROI and returns
rectified phone-display frames.

The saved-camera adapter also exposes an additive high-level `HikCamera`
compatibility facade in `hik/camera.py`. Consumers may use
`import acquisition.rig_calibration.hik.camera as hikcam` and retain common
context-manager, frame, exposure, and setting calls. It does not emulate the
vendor ctypes `MV_CC_*` ABI, and it rejects ROI/shape mutation because those
changes invalidate calibration.

The root `demo-hik-camera.bat` is an additive operator path for viewing this
adapter. It selects the newest saved calibration unless one is supplied, opens
the rectified GUI stream, and manages only Android panel power. One best-effort
`KEYCODE_WAKEUP` and the cleanup `KEYCODE_SLEEP` are the complete ADB command
surface: there is no power-state validation or frame-loop ADB traffic, and phone
settings, activities, rotation, brightness, and touch input are outside this
demo. Camera and display cleanup are protected by the same `finally` boundary.

The production `RectifiedHikCamera` deliberately trusts the saved calibration.
It does not repeat calibration identity/orientation checks or reject a camera-
aligned effective ROI; it composes rectification from the actual ROI returned
by the driver. Calibration retains all strict evidence gates. This keeps the
steady-state production path to camera read plus one OpenCV remap and prevents
diagnostic policy from blocking a valid user stream.

An explicit production `rectify=False` / CLI `--no-rectify` mode removes that
last image transform as well. It returns the hardware-ROI frame object directly,
does not load dense maps, and performs neither `cv2.remap` nor
`cv2.warpPerspective`. The tradeoff is deliberate: output stays in camera ROI
coordinates and does not guarantee canonical phone geometry or app-up. Default
behavior remains rectified for compatibility.

## Interpretations and compatibility

- Faster candidates retain the original `N * refresh_hz` shutter-rate rule.
  Slower candidates are reciprocal-integer rates so their exposure integrates
  an exact number of complete panel periods. The defaults span 0.5x, 1x, and 2x
  refresh rate (two periods through half a period).
- The temporary quarter exposure used for white balance is an explicit
  exception requested by the workflow and is restored afterward.
- The operator-facing Data Matrix result is an exact-payload **decode success
  rate**, not an overall ISO/IEC 15415 grade. Internally, a successful ZXing
  decode corresponds only to ISO/IEC 15415's binary Decode parameter; contrast,
  modulation, nonuniformity, damage, and aggregate symbol grading are not claimed.
  A complete external verifier can still be supplied as a plugin.
- ZXing 2.x and 3.x writer/decoder signatures are supported. A runtime error in
  optional Data Matrix decode testing produces an unavailable result and returns control
  to calibration; it cannot discard an otherwise usable rig session.
- Built-in decode testing drains buffered frames after every phone target change,
  rectifies and crops each transformed symbol independently, then uses a
  bounded ZXing downscale/global-threshold/Otsu fallback sequence measured on the
  connected rig. ZXing 2.3.0 decoded real 2 px/module crops and full-frame targets
  from 4 px/module, so no decoder replacement is required.
- Every failed exact-payload decode is preserved as review evidence independently
  of whether the calibration bundle is later saved. The evidence includes a red
  polygon on the original HIK frame, the raw-camera crop, the rectified decoder
  crop, the complete rectified camera frame, the displayed target, and a JSON
  index carrying the condition and decoder failure reason. This does not change
  qualification, exposure, geometry, or camera controls.
- Calibration files bind applicability to camera identity/mode, phone identity,
  orientation, screen size, and refresh rate. Another mode or device must not
  silently reuse the result.
- The workflow locks the observed Android rotation to keep the presentation
  stable, but treats ChArUco correspondences as the orientation authority. It
  records app-up/app-right in camera coordinates, thereby including phone and
  viewer rotation, and the saved rectification always maps app-up to output-up.
  Natural raster and fullscreen viewer canvas/activity remain diagnostic data;
  orientation drift fails a paint acknowledgement.

## Failure and recovery

- Vendor SDK absence, unsupported pixel conversion/control nodes, missing ADB,
  target paint timeout, inadequate ChArUco support, exposure/WB failure, or lost
  frames fail explicitly.
- ADB keep-awake, timeout, immersive policy, auto-rotate/user-rotation, pushed
  target files, and screenshot probes are
  restored in `finally`; the selected display is then turned off.
- Camera acquisition and the Display presenter are always closed. Files are
  written only on an explicit save request, using a temporary directory followed
  by an atomic directory rename.
- No firmware, persistent camera user set, phone application, recorder session,
  Workbench state, or game input is modified.

## Performance and verification

- Exposure/WB use bounded frame counts and control searches. Data Matrix testing
  is operator-triggered, requires at least 20 patterns per module size, defaults
  to 40, and passes at 95% exact-payload decode success. Up to eight independent
  patterns share one phone presentation; the connected-rig geometry fits eight
  patterns through 8 px/module and four at 16 px/module. A size is rejected early
  only when its mathematically best possible final rate is below 95%.
- Automatic camera optimization is intentionally limited to shutter multiplier,
  gain, white balance, black level, and hardware ROI. Other GenICam values are
  diagnostic evidence, not a high-dimensional search space.
- A controlled neutral-gray phone target now seeds HIK one-shot exposure, gain,
  and white balance. The camera's declared Auto Function AOI1/AOI2 regions are
  restricted to the projected target when supported; unsupported cameras fall
  back explicitly to their default auto area. The final exposure remains manual,
  refresh-quantized, mask-measured, and clipping/noise checked.
- Android brightness is locked to manual 255/255 for measurement and restored on
  cleanup. The HIK one-shot exposure ceiling is raised to the longest permitted
  refresh-safe duration. By default this now includes two complete display
  periods (16667 us at 120 Hz), while clipping/noise measurement and the faster
  refresh-rate-multiple candidates are preserved. Pre-WB selection predicts the
  residual-WB result from the brightest channel instead of rejecting the
  unbalanced channel average. A post-WB camera burst checks the result; a poor
  residual correction restores the HIK one-shot WB. Remaining target misses are
  saved as warnings instead of aborting calibration. Device I/O, missing-target,
  and unusable-geometry failures remain hard errors because no calibration can
  be produced from them.
- The focus target adds a complete rectangular frame for camera-only planar pose
  guidance inside a centered 62% chart with 19% movement margin per side. It
  uses thresholding, contour geometry, orthogonal vanishing points,
  and known Android physical pixel pitch. It does not call or change game-content
  pose estimation. Absolute distance is withheld until focal length is observable.
- The same detected rectangle reports phone in-plane rotation clockwise from
  camera-up during focusing, alongside pitch, yaw, and distance.
- The focus UI combines a complete uncropped fit overview for positioning with
  the existing four native 1:1 optical evidence crops; metric computation is unchanged.
- Unit tests use fake HIK and ADB backends to verify control quantization,
  clipping constraints, WB ratios, viewport cropping, cleanup, saved-config ROI
  composition, and UVC-like reads.
- A temporary camera-auto bootstrap makes ChArUco visible; final exposure/gain
  remain manual results from the measured white mask. Screenshot correlation
  verifies each built-in Display presentation and records viewer rotation.
- The final hardware ROI is exercised before save. The bundle records estimated
  payload reduction, measured adapter read/frame cadence, and three alternating
  display-to-camera transition trials as reference timing evidence.
- The MVS loader resolves native runtime directories independently of inherited
  `PATH`, with `MVS_RUNTIME_PATH` as an explicit override for nonstandard installs.
- Hardware verification results are recorded only after exercising the actual
  HIK MVS SDK, camera, display, and phone; offline tests remain deterministic.
