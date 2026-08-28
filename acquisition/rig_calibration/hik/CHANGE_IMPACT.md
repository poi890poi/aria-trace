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

## Interpretations and compatibility

- Faster candidates retain the original `N * refresh_hz` shutter-rate rule.
  Slower candidates are reciprocal-integer rates so their exposure integrates
  an exact number of complete panel periods. The defaults span 0.5x, 1x, and 2x
  refresh rate (two periods through half a period).
- The temporary quarter exposure used for white balance is an explicit
  exception requested by the workflow and is restored afterward.
- The built-in Data Matrix evaluator reports the standardized ISO/IEC 15415
  **Decode parameter** (`4/A` or `0/F`). It does not claim complete ISO/IEC
  15415 conformance. A complete external grader can be supplied as a plugin.
  The requested 99.9% value is labeled an observed operational rate, not an ISO
  aggregate grade.
- ZXing 2.x and 3.x writer/decoder signatures are supported. A runtime error in
  optional Data Matrix grading produces an unavailable result and returns control
  to calibration; it cannot discard an otherwise usable rig session.
- Built-in Decode grading drains buffered frames after every phone target change,
  rectifies and crops to the transformed symbol bounds with a four-module quiet
  margin, then uses a
  bounded ZXing downscale/global-threshold/Otsu fallback sequence measured on the
  connected rig. ZXing 2.3.0 decoded real 2 px/module crops and full-frame targets
  from 4 px/module, so no decoder replacement is required.
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

- Exposure/WB use bounded frame counts and control searches. Data Matrix trials
  are operator-triggered and bounded by CLI parameters. The live preview remains
  active during grading, and a size is rejected early only when its mathematically
  best possible final rate is already below 99.9%.
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
