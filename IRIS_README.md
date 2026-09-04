# IRIS — Invariant Rig System

IRIS calibrates a fixed camera/Android-display rig and exposes normalized,
space-aware camera streams for computer-vision applications. It includes rig
calibration and reuse checks, synchronized Android/HIK acquisition, game-screen
orientation, mini-map and color calibration, profile management, evidence
generation, and a HIK-compatible Python camera adapter.

IRIS does not include a gameplay workbench, route learner, tracker, map model,
or gameplay policy. Those systems may consume IRIS streams and metadata but are
outside this product.

## Requirements

- Windows x64 and Python 3.12.10 for building the packaged applications.
- Hikrobot MVS for a HIK camera. Its SDK and driver remain user-managed.
- Android platform-tools with `adb` available on `PATH`.
- An authorized Android device and a supported camera.
- The standalone Windows release bundles scrcpy server and an LGPL FFmpeg
  executable for continuous scrcpy capture, with license texts and a companion
  corresponding-source archive. Source-tree use may instead supply either tool
  explicitly. ADB-screenshot acquisition requires neither.

Set a shared profile root before using more than one command:

```powershell
$env:IRIS_PROFILE_ROOT = "E:\iris-profiles"
python -m iris_tools setup configure `
  --camera-id CAMERA_ID `
  --phone-id ADB_SERIAL
python -m iris_tools setup show
```

When `IRIS_PROFILE_ROOT` is unset, the local `profiles/` directory is used.
Set it to an absolute path when IRIS is imported by another application. This
keeps profile selection independent of that application's working directory.

The profile root is relocatable. Move or copy the complete directory, update
`IRIS_PROFILE_ROOT`, and start the application normally. Immutable revisions
store runtime files by relative path. If a copy tool omits the `.registry`
directory, IRIS reconstructs its SQLite index and active selections from the
portable `profile.json` and `active.json` files. Absolute paths under
`provenance` are audit history only and are never used to open the adapter.

## Initial rig calibration

For interactive positioning and focus adjustment:

```powershell
python -m iris_tools rig-calibration
```

For unattended reuse of a calibrated rig, with full calibration only when the
ChArUco displacement check rejects reuse:

```powershell
python -m iris_tools rig-calibration `
  --reuse-if-unchanged `
  --headless `
  --save
```

Headless `--final-benchmark auto` (the default) programs and verifies the
runtime ROI, samples six frames, and skips reference-only display-transition
latency trials. Use `--final-benchmark full` for the complete benchmark,
`reduced` explicitly, or `skip` when only ROI programming evidence is needed.

Rig calibration uses full-sensor frames. Hardware ROI belongs to the runtime
adapter plan and is applied only after calibration and space conversion.

ChArUco determines the coarse phone pose. IRIS then measures broad horizontal and
vertical target edges as the standard high-precision panel-axis refinement. It runs
automatically in headless calibration and appears as a temporally stabilized metric
during GUI focusing. Corrections are accepted only below 5 degrees, then folded
into the authoritative camera-to-phone homography and existing dense rectification
map with no additional streaming pass or coordinate convention. If the measurement
is unavailable or disagrees with ChArUco, calibration continues with the ChArUco
result and records the reason and review evidence. In `--no-rectify` mode, the
adapter's frame metadata exposes the saved full-sensor panel-up vector as the
reference for aligning downstream mini-map axes to Android display space.

## Game acquisition and calibration

Capture a game-agnostic zigzag session from the prepared foreground game:

```powershell
python -m iris_tools zigzag-acquisition --android-capture adb-screenshot
```

Each sample is one continuous swipe with two independent timings. The default
travels for 0.12 seconds, keeps the finger down at the endpoint for 0.10
seconds, and only then sends `UP`. Configure them with
`--travel-seconds SECONDS` and `--endpoint-hold-seconds SECONDS`.
`--reset-seconds` remains the separate delay after release and before the next
swipe; `--screenshot-settle-seconds` remains the post-release image delay.

Capture balanced short movement pulses when a separate character-motion series
is useful:

```powershell
python -m iris_tools zigzag-acquisition `
  --capture-mode micro-movement `
  --android-capture adb-screenshot
```

The acquisition name describes only the physical input pattern. It does not
claim that the on-screen cursor rotates. The active game model supplies that
meaning:

| `cursor_follows` | Zigzag series | Micro-movement series |
|---|---|---|
| `character` (default) | static cursor | rotating cursor |
| `camera` | rotating cursor | static cursor |

Every game capture carries two independent identities. `metadata.image_space`
declares raster, ROI, orientation, and transforms; `metadata.game_context`
declares game ID, foreground package/component, landscape/portrait state,
physical USB-edge orientation, and the observed Android surface turn. The same
facts are summarized in `manifest.json` under `context.game_facts`. Consumers
must not infer either identity from a filename, image dimensions, or the app
that happens to be foreground later.

The calibrated mini-map circle also carries directed canonical game-up and
game-right unit vectors. This keeps orientation observable even though a circle
alone is rotationally symmetric. When a fresh rig is published, IRIS transforms
the portable circle and its axes together, then chooses the one quarter-turn that
puts those axes upright. A conflicting Android Surface hint is retained as
non-gating diagnostic evidence; it cannot override the image-space geometry.
Legacy profiles without directed axes continue through the prior Surface/game
orientation fallback instead of blocking unattended operation.

Rotation-center fitting uses the color-agnostic temporal symmetry of those
balanced samples. It does not require a predefined cursor HSV range or a cursor
shape fit; an unavailable shape is reported separately while the center remains
usable.

Configure the model before calibration when the default is wrong:

```powershell
python -m iris_tools profiles configure-game GAME_ID `
  --cursor-follows camera `
  --minimap-orientation rotating `
  --game-orientation landscape
python -m iris_tools profiles show-game GAME_ID
```

`game-orientation` is a phone-pose-independent game fact with only two values:
`landscape` (default) or `portrait`. Acquisition separately records the
foreground package, exact Android surface turn, and phone pose. Profile
composition uses that observed placement when available; otherwise it assumes
USB-right for landscape or USB-bottom for portrait without storing that
placement assumption as a property of the game.

If synchronized HIK evidence is required, add `--require-hik`. Use
`--game-id GAME_ID` when the resulting calibration should be published under a
stable game identity.

Run the available calibration work by passing the captured session folder:

```powershell
python -m iris_tools game-calibration SESSION --game-id GAME_ID
```

When zigzag and micro-movement were captured as separate sessions, pass both
folders in the same command. IRIS reads their manifests, identifies each
acquisition pattern, runs zigzag first to establish the mini-map boundary, and
then composes the cursor result from micro-movement. Their command-line order
does not matter.

```powershell
python -m iris_tools game-calibration `
  ZIGZAG_SESSION MICRO_MOVEMENT_SESSION `
  --game-id GAME_ID
```

Evidence is written automatically under the configured profile root. Use
`--output EVIDENCE_FOLDER` only when a specific evidence location is needed.

The workflow uses the available, space-tagged Android evidence and skips
calibration that cannot be supported by the session. Its phone-game result is
portable and has no rig revision dependency. Cursor evidence is optional: a
rotating series fits the rotation center, rotating envelope diameter, and
shape; a static series fits only the observable shape/span unless a verified
center already exists; both series may accumulate into the same active
phone-game profile. IRIS never fabricates a rotation center from static data.

Locked rig imaging and HIK auto white balance are the default color policy.
Optional synchronized color fitting is non-gating and must be requested; it is
published for review unless activation is separately authorized:

```powershell
python -m iris_tools game-calibration SESSION --game-id GAME_ID `
  --include-color

python -m iris_tools game-calibration SESSION --game-id GAME_ID `
  --include-color --activate-color
```
Dedicated commands remain available:

```powershell
python -m iris_tools minimap-calibration SESSION `
  --rotation ROTATION_START ROTATION_END `
  --movement MOVEMENT_START MOVEMENT_END
python -m iris_tools game-color-calibration SESSION --game-id GAME_ID
```

## Profile management

```powershell
python -m iris_tools profiles list --active-only
python -m iris_tools profiles resolve --game-id GAME_ID --mode dual
python -m iris_tools profiles show REVISION_ID
python -m iris_tools profiles activate REVISION_ID
```

Automatic adapter resolution ranks active variants lexicographically from
physical/static facts to software/fluid facts: camera identity and adapter,
panel raster/density/refresh, platform and game identity/package, then game
logical raster, viewport/layout, rotation/insets, version, and finally
revision time. Requested camera and game IDs remain hard candidate boundaries.

Portable phone/game geometry and color references can be exported and imported.
Rig and rig/game compositions remain local because they depend on the camera,
lens, panel position, and the exact rig revision.

Export or import every active portable game model and calibration in one
deployment bundle:

```powershell
python -m iris_tools profiles export-deployment iris-games.zip
python -m iris_tools profiles import-deployment iris-games.zip --activate
```

The bundle contains game models, canonical phone-game geometry (including
cursor geometry), and portable phone-game color references for all games. It
does not contain a rig or camera calibration; import composes compatible
phone-game geometry with the active local rig.

## Camera adapter

For an existing application that imports Hikrobot's low-level
`MvCameraControl_class` module and cannot be changed, see
[Replacing the MVS Python wrapper without changing an application](IRIS_MVS_DROP_IN_COMPATIBILITY.md).
That document describes a proposed compatibility shim; this low-level replacement is
not currently implemented.

### Import from a project in another folder

The Windows release ships the adapter as Python source rather than as a pip
package. A third-party project can live anywhere, but its Python process must be
able to import both `hikcam.py` and the bundled `rig_runtime` package from the
release's `python/` directory.

The recommended arrangement is to keep IRIS in one stable location and launch
the third-party application with an explicit module path and shared profile
root. In PowerShell:

```powershell
$irisRelease = "C:\Tools\IRIS-Windows-x64"
$projectPython = "D:\my-vision-app\.venv\Scripts\python.exe"

# One-time dependency installation into the third-party project's environment.
$env:IRIS_PYTHON = $projectPython
& "$irisRelease\install-camera-adapter.bat"

# Required each time the third-party application is launched.
$env:PYTHONPATH = "$irisRelease\python" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:IRIS_PROFILE_ROOT = "E:\iris-profiles"
& $projectPython "D:\my-vision-app\app.py"
```

`IRIS_PROFILE_ROOT` must identify the same profile registry used by rig and game
calibration. Otherwise the adapter falls back to a `profiles/` directory under
the current working directory and may appear uncalibrated. `IRIS_PYTHON` only
tells the helper batch file which interpreter receives dependencies; it does
not make IRIS importable by itself.

For a project-specific launcher, put the same setup beside the application:

```powershell
# run-with-iris.ps1 in D:\my-vision-app
$irisRelease = "C:\Tools\IRIS-Windows-x64"
$env:PYTHONPATH = "$irisRelease\python" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:IRIS_PROFILE_ROOT = "E:\iris-profiles"
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\app.py"
```

Verify the selected adapter before starting the camera:

```powershell
& $projectPython -c "import hikcam; print(hikcam.__file__)"
```

After construction, the adapter exposes the effective migrated registry root
and any automatic index recovery in
`camera.resolved_config["profile_registry"]`. This is resolved once during
initialization; frame streaming does not consult the registry.

The printed path should be inside `IRIS-Windows-x64\python`. Do not copy only
`hikcam.py`, and do not add an executable directory under `apps/` to
`PYTHONPATH`; the adapter depends on the accompanying Python package. Hikrobot
MVS remains user-managed and must be available to the same interpreter.

When importing directly from an IRIS source checkout, use the checkout root in
`PYTHONPATH` instead of the release's `python/` directory. The checkout root
contains both `hikcam.py` and `rig_runtime/`.

The public adapter keeps the HIK-shaped import and common acquisition methods:

```python
import hikcam

with hikcam.HikCamera(config={
    "game_id": "GAME_ID",
    "mode": "dual",       # full, minimap, or dual
    "rectify": True,
    "color_order": "BGR",
    "color_policy": "rig_locked",
    "mask_policy": "minimap_circle",  # or "none"
    "orientation_behavior": "projection",  # as_is, projection, or image
    "rotate": 0,                      # 0, 90, 180, or 270 clockwise
}) as camera:
    frames = camera.get_frames()
    phone_frame = frames["full"]
    minimap_frame = frames["minimap"]
    boundary = camera.get_minimap_geometry("minimap")
    cursor = camera.get_cursor_geometry("minimap")
    game_model = camera.get_game_model()
```

Callers may inspect the same ranked candidates and pin immutable revisions at
construction time:

```python
candidates = hikcam.HikCamera.list_profiles(
    {"camera_id": "CAMERA_ID", "game_id": "GAME_ID"},
    kinds=["rig", "rig_game", "rig_game_orientation"],
    active_only=False,
)

camera = hikcam.HikCamera(config={
    "camera_id": "CAMERA_ID",
    "game_id": "GAME_ID",
    "mode": "dual",
    "profile_revisions": {
        "rig_game": candidates["rig_game"][0]["revision_id"],
    },
})
```

A selected rig-game profile carries its exact rig and phone-game dependencies;
IRIS rejects manual combinations that splice incompatible geometry.

`get_minimap_geometry()` reports the fitted outer boundary in canonical phone
space and, for a rectified runtime stream, a space-tagged center and ellipse
size suitable for rendering on that exact frame. `get_cursor_geometry()`
reports canonical spatial geometry and, when a verified
rotation center exists, its center and rotating-envelope diameter in the
selected runtime stream space. Static-only calibration reports the observed
static span but explicitly leaves the rotation center unavailable.

`mask_policy="minimap_circle"` is optional and requires rectification. IRIS
uses the final verified outer-boundary circle (the radial temporal fit, not the
raw heatmap or coarse Hough seed), writes `[-1, -1]` for off-circle entries in
the prebuilt mini-map remap, and therefore retains one `cv2.remap` call per
frame. In dual mode only the mini-map product is masked; the full phone stream
is unchanged.

Rectification is enabled by default. Orientation behavior is explicit and has
no per-frame registry lookup:

- `projection` (default) applies the game-up turn already composed by the
  profile builder from portable game/surface facts and the current rig, plus
  `rotate`.
- `as_is` preserves the rig output, plus the optional explicit `rotate` value.
- `image` acquires one full rig-normalized HIK frame and one ADB screenshot at
  initialization, tests all four turns, then opens the requested stream. A
  failed or ambiguous match falls back to `as_is`; it does not mutate a
  portable profile.

With rectification, the selected turn is folded into the existing map. Without
rectification, it is a discrete quarter-turn. `rotate` always means an
additional clockwise 0, 90, 180, or 270 degrees.

Run the GUI demo against calibrated or native HIK acquisition:

```powershell
python -m iris_tools camera-adapter-demo `
  --game-id GAME_ID --mode dual --orientation-behavior projection --gui

python -m iris_tools camera-adapter-demo `
  --game-id GAME_ID --mode dual --orientation-behavior image --gui

python -m iris_tools camera-adapter-demo `
  --game-id GAME_ID --mode dual --mask-policy minimap_circle --gui

python -m iris_tools camera-adapter-demo `
  --camera-library native --camera-id CAMERA_ID --gui
```

The calibrated GUI draws the mini-map boundary in cyan, the cursor rotation
center/envelope in magenta, and the canonical `GAME UP`/`GAME RIGHT` axes in
green/orange. Press `G` to toggle all geometry, `B` to toggle only the boundary,
`C` to toggle only cursor geometry, and `A` to toggle the game axes. The game
axes remain visible when the circular boundary is hidden. These are
display-only overlays; returned camera frames are unchanged. FPS is a rolling
average and the telemetry label refreshes twice per second so it remains
readable.

Mini-map masking requires an active mini-map profile, `--mode minimap` or
`--mode dual`, and rectification (the default; do not pass `--no-rectify`). It
is an adapter output policy, not a GUI overlay: off-circle pixels in the
mini-map stream are black, while the full stream in dual mode remains intact.

The adapter resolves profiles once when it opens. It does not wake, unlock,
touch, launch applications on, or power-manage the phone during ordinary
streaming.

After recognizable game content is ready, an integrator can request a
non-blocking four-orientation ADB/HIK correction and poll it without adding work
to the frame path:

```python
job = camera.request_game_orientation_correction(adb_bgr, hik_full_bgr)
state = camera.get_game_orientation_correction()
```

A confident result rebuilds the current adapter maps and persists only the
rig-dependent orientation composition. Ambiguous or failed evidence is a
retryable non-gating result. `camera.get_iris_geometry_postmortem()` returns the
requested/effective ROI, output/map observations, and calibration confidence
captured at open for retrospective diagnosis.

## Spatial evidence contract

Every IRIS-produced camera image, derived image, video stream, mask, crop, circle,
point, and quadrilateral carries an explicit coordinate-space record. Full ADB
screenshots are the only canonical-space exemption. Consumers must apply an
explicit registered transform before combining media or geometry from different
spaces.

Review evidence expands images onto a magenta checkerboard canvas and renders
the complete camera sensor and projected phone-display quadrilaterals. This makes
cropping, ROI, orientation, and out-of-image regions visible to a human reviewer.

## Standalone Windows release

Build with the pinned Python toolchain:

```powershell
.\setup-python-3.12.10.bat
.\build-standalone-release.bat
```

The default output is `artifacts/standalone-release/IRIS-Windows-x64` plus a ZIP
and SHA-256 file. The release bundles Python applications, Python source,
dependencies, helper scripts, and the phone-target APK. It does not bundle ADB
or Hikrobot MVS.

Release exports contain no `.git` tree or `.gitmodules`. The release manifest
records the source commit when available plus a deterministic hash of the
exported Python source tree, and publication recomputes that hash before upload.

Detailed operator and packaging documentation is under
`docs/standalone-release/`.
