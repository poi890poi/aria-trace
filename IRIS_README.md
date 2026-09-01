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
- scrcpy and FFmpeg only when continuous scrcpy capture is explicitly selected.
  ADB-screenshot acquisition requires neither.

Set a shared profile root before using more than one command:

```powershell
$env:IRIS_PROFILE_ROOT = "E:\iris-profiles"
python -m iris_tools setup configure `
  --camera-id CAMERA_ID `
  --phone-id ADB_SERIAL
python -m iris_tools setup show
```

When `IRIS_PROFILE_ROOT` is unset, the local `profiles/` directory is used.

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

Rig calibration uses full-sensor frames. Hardware ROI belongs to the runtime
adapter plan and is applied only after calibration and space conversion.

## Game acquisition and calibration

Capture a game-agnostic zigzag session from the prepared foreground game:

```powershell
python -m iris_tools zigzag-acquisition --android-capture adb-screenshot
```

If synchronized HIK evidence is required, add `--require-hik`. Use
`--game-id GAME_ID` when the resulting calibration should be published under a
stable game identity.

Run the available calibration work from one captured session:

```powershell
python -m iris_tools game-calibration SESSION --game-id GAME_ID
```

The workflow uses the available, space-tagged evidence and skips calibration
that cannot be supported by the session. Dedicated commands remain available:

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

Portable phone/game geometry and color references can be exported and imported.
Rig and rig/game compositions remain local because they depend on the camera,
lens, panel position, and the exact rig revision.

## Camera adapter

For an existing application that imports Hikrobot's low-level
`MvCameraControl_class` module and cannot be changed, see
[Replacing the MVS Python wrapper without changing an application](IRIS_MVS_DROP_IN_COMPATIBILITY.md).
That document describes a proposed compatibility shim; this low-level replacement is
not currently implemented.

### Import from a project in another folder

The Windows release ships the adapter as Python source rather than as a pip
package. A third-party project can live anywhere, but its Python process must be
able to import both `hikcam.py` and the bundled `aria_trace` package from the
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

The printed path should be inside `IRIS-Windows-x64\python`. Do not copy only
`hikcam.py`, and do not add an executable directory under `apps/` to
`PYTHONPATH`; the adapter depends on the accompanying Python package. Hikrobot
MVS remains user-managed and must be available to the same interpreter.

When importing directly from an IRIS source checkout, use the checkout root in
`PYTHONPATH` instead of the release's `python/` directory. The checkout root
contains both `hikcam.py` and `aria_trace/`.

The public adapter keeps the HIK-shaped import and common acquisition methods:

```python
import hikcam

with hikcam.HikCamera(config={
    "game_id": "GAME_ID",
    "mode": "dual",       # full, minimap, or dual
    "rectify": True,
    "color_order": "BGR",
    "color_policy": "game_matched",
}) as camera:
    frames = camera.get_frames()
    phone_frame = frames["full"]
    minimap_frame = frames["minimap"]
```

Rectification is enabled by default. A rectified game profile folds game-upright
orientation into the existing transformation map, so it does not add a second
per-frame rotation pass. With rectification disabled, the adapter returns the
documented native ROI space and applies only the required discrete orientation.

Run the GUI demo against calibrated or native HIK acquisition:

```powershell
python -m iris_tools camera-adapter-demo `
  --game-id GAME_ID --mode dual --color-policy game_matched --gui

python -m iris_tools camera-adapter-demo `
  --camera-library native --camera-id CAMERA_ID --gui
```

The adapter resolves profiles once when it opens. It does not wake, unlock,
touch, launch applications on, or power-manage the phone during ordinary
streaming.

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

Detailed operator and packaging documentation is under
`docs/standalone-release/`.
