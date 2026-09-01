# Aria Trace HIK calibration release

This Windows x64 package contains four independent console applications and a
source-distributed Python camera adapter. The applications are built with
CPython 3.12.10 and PyInstaller in one-folder mode. One-folder mode was chosen
over one-file extraction and Nuitka because it starts predictably, keeps native
OpenCV DLLs inspectable, and works with the separately installed HIK MVS runtime
without a C compiler.

## Prerequisites

- Windows x64.
- HIK MVS installed, including its camera driver, runtime, and Python `MvImport`
  wrapper. The vendor runtime is not redistributed.
- Android platform-tools (`adb`) available on `PATH`, or supplied with `--adb`.
- For continuous `--android-capture scrcpy` only: scrcpy 4.1 available on
  `PATH`, or its server supplied with `--scrcpy-server`, and FFmpeg available
  on `PATH` or supplied with `--ffmpeg`.
- `--android-capture adb-screenshot` requires neither scrcpy nor FFmpeg.
- USB debugging authorized on the Android phone.
- For `import hikcam`, a user-managed Python environment. Run
  `install-camera-adapter.bat` once in that environment.
- For every tool in a user-managed Python environment, run
  `install-python-tools.bat` once in that environment.

Python executables, their Python/native-wheel dependencies, Python source, and
the small AriaTrace native phone-target APK are bundled. ADB and HIK MVS remain
user-managed environment dependencies; scrcpy and FFmpeg are user-managed only
for continuous scrcpy acquisition. Rig calibration
installs the target APK only when its package is absent, then launches its
immersive `SurfaceView` through ADB reverse. The camera adapter does not operate
the phone during streaming.

## Start here

Run these from the extracted release directory:

```bat
rig-calibration.bat
zigzag-acquisition.bat
minimap-calibration.bat SESSION --rotation START END --movement START END
game-color-calibration.bat SESSION
camera-adapter-demo.bat --game-id genshin-impact --mode dual
```

Acquisition is game-agnostic. With no `--game-id` or `--android-package`, it
records whichever landscape game the user prepares at the confirmation prompt.
Use `--android-package org.example.game` to launch any explicit installed
package, or add `--game-id` when known-game identity should be retained for
later profile publication.

All arguments remain available with `--help`. The helper scripts only supply
the bundled tool locations and forward every user argument to the application.

Rig calibration uses ADB screenshot and Android display-state probes as
diagnostic evidence. These probes do not reject a product run; camera-observed
ChArUco geometry and final calibration evidence remain authoritative. Test
harnesses can opt into the old strict behavior with
`--strict-display-screenshot-verification`.

The default native presenter removes browser/Gallery fullscreen behavior. Set
`ARIA_PHONE_TARGET_APK` or pass `--phone-target-apk` only when the bundled APK
is relocated. `--panel-scale auto` enables ChArUco/native-surface scaling on MTK
platforms; use `--panel-scale adb` or `--panel-scale hik_charuco` explicitly to
override it. Active-app size mismatch and physical-DPI anisotropy are saved as
non-gating diagnostics in both JSON and commented YAML bundles.

## Pure-Python helpers

The release exposes the same tools through the user's current Python
environment; no bundled executable or interpreter is used:

```bat
install-python-tools.bat
python-tools.bat rig-calibration
python-tools.bat setup show
python-tools.bat zigzag-acquisition
python-tools.bat minimap-calibration SESSION --rotation START END --movement START END
python-tools.bat game-color-calibration SESSION
python-tools.bat camera-adapter-demo --game-id genshin-impact --mode dual --gui
```

Set `ARIA_PYTHON` before these commands to select a particular interpreter.
Python applications can call the helpers without a subprocess:

```python
import aria_tools

aria_tools.rig_calibration(["--headless", "--save"])
aria_tools.zigzag_acquisition(["--android-package", "org.example.game"])
```

Configure shared defaults once under the effective profile root:

```bat
python-tools.bat setup configure --camera-id DA9066154 --phone-id RFCR91GWXLX --game-id genshin-impact --rig-repeatability relaxed
python-tools.bat setup show
python-tools.bat setup profiles --active-only
python-tools.bat profiles list --active-only
python-tools.bat profiles show REVISION_ID
python-tools.bat profiles activate REVISION_ID
```

`rig-repeatability` is the single gate policy for both conservative rig reuse
and GUI save protection. `strict`, `balanced`, and `relaxed` expand to
documented metric-specific limits internally; commands cannot persist
independent thresholds that drift from the selected policy. The default
`relaxed` reuse check accepts ChArUco corner-alignment p95 <= 16 full-sensor px;
its GUI save guard remains 12 camera pixels for three consecutive frames.

The release `python` directory must be on `PYTHONPATH` when importing
`aria_tools` or `hikcam` from outside that directory.

## Camera adapter

Add the release `python` directory to `PYTHONPATH`, then:

```python
import hikcam

with hikcam.HikCamera(config={
    "game_id": "genshin-impact",
    "mode": "dual",          # full, minimap, or dual
    "rectify": True,
    "color_order": "BGR",
}) as camera:
    frames = camera.get_frames()
    phone = frames["full"]
    minimap = frames["minimap"]
```

Modes and transforms:

| Mode | `rectify=True` | `rectify=False` |
|---|---|---|
| `full` | rig-normalized phone image | hardware-ROI camera image |
| `minimap` | phone/game-normalized mini-map | native camera mini-map ROI |
| `dual` | synchronized normalized phone + mini-map | synchronized native rig ROI + native mini-map ROI |

`dual` derives both images from exactly one camera acquisition. Automatic
profile selection happens once at construction; there are no registry reads per
frame. Use `camera.get_frame()` for one output and `camera.get_frames()` for
dual output. Coordinate conversion methods are available on the same object:
`adb_to_camera_adapter_points(...)` and
`camera_adapter_to_adb_points(...)` when rectification is enabled.
These HIK-compatible image return values are unchanged. AriaTrace consumers
can separately call `camera.get_aria_frame_metadata()` (or pass `"full"` /
`"minimap"` after `get_frames()`) to obtain the last image's declared space,
ROI, orientation, and provenance.

The adapter deliberately locks the calibrated exposure, gain, white balance,
ROI, and optional MVS Bayer gamma/color matrix for the session.

The same demo can instead use Hikrobot's installed MVS Python SDK
(`MvCameraControl_class`) through the native full-sensor acquisition source. This
comparison bypasses AriaTrace profiles and calibration:

```bat
camera-adapter-demo.bat --camera-library native --camera-id CAMERA_ID
```

Hikrobot MVS is not bundled. Install MVS normally; if its Python wrapper cannot
be found automatically, pass `--mvs-python-path` pointing to the directory that
contains `MvCameraControl_class.py`. Native mode supports `--mode full` only and
does not manage a phone display.

Rig calibration can conservatively reuse the active immutable rig revision:

```bat
rig-calibration.bat --reuse-if-unchanged --headless --save
```

The precheck resets acquisition to the full sensor and writes the fresh frame
plus a saved-versus-detected ChArUco corner overlay. It skips only when the
board alignment meets the configured full-sensor displacement limit. Pixel
brightness, color, background, and environmental lighting are not reuse gates;
otherwise the same command continues with full calibration.

GUI calibration pauses with the full ChArUco board and live camera guide before
collecting geometry. Press Enter/Space in that preview to begin, or Q/Esc to
cancel. The default output is
`%ARIA_PROFILE_ROOT%\calibrations\hik-calibration-TIMESTAMP` (or the same path
under local `profiles\` when the environment variable is unset).

Game contrast/color calibration consumes one complete synchronized ADB +
rig-normalized HIK session. It verifies that the session used the currently
active rig revision, fits MVS gamma and CCM on held-out frame pairs, publishes
an independent `rig_game_color` profile, and writes review images:

```bat
game-color-calibration.bat sessions\calibration\SESSION
```

The preferred game-level command automatically runs every capability supported
by the session and skips unavailable ones without inventing results:

```bat
game-calibration.bat sessions\calibration\SESSION --game-id GAME_ID
```

Its JSON summary distinguishes `accepted`, `review_required`,
`skipped_missing_or_ineligible_data`, and `failed`. Screen-upright orientation
is calibrated from multiple synchronized ADB/HIK still pairs and published as
an independent `rig_game_orientation` revision. The adapter resolves it once
at construction. Rectified modes precompose the quarter-turn into the existing
lookup map and retain one remap per frame; an unrectified mode performs the
zero-interpolation quarter-turn while preserving
the complete image-space parent transform. This is not game-world north.

To collect the session without running scrcpy, use:

```bat
zigzag-acquisition.bat --android-capture adb-screenshot --require-hik
```

Swipe distances default to 20% of landscape display height on both axes
(216 px horizontally and vertically at 2400x1080). Use
`--horizontal-swipe-distance-px PX` and `--vertical-swipe-distance-px PX` to
override them independently.

One lossless ADB screenshot and one rig-normalized HIK frame are retained after
each settled swipe. Omitting `--require-hik` permits an Android-only mini-map
session, but such a session cannot fit HIK color because it contains no HIK
pixels. The ADB stream is only a lossless PNG image series and never creates a
video. The optional HIK stream remains MJPEG. This mode never locates or
executes external FFmpeg. Every retained PNG is referenced from a `frames.jsonl`
record carrying its `metadata.image_space`; filenames or dimensions never imply
coordinate space.

The color command's `SESSION` is immutable measurement provenance. Automatic
configuration selects the game context, active rig revision, evidence directory,
and publication root. An explicit output path is diagnostic-only. Requiring the
source session prevents a silently chosen recording from changing calibration
results.

Mini-map calibration follows the same registry policy: its evidence defaults
below the effective profile root, the portable `phone_game` revision is
published there, and an available active local rig is composed into `rig_game`.
Use `--candidate` when the evidence should be retained without activation.

Every successful active rig or game-color calibration also writes
`hikcam_adapter.py`. This generated module embeds the exact rig JSON, dense
rectification map, and applicable game profiles. It reads no profile registry
at runtime. It still requires the AriaTrace Python package and the installed
HIK MVS runtime.

An adapter snapshot can also be regenerated from automatic profile selection:

```bat
python-tools.bat profiles export-adapter my_hikcam.py --game-id genshin-impact --mode dual --color-policy game_matched
```

## Outputs and profiles

Rig, mini-map, and game-color calibration evidence and profiles write below the
effective profile root; synchronized capture sessions continue to write below
`sessions/` unless overridden. Production camera opening
resolves active profiles by camera, phone, game, display context, and adapter
mode. Explicit calibration paths are diagnostic overrides only.

Automatic profile storage uses `ARIA_PROFILE_ROOT` when it is set; otherwise
it uses `profiles/` below the current working directory. The release helper
scripts start from the extracted release directory, so the executables and
`import hikcam` share the same local registry. Applications launched from a
different directory should set `ARIA_PROFILE_ROOT` to the shared registry.

See [ARCHITECTURE.md](ARCHITECTURE.md) for diagrams and
[REFERENCE_BENCHMARKS.md](REFERENCE_BENCHMARKS.md) for measured reference data.

## Rebuild the executables

The complete application source and build entry points are included under
`python/`. From an extracted release:

```bat
cd python
setup-python-3.12.10.bat
build-standalone-release.bat
```

The build script rejects every interpreter version except exactly 3.12.10,
creates an isolated environment, uses the exact package versions recorded in
`requirements-standalone-release.txt`, smoke-tests all executable `--help`
surfaces, and emits a ZIP plus a SHA-256 sidecar.
