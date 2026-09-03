# IRIS HIK calibration release

This Windows x64 package contains five independent console applications and a
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
- For continuous `--android-capture scrcpy`, the release helpers use the
  bundled scrcpy 4.1 server and pinned LGPL FFmpeg executable. Explicit
  `--scrcpy-server` and `--ffmpeg` arguments still override them.
- `--android-capture adb-screenshot` requires neither scrcpy nor FFmpeg.
- USB debugging authorized on the Android phone.
- For `import hikcam`, a user-managed Python environment. Run
  `install-camera-adapter.bat` once in that environment.
- For every tool in a user-managed Python environment, run
  `install-python-tools.bat` once in that environment.

Python executables, their Python/native-wheel dependencies, Python source, the
small IRIS native phone-target APK, scrcpy server, and FFmpeg are bundled. ADB
and HIK MVS remain user-managed environment dependencies. Third-party license
texts are under `third_party/`; corresponding source is published as the
companion `IRIS-Third-Party-Source.zip` release asset. Rig calibration
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
`IRIS_PHONE_TARGET_APK` or pass `--phone-target-apk` only when the bundled APK
is relocated. Its canonical release path is
`phone-target/iris-phone-target.apk`; the same binary is also present at
`python/android/phone-target/iris-phone-target.apk` for imports from a different
project directory. `--panel-scale auto` enables ChArUco/native-surface scaling on MTK
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

Set `IRIS_PYTHON` before these commands to select a particular interpreter.
Python applications can call the helpers without a subprocess:

```python
import iris_tools

iris_tools.rig_calibration(["--headless", "--save"])
iris_tools.zigzag_acquisition(["--android-package", "org.example.game"])
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
`iris_tools` or `hikcam` from outside that directory.

## Camera adapter

Add the release `python` directory to `PYTHONPATH`, then:

```python
import hikcam

with hikcam.HikCamera(config={
    "game_id": "genshin-impact",
    "mode": "dual",          # full, minimap, or dual
    "rectify": True,
    "color_order": "BGR",
    "orientation_behavior": "projection",
    "rotate": 0,
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
These HIK-compatible image return values are unchanged. IRIS consumers
can separately call `camera.get_iris_frame_metadata()` (or pass `"full"` /
`"minimap"` after `get_frames()`) to obtain the last image's declared space,
ROI, orientation, and provenance.

Orientation is explicit. `orientation_behavior="as_is"` is the default and
preserves rig output. `"projection"` applies the game-up turn already composed
by profile management. `"image"` obtains one full rig-normalized HIK frame and
one ADB screenshot during initialization, checks all four turns, then opens the
requested stream. `rotate=0|90|180|270` adds a manual clockwise turn. The
selected turn is folded into rectification maps, or applied as one discrete
quarter-turn when rectification is disabled; there is no per-frame profile or
matching work.

The adapter deliberately locks the calibrated exposure, gain, white balance,
ROI, and optional MVS Bayer gamma/color matrix for the session.

The same demo can instead use Hikrobot's installed MVS Python SDK
(`MvCameraControl_class`) through the native full-sensor acquisition source. This
comparison bypasses IRIS profiles and calibration:

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
`%IRIS_PROFILE_ROOT%\calibrations\hik-calibration-TIMESTAMP` (or the same path
under local `profiles\` when the environment variable is unset).

Game contrast/color calibration consumes one complete synchronized ADB +
rig-normalized HIK session. It verifies that the session used the currently
active rig revision, fits MVS gamma and CCM on held-out frame pairs, publishes
an independent `rig_game_color` profile, and writes review images:

```bat
game-color-calibration.bat sessions\calibration\SESSION
```

The preferred game-level command automatically runs every capability supported
by the session and skips unavailable ones without inventing results. A session
with synchronized HIK images and its coordinate-space conversion also updates
the portable phone-game color reference and active rig-specific game-color
profile:

```bat
game-calibration.bat sessions\calibration\SESSION --game-id GAME_ID
```

For separate zigzag and micro-movement captures, pass both session folders.
Their order does not matter: IRIS classifies the recorded input patterns and
establishes mini-map geometry before composing cursor evidence.

```bat
game-calibration.bat sessions\calibration\ZIGZAG_SESSION sessions\calibration\MICRO_SESSION --game-id GAME_ID
```

The evidence output defaults under the configured profile root. Override it
only when needed with `--output EVIDENCE_FOLDER`.

After rig displacement, publishing the fresh rig centrally recomposes portable
mini-map geometry and game orientation. Game-up is stored relative to the
phone's rotation-0 panel space; recomposition derives a new adapter-relative
quarter-turn from the fresh rig's ChArUco calibration-display orientation. It
does not copy the old rig-relative turn. An older rig-specific color fit is
reported as requiring fresh evidence; the adapter safely falls back to
rig-locked color—even when `game_matched` was requested—instead of failing to
open. Run `game-calibration` on a synchronized ADB/HIK session to restore
game-matched color for the new rig revision.

Cursor interpretation comes from the active game model, independently of the
acquisition name. The default (`cursor_follows=character`) treats zigzag as a
static-cursor series and micro-movement as a rotating-cursor series; a
camera-following cursor reverses that mapping:

```bat
python-tools.bat profiles configure-game GAME_ID --cursor-follows camera
zigzag-acquisition.bat --capture-mode micro-movement --android-capture adb-screenshot
```

Static and rotating cursor series are both optional. Static-only evidence does
not manufacture a rotation center. A rotating series adds the center and
rotating-envelope diameter to the active phone-game profile, which callers can
query with `camera.get_cursor_geometry()`. Cursor calibration reports evidence
levels independently: `shape_only`, `rotation_center_only`, or
`rotation_center_and_shape`. A valid rotation center is retained and published
even if the later persistent-contour or polar-shape fit is unavailable. Pivot
fitting is color agnostic: it searches the temporal cursor envelope for its
center of rotation and does not require a predefined HSV interval or a fitted
cursor shape.

Its JSON summary distinguishes `accepted`, `review_required`,
`skipped_missing_or_ineligible_data`, and `failed`. Screen-upright orientation
is calibrated from multiple synchronized ADB/HIK still pairs and published as
an independent `rig_game_orientation` revision. The adapter resolves it once
at construction. Rectified modes precompose the quarter-turn into the existing
lookup map and retain one remap per frame; an unrectified mode performs the
zero-interpolation quarter-turn while preserving
the complete image-space parent transform. This is not game-world north.

For a skipped or partial cursor series, the console and JSON report the
acquisition pattern, expected cursor behavior, failing stage, and measured
details. `cursor_center_heatmap.png` and `cursor_center_symmetry.png` expose the
color-independent pivot evidence. The optional shape stage reports its own
component and contour failures without invalidating the fitted center.

To collect the session without running scrcpy, use:

```bat
zigzag-acquisition.bat --android-capture adb-screenshot --require-hik
```

Swipe distances default to 10% of the current game-display width horizontally
and 20% of its height vertically (240x216 px at 2400x1080). Each 120 ms swipe
starts at `(72% width, 50% height)`. Use
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

Each retained game frame also has `metadata.game_context`: game ID, foreground
package/component, logical landscape/portrait state, physical USB-edge
orientation, observed Android surface turn, and canonical phone rotation-0
basis. This content identity complements rather than replaces
`metadata.image_space`. The session-level copy is
`manifest.json#context.game_facts`.

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
at runtime. It still requires the IRIS Python package and the installed
HIK MVS runtime.

An adapter snapshot can also be regenerated from automatic profile selection:

```bat
python-tools.bat profiles export-adapter my_hikcam.py --game-id genshin-impact --mode dual --color-policy game_matched
```

The optional `--mask-policy minimap_circle` adapter setting precomposes the
verified fitted mini-map circle into the dense map and retains one remap per
frame. In the GUI demo, enable it with:

```bat
camera-adapter-demo.bat --game-id GAME_ID --mode dual ^
  --mask-policy minimap_circle --gui
```

Rectification must remain enabled. Press `G`, `B`, or `C` to toggle the
display-only geometry overlay, boundary, or cursor respectively; these keys do
not change masking or returned frames. Export every active portable game model,
mini-map/cursor geometry, and phone-game color reference with:

```bat
python-tools.bat profiles export-deployment iris-games.zip
python-tools.bat profiles import-deployment iris-games.zip --activate
```

Rig and camera calibration remain local and are never included in this bundle.

## Outputs and profiles

Rig, mini-map, and game-color calibration evidence and profiles write below the
effective profile root; synchronized capture sessions continue to write below
`sessions/` unless overridden. Production camera opening
resolves active profiles by camera, phone, game, display context, and adapter
mode. Explicit calibration paths are diagnostic overrides only.

Automatic profile storage uses `IRIS_PROFILE_ROOT` when it is set; otherwise
it uses `profiles/` below the current working directory. The release helper
scripts start from the extracted release directory, so the executables and
`import hikcam` share the same local registry. Applications launched from a
different directory should set `IRIS_PROFILE_ROOT` to the shared registry.
Use an absolute path, especially when `hikcam` is imported from a third-party
application whose working directory is not the IRIS release directory. A
profile tree may be moved as a unit: runtime-file paths are relative, and IRIS
can rebuild a missing `.registry` index from the revision manifests and active
pointers stored in the tree.

The recovery restores profiles and active selections. Operator defaults in
`.registry/settings.json` are separate machine-local configuration; preserve
that file during a complete move or rerun
`python-tools.bat setup configure ...` after a copy that intentionally omits
`.registry`.

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
