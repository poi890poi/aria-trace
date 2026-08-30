# Aria Trace HIK calibration release

This Windows x64 package contains three independent console applications and a
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
- scrcpy 4.1 available on `PATH`, or its server supplied with
  `--scrcpy-server`.
- FFmpeg available on `PATH`, or supplied with `--ffmpeg`.
- USB debugging authorized on the Android phone.
- For `import hikcam`, a user-managed Python environment. Run
  `install-camera-adapter.bat` once in that environment.

Only Python executables, their Python/native-wheel dependencies, and Python
source are bundled. ADB, scrcpy, FFmpeg, and HIK MVS are all user-managed
environment dependencies. The camera adapter does not operate the phone during
streaming.

## Start here

Run these from the extracted release directory:

```bat
rig-calibration.bat
zigzag-acquisition.bat --game-id genshin-impact
minimap-calibration.bat SESSION OUTPUT --rotation START END --movement START END
camera-adapter-demo.bat --game-id genshin-impact --mode dual
```

All arguments remain available with `--help`. The helper scripts only supply
the bundled tool locations and forward every user argument to the application.

Rig calibration uses ADB screenshot and Android display-state probes as
diagnostic evidence. These probes do not reject a product run; camera-observed
ChArUco geometry and final calibration evidence remain authoritative. Test
harnesses can opt into the old strict behavior with
`--strict-display-screenshot-verification`.

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

The adapter deliberately locks the calibrated exposure, gain, white balance,
ROI, and optional MVS Bayer gamma/color matrix for the session.

## Outputs and profiles

The executables write below `artifacts/`, `sessions/`, and `profiles/` in the
release directory unless overridden. Production camera opening resolves active
profiles by camera, phone, game, display context, and adapter mode. Explicit
calibration paths are diagnostic overrides only.

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
