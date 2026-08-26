# Standalone Windows rig-calibration app

The PySide6 application wraps the independent calibration core with an
operator-controlled desktop workflow. Merely importing or launching it does
not open a camera, run ADB, bind a phone-target port, or write an artifact.
Each external action has a visible button and each result is saved only through
the final review action.

## Source launch

```powershell
python -m acquisition.rig_calibration.app
```

The normal sequence is:

1. Enter the phone's canonical pixel dimensions and start the target service.
2. Open its URL on the phone, tap fullscreen, and use the displayed ChArUco board.
3. Enter a camera ID and click **Start camera**. Camera probing is optional and
   happens only through **Probe indices**, because probing briefly claims devices.
4. Set the required ROI and fit one reviewed geometry frame.
5. Inspect the boundary overlay, normalized frame, exact 1:1 pixels, and 4x
   nearest-neighbour magnification.
6. After the implementation is updated, run the ISO 12233 e-SFR/MTF capture
   and established feature-matching evaluation; latency remains a separate
   alternating-signal measurement.
7. Optionally list ADB devices or capture one ADB reference image.
8. Save the reviewed YAML and evidence bundle.

Distance and pose are explicitly approximate when the entered phone diagonal
or camera HFOV are assumptions. ISO 12233:2024 e-SFR is the standardized
imaging-resolution method; MTF50 and MTF10 are derived crossing-frequency
summaries. Feature repeatability, matching score, MMA at declared
normalized-screen-pixel thresholds, match counts, coverage, and downstream
pose error measure computer-vision matching separately.

The replacement UI must label both sampled grids explicitly. Its primary curve
and MTF crossings are display-referred cycles per display pixel (`cy/dpx`),
including the one-pixel-line Nyquist endpoint at `0.5 cy/dpx`. Cycles per camera
pixel (`cy/cpx`) is a secondary native-analysis view. A `cy/mm` axis is permitted
only with measured display pitch. The UI must not say merely `lines/pixel`, and
it must derive the display-referred axis from pre-warp camera samples and local
geometry rather than measuring authoritative MTF from the resampled preview.

Important current limitation: the built application still exposes a
project-defined `MR95` sweep. That name and calculation are deprecated and
must not be used as standards-compliant calibration or acceptance evidence.
The controlled generated/ADB reference acquisition remains useful raw evidence,
but the evaluator, UI labels, YAML fields, and quality gates require the
standards-aligned implementation described in `RIG_CALIBRATION.md`. Legacy
results must not be silently relabeled as MTF or MMA.

## Adapter customization

Camera, ADB, and target presentation are public adapter boundaries in
`device_adapters.py` and `phone_target.py`. A custom implementation is selected
without editing the GUI:

```powershell
python -m acquisition.rig_calibration.app `
  --camera-adapter my_package.camera:create_camera `
  --adb-adapter my_package.phone:create_adb `
  --target-adapter my_package.target:create_presenter
```

When Windows chooses the wrong LAN address for the built-in phone page, pass
`--target-advertised-host 192.168.x.y`. Binding and advertising are separate;
the default bind port is ephemeral to avoid collisions with other services.

Factories take no arguments and return `CameraAdapter`, `AdbAdapter`, or
`PhoneTargetAdapter` instances respectively. A camera adapter supplies
timestamped BGR `FrameSample` values. An ADB adapter may represent real ADB,
another phone-side agent, or a prerecorded/deterministic reference source.

The built-in OpenCV camera adapter accepts numeric indices and adapter-specific
stream paths. Its DirectShow index probe is bounded and opt-in. The built-in
ADB adapter uses `adb devices` and `adb exec-out screencap -p`, also only after
an explicit button action.

## Isolated Windows build

```powershell
.\build-rig-calibration-app.ps1
```

The script creates `.tools/rig-calibration-app-build`, installs dependencies
there, and writes a one-folder distribution beneath the ignored
`artifacts/rig-calibration-app/windows` directory. It does not install packages
globally and does not launch the resulting application. The executable is:

```text
artifacts/rig-calibration-app/windows/AriaTraceRigCalibration/AriaTraceRigCalibration.exe
```
