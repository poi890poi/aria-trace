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
6. Run the controlled MR95 sweep and alternating-signal latency measurement.
7. Optionally list ADB devices or capture one ADB reference image.
8. Save the reviewed YAML and evidence bundle.

Distance and pose are explicitly approximate when the entered phone diagonal
or camera HFOV are assumptions. The acceptance-oriented resolving-power value
is the end-to-end `MR95` matchability result, not the relative Laplacian focus
number.

The MR95 sweep normally compares the known generated raster with normalized
camera observations. Its explicit ADB-reference checkbox instead captures each
displayed target through the configured `AdbAdapter`, enabling the intended
ADB-to-camera matching measurement without coupling the core to ADB.

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
