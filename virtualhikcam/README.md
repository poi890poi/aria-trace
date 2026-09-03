# Virtual HIK-compatible camera

`virtualhikcam` exposes the same high-level adapter contract used by the real
HIK rig workflow, while acquiring an Android Camera2 stream through
`androidcam`. It is selected through the production rig application's camera
adapter factory; calibration and reuse code contain no virtual-camera branch.

Each camera model has a URI identity containing the Android device, Camera2
camera/lens ID, resolution, frame rate, zoom, and bit rate:

```text
virtual-hik://RFCR91GWXLX/camera/1?width=1280&height=720&fps=30&zoom=1&bit_rate=12000000
```

Different Camera2 IDs, resolutions, frame rates, or zooms are different camera
models and therefore have independent persistent state. State defaults under
`%LOCALAPPDATA%\IRIS\virtual-cameras`; set
`IRIS_VIRTUAL_HIK_STATE_ROOT` to override it.

ROI is persisted in full-frame coordinates. Every acquired frame declares its
effective ROI and local-to-parent transform. Zoom and ROI are software
operations after full-frame decode, so this driver does not validate physical
HIK controls, sensor-side bandwidth reduction, or HIK latency.

## Persistent displacement simulation

Each virtual camera model also persists an artificial physical displacement.
It is applied to the zoomed full-sensor image before the fixed sensor ROI:

- positive X moves scene content right;
- positive Y moves scene content down;
- positive rotation turns scene content clockwise around the sensor center.

The driver precomputes the inverse dense remap when the camera opens or the
pose changes. Streaming reuses that map and performs one `cv2.remap`; canonical
zero displacement bypasses remapping. Newly exposed pixels use BGR magenta so
synthetic coverage is visually unmistakable. Every `FrameSample` records the
pose, canonical/current 3x3 transforms, map generation, build time, and remap
time. Resetting displacement does not reset ROI or imaging state.

Use the operator control without placing a shell-sensitive URI on the Windows
command line:

```bat
virtual-hik-control.bat show --camera-source-serial RFCR91GWXLX
virtual-hik-control.bat randomize --camera-source-serial RFCR91GWXLX --seed 1729
virtual-hik-control.bat set --camera-source-serial RFCR91GWXLX --x-px 24 --y-px -12 --rotation-deg 1.5
virtual-hik-control.bat reset --camera-source-serial RFCR91GWXLX
```

Camera/lens ID, resolution, frame rate, zoom, and bit rate may be supplied with
the corresponding options. They remain part of the camera-model identity.

## Standard real-device instrument procedure

The reusable rig-only procedure calls the production rig-calibration entry
point through its public camera-plugin interface. It creates an isolated state
root and profile registry, and runs:

1. canonical fresh calibration;
2. unchanged reuse/skip;
3. seeded persistent displacement and expected recalibration;
4. unchanged displaced reuse/skip;
5. canonical reset and expected recalibration;
6. unchanged canonical reuse/skip.

Run it with the Android phone acting as camera source and the tablet acting as
the presented calibration target:

```bat
instrument-virtual-rig.bat ^
  --camera-source-serial RFCR91GWXLX ^
  --phone-serial R9JT201YLJF ^
  --adb E:\Android\Sdk\platform-tools\adb.exe
```

The procedure is fail-fast. It does not compensate for a failed stage or alter
calibration logic. Its timestamped `artifacts/virtual-rig-instrument-*` bundle
contains `instrument_run.json`, per-stage console logs and timings, native rig
evidence, and native expanded precheck reviews. Both Android displays receive
the sleep key in the final cleanup path even after a failure.

Use the real rig command with:

```bat
calibrate-hik-rig.bat ^
  --camera-adapter virtualhikcam.driver:create_camera_adapter ^
  --camera-id "virtual-hik://PHONE/camera/1?width=1280&height=720&fps=30&zoom=1&bit_rate=12000000" ^
  --phone-serial TABLET
```
