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

Use the real rig command with:

```bat
calibrate-hik-rig.bat ^
  --camera-adapter virtualhikcam.driver:create_camera_adapter ^
  --camera-id "virtual-hik://PHONE/camera/1?width=1280&height=720&fps=30&zoom=1&bit_rate=12000000" ^
  --phone-serial TABLET
```
