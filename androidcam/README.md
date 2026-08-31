# Android camera driver

This standalone driver streams an Android Camera2 source through the pinned
scrcpy 4.1 server, decodes H.264 with FFmpeg, and returns BGR NumPy frames. It
does not use Android screen capture and it is not a UVC device.

With more than one ADB device connected, select both sides of the facing rig:

```bat
demo-android-camera.bat --serial RFCR91GWXLX --target-serial R9JT201YLJF
```

The front camera and 1280x720 at 30 fps are the defaults. Press `Q` or `Esc`, or
close the window, to exit. The demo releases the camera first, then sends
`KEYCODE_SLEEP` to the source and optional target so their displays are off
after use. Pass `--leave-displays-on` to suppress that cleanup.

Python interface:

```python
import androidcam

with androidcam.AndroidCamera("RFCR91GWXLX", camera_facing="front") as camera:
    image = camera.get_frame()
    image, metadata = camera.get_frame_with_metadata()
```

The transport supplies Camera2/MediaCodec presentation timestamps mapped to the
host monotonic clock. Camera exposure, focus, and white-balance controls are not
exported by scrcpy; Android's `TEMPLATE_RECORD` defaults remain active.
