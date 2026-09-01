# AriaTrace native phone target

This small Android Activity is the default rig-calibration presenter. It opens
only the loopback HTTP endpoint supplied by the host through `adb reverse`,
renders the current target directly into a full-bleed `SurfaceView`, keeps the
panel awake, and posts native surface telemetry and paint acknowledgements to
the existing `/telemetry` and `/ack` endpoints.

Build it with:

```powershell
.\android\phone-target\build-phone-target.bat
```

The APK is written to `artifacts/android-phone-target/aria-phone-target.apk`.
The calibrator installs it only when the package is absent; an already installed
compatible target is launched directly. Set `ARIA_PHONE_TARGET_APK` or pass
`--phone-target-apk` when the APK is stored elsewhere.
