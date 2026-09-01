# AriaTrace native phone target

This small Android Activity is the default rig-calibration presenter. It opens
only the loopback HTTP endpoint supplied by the host through `adb reverse`,
renders the current target directly into a full-bleed `SurfaceView`, keeps the
panel awake, and posts native surface telemetry and paint acknowledgements to
the existing `/telemetry` and `/ack` endpoints.

The repository includes the signed presenter at
`android/phone-target/aria-phone-target.apk`. Rig calibration resolves and
installs this copy automatically when the app is not already installed.

Rebuild the checked-in APK after changing the Activity with:

```powershell
.\android\phone-target\build-phone-target.bat `
  -Output .\android\phone-target\aria-phone-target.apk
```

Without `-Output`, the build remains a local verification build at
`artifacts/android-phone-target/aria-phone-target.apk`. The calibrator installs
an APK only when the package is absent; an already installed compatible target
is launched directly. Set `ARIA_PHONE_TARGET_APK` or pass
`--phone-target-apk` to use another build explicitly. Standalone releases carry
the same binary at `phone-target/aria-phone-target.apk`.
