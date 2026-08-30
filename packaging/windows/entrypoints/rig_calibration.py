"""PyInstaller entry point for HIK/Android rig calibration."""

from aria_trace.services.calibration.rig.hik.calibrate import main


if __name__ == "__main__":
    raise SystemExit(main())
