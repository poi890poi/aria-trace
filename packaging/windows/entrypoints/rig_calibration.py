"""PyInstaller entry point for HIK/Android rig calibration."""

from acquisition.rig_calibration.hik.calibrate import main


if __name__ == "__main__":
    raise SystemExit(main())
