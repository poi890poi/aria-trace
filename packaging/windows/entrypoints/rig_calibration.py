"""PyInstaller entry point for HIK/Android rig calibration."""

from rig_runtime.apps.hik_rig_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
