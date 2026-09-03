"""Stable source and PyInstaller entry point for the rig-calibration app."""

from rig_runtime.apps.rig_calibrator.application import main


if __name__ == "__main__":
    raise SystemExit(main())
