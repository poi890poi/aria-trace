"""Stable source and PyInstaller entry point for the rig-calibration app."""

from acquisition.rig_calibration.app.qt_app import main


if __name__ == "__main__":
    raise SystemExit(main())
