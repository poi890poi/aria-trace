"""PyInstaller entry point for task-oriented game calibration."""

from rig_runtime.workflows.game_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
