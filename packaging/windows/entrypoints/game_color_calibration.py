"""PyInstaller entry point for synchronized HIK game-color calibration."""

from aria_trace.workflows.hik_game_color_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
