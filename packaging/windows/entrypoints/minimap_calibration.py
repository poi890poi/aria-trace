"""PyInstaller entry point for the verified mini-map calibration backend."""

from acquisition.minimap_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
