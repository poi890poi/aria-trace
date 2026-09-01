"""PyInstaller entry point for registry-managed mini-map calibration."""

from aria_trace.workflows.minimap_profile_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
