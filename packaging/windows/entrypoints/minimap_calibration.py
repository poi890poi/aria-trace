"""PyInstaller entry point for registry-managed mini-map calibration."""

from rig_runtime.workflows.minimap_profile_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
