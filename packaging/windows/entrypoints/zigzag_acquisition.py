"""PyInstaller entry point for synchronized zigzag acquisition."""

from rig_runtime.workflows.minimap_capture import main


if __name__ == "__main__":
    raise SystemExit(main())
