"""PyInstaller entry point for the standalone camera-adapter demo."""

from rig_runtime.apps.hik_stream import main


if __name__ == "__main__":
    raise SystemExit(main())
