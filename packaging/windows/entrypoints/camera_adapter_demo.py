"""PyInstaller entry point for the standalone camera-adapter demo."""

from aria_trace.apps.hik_stream import main


if __name__ == "__main__":
    raise SystemExit(main())
