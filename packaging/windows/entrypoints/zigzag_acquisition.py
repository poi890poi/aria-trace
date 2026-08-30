"""PyInstaller entry point for synchronized zigzag acquisition."""

from aria_trace.workflows.minimap_capture import main


if __name__ == "__main__":
    raise SystemExit(main())
