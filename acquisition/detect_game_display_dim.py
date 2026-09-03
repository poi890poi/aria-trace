"""Compatibility exports and CLI for display-state diagnosis."""

from rig_runtime.workflows.display_detection import *  # noqa: F401,F403
from rig_runtime.workflows.display_detection import _active_rig_calibration


if __name__ == "__main__":
    main()

