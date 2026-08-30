"""Compatibility exports for the canonical mini-map calibration service."""

from aria_trace.services.calibration.minimap.calibration import *  # noqa: F401,F403
from aria_trace.services.calibration.minimap.calibration import _validate_segments


if __name__ == "__main__":
    raise SystemExit(main())
