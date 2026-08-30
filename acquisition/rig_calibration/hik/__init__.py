"""Compatibility exports for canonical HIK rig calibration services."""

from aria_trace.services.calibration.rig.hik import *  # noqa: F401,F403
from aria_trace.adapters.hik.compat import HikCamera
from aria_trace.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from aria_trace.adapters.hik.game_camera import HikGameFrameSet, ProfiledHikGameCamera
