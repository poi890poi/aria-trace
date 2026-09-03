"""Compatibility exports for canonical HIK rig calibration services."""

from rig_runtime.services.calibration.rig.hik import *  # noqa: F401,F403
from rig_runtime.adapters.hik.compat import HikCamera
from rig_runtime.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from rig_runtime.adapters.hik.game_camera import HikGameFrameSet, ProfiledHikGameCamera
