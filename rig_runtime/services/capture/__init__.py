"""Capture scheduling services independent of device implementations."""

from .frame_pump import LatestFramePump

__all__ = ["LatestFramePump"]
