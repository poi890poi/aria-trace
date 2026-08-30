"""Map stitching, derived layers, and cross-layer reference services."""

from .layers import LayeredGlobalLocalizer, build_map_atlas
from .stitching import stitch_map_session

__all__ = ["LayeredGlobalLocalizer", "build_map_atlas", "stitch_map_session"]
