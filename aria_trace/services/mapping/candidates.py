"""Explicitly selected tracker candidates; live defaults do not select these.

These are ordinary localizer implementations, not process-wide method patches.
Observed atlas zones remain available as evidence without vetoing image-based
mode confirmation in the visual-transition candidate.
"""

from aria_trace.services.mapping.layers import LayeredGlobalLocalizer


class VisualTransitionLocalizer(LayeredGlobalLocalizer):
    """Keep the existing XY matcher; remove spatial vetoes on mode evidence."""

    spatial_transition_gating = False
