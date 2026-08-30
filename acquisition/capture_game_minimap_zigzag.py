"""Compatibility exports and CLI for the mini-map capture workflow."""

from aria_trace.workflows.minimap_capture import *  # noqa: F401,F403
from aria_trace.workflows.minimap_capture import (
    _game_booster_lock_showing,
    _hik_fallback_allowed,
    _keyguard_showing,
    _launch_or_defer_game,
    _session_game_label,
    _wake_phone_for_preparation,
)


if __name__ == "__main__":
    main()
