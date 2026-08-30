"""Compatibility exports and CLI for the Workbench HUD application."""

from aria_trace.apps.workbench.hud import *  # noqa: F401,F403
from aria_trace.apps.workbench.hud import _HudWindow, _UrlStatusProvider


if __name__ == "__main__":
    main()
