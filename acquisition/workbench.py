"""Compatibility entry point for the Workbench application.

The implementation is owned by :mod:`aria_trace.apps.workbench.application`.
This module remains only for existing imports and ``python -m`` invocations.
"""

from aria_trace.apps.workbench.application import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
