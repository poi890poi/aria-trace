"""Pure-Python public helpers for the Aria Trace calibration tools.

These functions use the caller's Python environment. They do not select or
start a bundled interpreter and accept the same argument sequences as the
corresponding console applications.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional, Sequence


def _invoke(module_name: str, argv: Optional[Sequence[str]]) -> int:
    module = importlib.import_module(module_name)
    return int(module.main(argv))


def rig_calibration(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.apps.hik_rig_calibration", argv)


def zigzag_acquisition(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.minimap_capture", argv)


def minimap_calibration(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("acquisition.minimap_calibration", argv)


def camera_adapter_demo(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.apps.hik_stream", argv)


COMMANDS = {
    "rig-calibration": rig_calibration,
    "zigzag-acquisition": zigzag_acquisition,
    "minimap-calibration": minimap_calibration,
    "camera-adapter-demo": camera_adapter_demo,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run Aria Trace tools with the current Python environment."
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    if not values or values[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    forwarding_parser = argparse.ArgumentParser(add_help=False)
    forwarding_parser.add_argument("command", choices=tuple(COMMANDS))
    arguments, forwarded = forwarding_parser.parse_known_args(values)
    return COMMANDS[arguments.command](forwarded)


__all__ = [
    "camera_adapter_demo",
    "main",
    "minimap_calibration",
    "rig_calibration",
    "zigzag_acquisition",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
