"""Pure-Python public helpers for IRIS calibration and acquisition tools.

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
    return _invoke("aria_trace.workflows.minimap_profile_calibration", argv)


def game_color_calibration(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.hik_game_color_calibration", argv)


def game_calibration(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.game_calibration", argv)


def game_repeatability(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.game_repeatability", argv)


def rig_evidence_review(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.rig_evidence_review", argv)


def profile_management(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.profile_management", argv)


def system_setup(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.workflows.system_setup", argv)


def camera_adapter_demo(argv: Optional[Sequence[str]] = None) -> int:
    return _invoke("aria_trace.apps.hik_stream", argv)


COMMANDS = {
    "rig-calibration": rig_calibration,
    "zigzag-acquisition": zigzag_acquisition,
    "minimap-calibration": minimap_calibration,
    "game-color-calibration": game_color_calibration,
    "game-calibration": game_calibration,
    "game-repeatability": game_repeatability,
    "rig-evidence-review": rig_evidence_review,
    "profiles": profile_management,
    "setup": system_setup,
    "camera-adapter-demo": camera_adapter_demo,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run IRIS tools with the current Python environment."
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
    "game_color_calibration",
    "game_calibration",
    "game_repeatability",
    "rig_evidence_review",
    "profile_management",
    "rig_calibration",
    "system_setup",
    "zigzag_acquisition",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
