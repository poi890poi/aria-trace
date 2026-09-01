"""Calibrate a mini-map session and publish it through automatic configuration."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from aria_trace.adapters.filesystem.profile_registry import ProfileRegistry
from aria_trace.adapters.filesystem.system_configuration import (
    load_system_configuration,
)
from aria_trace.services.calibration.minimap.calibration import calibrate_session
from aria_trace.workflows.profile_management import publish_minimap_profiles


def _safe_label(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "game")).strip("-.")
    return cleaned or "game"


def _default_output(registry: ProfileRegistry, game_id: Optional[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        registry.root
        / "calibrations"
        / "minimap"
        / "{}-{}".format(_safe_label(game_id), timestamp)
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the verified mini-map backend, retain review evidence under the "
            "effective profile root, and publish phone-game/rig-game revisions"
        )
    )
    value.add_argument("session", type=Path, help="immutable source session")
    value.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="diagnostic evidence override; default is under IRIS_PROFILE_ROOT",
    )
    value.add_argument(
        "--rotation", nargs=2, type=float, required=True, metavar=("START", "END")
    )
    value.add_argument(
        "--movement", nargs=2, type=float, required=True, metavar=("START", "END")
    )
    value.add_argument("--config", type=Path)
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument(
        "--candidate",
        action="store_true",
        help="publish for review without activating the resulting profiles",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    registry = ProfileRegistry(arguments.profile_root)
    settings = load_system_configuration(arguments.profile_root)
    game_id = arguments.game_id or settings["game"].get("game_id")
    phone_id = settings["devices"].get("phone_id")
    output = (
        Path(arguments.output).resolve()
        if arguments.output is not None
        else _default_output(registry, game_id)
    )
    config = (
        json.loads(arguments.config.read_text(encoding="utf-8"))
        if arguments.config
        else None
    )
    calibrate_session(
        arguments.session,
        output,
        {
            "rotation_only": arguments.rotation,
            "movement_only": arguments.movement,
        },
        config,
    )
    profiles = publish_minimap_profiles(
        output / "calibration.json",
        registry=registry,
        game_id=game_id,
        phone_id=phone_id,
        camera_id=settings["devices"].get("camera_id"),
        activate=not arguments.candidate,
    )
    print("Mini-map calibration: {}".format(output.resolve()))
    print("Phone-game profile: {}".format(profiles["phone_game"]["revision_id"]))
    if profiles["rig_game"] is not None:
        print("Rig-game profile: {}".format(profiles["rig_game"]["revision_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
