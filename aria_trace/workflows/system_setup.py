"""Configure shared IRIS defaults and inspect the active profile registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from aria_trace.adapters.filesystem.profile_registry import PROFILE_KINDS, ProfileRegistry
from aria_trace.adapters.filesystem.system_configuration import (
    RIG_REPEATABILITY_POLICIES,
    load_system_configuration,
    resolve_rig_repeatability_policy,
    save_system_configuration,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Configure shared device/tool defaults under IRIS_PROFILE_ROOT"
    )
    value.add_argument("--profile-root", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="show effective root and configured defaults")
    configure = commands.add_parser("configure", help="update shared defaults")
    configure.add_argument("--camera-id")
    configure.add_argument("--phone-id")
    configure.add_argument("--game-id")
    configure.add_argument("--adb")
    configure.add_argument("--mvs-python-path")
    configure.add_argument(
        "--rig-repeatability",
        choices=tuple(RIG_REPEATABILITY_POLICIES),
        help="single policy used by both reuse skipping and GUI save protection",
    )
    profiles = commands.add_parser("profiles", help="list profile revisions")
    profiles.add_argument("--kind", choices=PROFILE_KINDS)
    profiles.add_argument("--active-only", action="store_true")
    return value


def _validate(configuration) -> None:
    resolve_rig_repeatability_policy(configuration)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "show":
        print(json.dumps(load_system_configuration(arguments.profile_root), indent=2))
        return 0
    if arguments.command == "profiles":
        registry = ProfileRegistry(arguments.profile_root)
        print(
            json.dumps(
                registry.list_revisions(
                    kind=arguments.kind, active_only=arguments.active_only
                ),
                indent=2,
            )
        )
        return 0

    configuration = load_system_configuration(arguments.profile_root)
    updates = {
        "devices": {
            "camera_id": arguments.camera_id,
            "phone_id": arguments.phone_id,
        },
        "tools": {
            "adb": arguments.adb,
            "mvs_python_path": arguments.mvs_python_path,
        },
        "game": {"game_id": arguments.game_id},
        "rig_calibration": {"repeatability_policy": arguments.rig_repeatability},
    }
    for section, values in updates.items():
        for name, value in values.items():
            if value is not None:
                configuration[section][name] = value
    _validate(configuration)
    saved = save_system_configuration(configuration, arguments.profile_root)
    print("Configured profile root: {}".format(saved["effective_profile_root"]))
    print(json.dumps(saved, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
