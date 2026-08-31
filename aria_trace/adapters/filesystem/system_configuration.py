"""Shared operator configuration stored beside the production profile registry."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .commented_yaml import write_commented_yaml
from .profile_registry import default_profile_root


SCHEMA_VERSION = "1.0"
RIG_REPEATABILITY_POLICIES = {
    "strict": {
        "reuse_max_displacement_px": 4.0,
        "reuse_sample_frames": 3,
        "save_max_displacement_px": 8.0,
        "save_movement_consecutive_frames": 1,
    },
    "balanced": {
        "reuse_max_displacement_px": 8.0,
        "reuse_sample_frames": 3,
        "save_max_displacement_px": 10.0,
        "save_movement_consecutive_frames": 2,
    },
    "relaxed": {
        "reuse_max_displacement_px": 16.0,
        "reuse_sample_frames": 3,
        "save_max_displacement_px": 12.0,
        "save_movement_consecutive_frames": 3,
    },
}
DEFAULT_RIG_REPEATABILITY_POLICY = "relaxed"

SETTINGS_HEADER = """# AriaTrace operator defaults.
#
# This file lives under the effective ARIA_PROFILE_ROOT and is shared by all
# Python commands. Command-line arguments always override these values."""

SETTINGS_COMMENTS = {
    "devices": "Default physical device identities used when a command does not specify them.",
    "tools": "Optional external executable and SDK paths; null means auto-detect.",
    "game": "Default game identity for profile selection; commands may override it.",
    "rig_calibration": (
        "One named repeatability policy owns both ChArUco geometry reuse and GUI save protection. "
        "Both displacement checks use full-sensor camera pixels; lighting is not gated."
    ),
}


def settings_paths(profile_root: Optional[Path] = None) -> Dict[str, Path]:
    root = default_profile_root(profile_root)
    directory = root / ".registry"
    return {
        "root": root,
        "json": directory / "settings.json",
        "yaml": directory / "settings.yaml",
    }


def default_system_configuration() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "devices": {"camera_id": None, "phone_id": None},
        "tools": {"adb": None, "mvs_python_path": None},
        "game": {"game_id": None},
        "rig_calibration": {
            "repeatability_policy": DEFAULT_RIG_REPEATABILITY_POLICY
        },
    }


def load_system_configuration(profile_root: Optional[Path] = None) -> Dict[str, Any]:
    paths = settings_paths(profile_root)
    result = default_system_configuration()
    if paths["json"].is_file():
        stored = json.loads(paths["json"].read_text(encoding="utf-8"))
        for section in ("devices", "tools", "game"):
            result[section].update(dict(stored.get(section) or {}))
        policy = (stored.get("rig_calibration") or {}).get("repeatability_policy")
        if policy is not None:
            result["rig_calibration"]["repeatability_policy"] = policy
        result["updated_utc"] = stored.get("updated_utc")
    result["effective_profile_root"] = str(paths["root"])
    result["profile_root_source"] = (
        "explicit"
        if profile_root is not None
        else "ARIA_PROFILE_ROOT"
        if os.environ.get("ARIA_PROFILE_ROOT")
        else "current_directory_fallback"
    )
    result["effective_rig_repeatability"] = resolve_rig_repeatability_policy(
        result
    )
    return result


def save_system_configuration(
    value: Mapping[str, Any], profile_root: Optional[Path] = None
) -> Dict[str, Any]:
    paths = settings_paths(profile_root)
    document = default_system_configuration()
    for section in ("devices", "tools", "game"):
        document[section].update(dict(value.get(section) or {}))
    policy = (value.get("rig_calibration") or {}).get("repeatability_policy")
    if policy is not None:
        document["rig_calibration"]["repeatability_policy"] = str(policy)
    document["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["json"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    last_error = None
    for attempt in range(5):
        try:
            os.replace(str(temporary), str(paths["json"]))
            break
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * float(attempt + 1))
    else:
        raise PermissionError(
            "Could not publish settings {}: {}".format(paths["json"], last_error)
        )
    write_commented_yaml(
        paths["yaml"],
        document,
        header=SETTINGS_HEADER,
        section_comments=SETTINGS_COMMENTS,
    )
    return load_system_configuration(profile_root)


def resolve_rig_repeatability_policy(
    configuration: Mapping[str, Any]
) -> Dict[str, Any]:
    name = str(
        (configuration.get("rig_calibration") or {}).get(
            "repeatability_policy", DEFAULT_RIG_REPEATABILITY_POLICY
        )
    ).lower()
    if name not in RIG_REPEATABILITY_POLICIES:
        raise ValueError(
            "Unknown rig repeatability policy {!r}; choose {}".format(
                name, ", ".join(RIG_REPEATABILITY_POLICIES)
            )
        )
    return {"name": name, **dict(RIG_REPEATABILITY_POLICIES[name])}


__all__ = [
    "DEFAULT_RIG_REPEATABILITY_POLICY",
    "RIG_REPEATABILITY_POLICIES",
    "default_system_configuration",
    "load_system_configuration",
    "resolve_rig_repeatability_policy",
    "save_system_configuration",
    "settings_paths",
]
