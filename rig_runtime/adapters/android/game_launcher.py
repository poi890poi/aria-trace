"""Resolve and launch Android games before a manual preparation checkpoint."""

from __future__ import annotations

import re
import time
from typing import Callable, Mapping, Optional, Sequence

from rig_runtime.adapters.android.phone import AdbPhoneSession


ANDROID_GAME_PACKAGE_CANDIDATES = {
    "genshin": (
        "com.miHoYo.GenshinImpact",
        "com.miHoYo.Yuanshen",
    ),
    "genshin-impact": (
        "com.miHoYo.GenshinImpact",
        "com.miHoYo.Yuanshen",
    ),
    "genshin-impact-android": (
        "com.miHoYo.GenshinImpact",
        "com.miHoYo.Yuanshen",
    ),
}


def _normalized_game_id(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def installed_packages(phone: AdbPhoneSession) -> set[str]:
    packages = set()
    for line in phone.shell("pm", "list", "packages").splitlines():
        match = re.match(r"\s*package:([^\s]+)\s*$", line)
        if match:
            packages.add(match.group(1))
    return packages


def resolve_game_package(
    game_id: str,
    available_packages: Sequence[str],
    explicit_package: Optional[str] = None,
) -> Optional[str]:
    """Choose an installed package; unknown games remain a manual workflow."""

    installed = set(map(str, available_packages))
    if explicit_package:
        selected = str(explicit_package).strip()
        if selected not in installed:
            raise RuntimeError(
                "Requested Android game package is not installed: {}".format(selected)
            )
        return selected
    candidates = ANDROID_GAME_PACKAGE_CANDIDATES.get(
        _normalized_game_id(game_id)
    )
    if candidates is None:
        return None
    selected = next((package for package in candidates if package in installed), None)
    if selected is None:
        raise RuntimeError(
            "No installed Android package matches game {!r}; checked {}".format(
                game_id, ", ".join(candidates)
            )
        )
    return selected


def foreground_component(phone: AdbPhoneSession) -> Optional[str]:
    text = phone.shell("dumpsys", "activity", "activities")
    match = re.search(
        r"(?:mResumedActivity:|topResumedActivity=).*?\s"
        r"([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)",
        text,
    )
    return match.group(1) if match else None


def _launcher_component(phone: AdbPhoneSession, package: str) -> Optional[str]:
    try:
        text = phone.shell(
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
            package,
        )
    except RuntimeError:
        return None
    matches = re.findall(
        r"({}/[A-Za-z0-9_.$]+)".format(re.escape(package)), text
    )
    return matches[-1] if matches else None


def launch_android_game(
    phone: AdbPhoneSession,
    game_id: str,
    *,
    explicit_package: Optional[str] = None,
    timeout_seconds: float = 15.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """Bring a known game to foreground and verify Android resumed it."""

    package = resolve_game_package(
        game_id, installed_packages(phone), explicit_package
    )
    result = {
        "game_id": str(game_id),
        "package": package,
        "calibration_controls_changed": False,
        "game_input_injected": False,
    }
    if package is None:
        result.update(
            {
                "status": "manual_unknown_game",
                "method": "none",
                "reason": "No package mapping exists for this game id",
            }
        )
        return result
    before = foreground_component(phone)
    result["foreground_before"] = before
    if before and before.split("/", 1)[0] == package:
        result.update(
            {
                "status": "already_foreground",
                "method": "none",
                "foreground_after": before,
            }
        )
        return result
    component = _launcher_component(phone, package)
    if component is not None:
        phone.shell(
            "am",
            "start",
            "-W",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
            "-n",
            component,
        )
        method = "resolved_launcher_activity"
    else:
        phone.shell(
            "monkey",
            "-p",
            package,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        )
        method = "package_launcher_fallback"
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    foreground = None
    while time.monotonic() < deadline:
        foreground = foreground_component(phone)
        if foreground and foreground.split("/", 1)[0] == package:
            break
        sleeper(0.25)
    if not foreground or foreground.split("/", 1)[0] != package:
        raise RuntimeError(
            "Android did not bring {} to foreground within {:.1f}s; last activity {}"
            .format(package, timeout_seconds, foreground or "unknown")
        )
    result.update(
        {
            "status": "launched",
            "method": method,
            "launcher_component": component,
            "foreground_after": foreground,
        }
    )
    return result


__all__ = [
    "ANDROID_GAME_PACKAGE_CANDIDATES",
    "foreground_component",
    "installed_packages",
    "launch_android_game",
    "resolve_game_package",
]
