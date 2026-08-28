"""ADB phone lifecycle used by the standalone HIK rig calibrator."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


Runner = Callable[[Sequence[str], float], str]


def resolve_adb_executable(value: Optional[str] = None) -> str:
    """Find ADB from an explicit value, PATH, or common Android SDK locations."""

    if value and str(value) != "adb":
        path = Path(str(value))
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(str(value))
        if resolved:
            return resolved
        raise FileNotFoundError("ADB executable does not exist: {}".format(value))
    resolved = shutil.which("adb")
    if resolved:
        return resolved
    candidates = []
    for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]) / "platform-tools" / "adb.exe")
    drives = {Path.cwd().drive or "C:", "C:"}
    for drive in sorted(drives):
        candidates.append(Path(drive + "\\Android\\Sdk\\platform-tools\\adb.exe"))
    candidates.append(Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "ADB was not found in PATH or a standard Android SDK location; pass --adb"
    )


def connected_adb_devices(adb_executable: Optional[str] = None) -> List[str]:
    adb = resolve_adb_executable(adb_executable)
    text = _subprocess_runner([adb, "devices"], 10.0)
    return [
        line.split("\t", 1)[0].strip()
        for line in text.splitlines()[1:]
        if "\tdevice" in line and line.split("\t", 1)[0].strip()
    ]


def _subprocess_runner(command: Sequence[str], timeout_seconds: float) -> str:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=float(timeout_seconds),
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ADB command failed ({}): {}".format(
                completed.returncode, completed.stderr.strip() or completed.stdout.strip()
            )
        )
    return completed.stdout.strip()


def _first_pair(text: str) -> Optional[List[int]]:
    match = re.search(r"(\d+)\s*x\s*(\d+)", text)
    return [int(match.group(1)), int(match.group(2))] if match else None


@dataclass(frozen=True)
class PhoneMetrics:
    serial: str
    manufacturer: str
    model: str
    android_version: str
    screen_size_px: List[int]
    density_dpi: int
    refresh_hz: float
    orientation_quarter_turns: int = 0
    natural_screen_size_px: Optional[List[int]] = None
    active_app_size_px: Optional[List[int]] = None
    display_state: str = "unknown"
    physical_dpi_xy: Optional[List[float]] = None
    physical_scale_source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        pitch = (
            [25.4 / float(value) for value in self.physical_dpi_xy]
            if self.physical_dpi_xy
            else None
        )
        return {
            "serial": self.serial,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "android_version": self.android_version,
            "screen_size_px": list(self.screen_size_px),
            "density_dpi": self.density_dpi,
            "logical_density_dpi": self.density_dpi,
            "physical_dpi_xy": (
                list(map(float, self.physical_dpi_xy)) if self.physical_dpi_xy else None
            ),
            "physical_pixel_pitch_mm_xy": pitch,
            "physical_size_mm": (
                [
                    float(self.screen_size_px[0]) * pitch[0],
                    float(self.screen_size_px[1]) * pitch[1],
                ]
                if pitch
                else None
            ),
            "physical_size_source": self.physical_scale_source,
            "refresh_hz": self.refresh_hz,
            "orientation_quarter_turns": self.orientation_quarter_turns,
            "orientation_degrees": self.orientation_quarter_turns * 90,
            "orientation_name": (
                "portrait"
                if self.orientation_quarter_turns == 0
                else "landscape"
                if self.orientation_quarter_turns == 1
                else "reverse_portrait"
                if self.orientation_quarter_turns == 2
                else "reverse_landscape"
            ),
            "natural_screen_size_px": list(
                self.natural_screen_size_px or self.screen_size_px
            ),
            "active_app_size_px": (
                list(self.active_app_size_px) if self.active_app_size_px else None
            ),
            "display_state": self.display_state,
        }


class AdbPhoneSession:
    """Wake, hold awake, present the target, then restore and sleep one phone.

    The session snapshots every setting it mutates. Cleanup is idempotent and
    intended to run from a ``finally`` block even after calibration failure.
    """

    def __init__(
        self,
        serial: str,
        adb_executable: str = "adb",
        runner: Optional[Runner] = None,
        timeout_seconds: float = 10.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not str(serial).strip():
            raise ValueError("An explicit ADB serial is required")
        self.serial = str(serial)
        self.adb_executable = (
            str(adb_executable) if runner is not None else resolve_adb_executable(adb_executable)
        )
        self.runner = runner or _subprocess_runner
        self.timeout_seconds = float(timeout_seconds)
        self.sleeper = sleeper
        self._saved: Dict[str, Optional[str]] = {}
        self._reverse_port: Optional[int] = None
        self._active = False
        self.viewer_activity: Optional[str] = None

    def _base(self) -> List[str]:
        return [self.adb_executable, "-s", self.serial]

    def run(self, *args: str) -> str:
        return self.runner(self._base() + [str(value) for value in args], self.timeout_seconds)

    def shell(self, *args: str) -> str:
        return self.run("shell", *args)

    def display_state(self) -> str:
        """Return the physical/default display power state reported by Android."""

        display_text = self.shell("dumpsys", "display")
        patterns = (
            r"mOverrideDisplayInfo=.*?\bstate\s+(ON|OFF|DOZE|DOZE_SUSPEND)",
            r"mBaseDisplayInfo=.*?\bstate\s+(ON|OFF|DOZE|DOZE_SUSPEND)",
            r"DisplayDeviceInfo.*?\bstate\s+(ON|OFF|DOZE|DOZE_SUSPEND)",
        )
        for pattern in patterns:
            match = re.search(pattern, display_text, re.DOTALL)
            if match:
                return match.group(1)
        try:
            power_text = self.shell("dumpsys", "power")
        except RuntimeError:
            power_text = ""
        match = re.search(
            r"Display Power:\s*state=(ON|OFF|DOZE|DOZE_SUSPEND)",
            power_text,
            re.IGNORECASE,
        )
        return match.group(1).upper() if match else "unknown"

    def ensure_display_on(self, timeout_seconds: float = 5.0) -> Dict[str, Any]:
        """Wake and verify the physical panel; a screenshot is not sufficient proof."""

        started = time.monotonic_ns()
        deadline = time.monotonic() + max(0.5, float(timeout_seconds))
        probes = 0
        last_state = "unknown"
        next_wakeup = 0.0
        while time.monotonic() < deadline:
            probes += 1
            last_state = self.display_state()
            if last_state == "ON":
                return {
                    "state": "ON",
                    "probes": probes,
                    "elapsed_ms": (time.monotonic_ns() - started) / 1.0e6,
                }
            now = time.monotonic()
            if now >= next_wakeup:
                self.shell("input", "keyevent", "KEYCODE_WAKEUP")
                self.shell("wm", "dismiss-keyguard")
                next_wakeup = now + 0.75
            self.sleeper(0.2)
        raise RuntimeError(
            "Android physical display did not report ON within {:.1f}s; last state {}"
            .format(timeout_seconds, last_state)
        )

    def _get_setting(self, namespace: str, name: str) -> Optional[str]:
        value = self.shell("settings", "get", namespace, name).strip()
        return None if value in ("", "null") else value

    def _restore_setting(self, namespace: str, name: str, value: Optional[str]) -> None:
        if value is None:
            self.shell("settings", "delete", namespace, name)
        else:
            self.shell("settings", "put", namespace, name, value)

    def display_brightness_state(self) -> Dict[str, Any]:
        mode = self._get_setting("system", "screen_brightness_mode")
        value = self._get_setting("system", "screen_brightness")
        return {
            "mode": "automatic" if mode == "1" else "manual" if mode == "0" else "unknown",
            "mode_value": mode,
            "brightness_value": int(value) if value is not None and value.isdigit() else None,
            "declared_maximum": 255,
        }

    def metrics(self, refresh_hz_override: Optional[float] = None) -> PhoneMetrics:
        size_text = self.shell("wm", "size")
        override_size = re.search(r"Override size:\s*(\d+)\s*x\s*(\d+)", size_text, re.IGNORECASE)
        natural_size = [int(override_size.group(1)), int(override_size.group(2))] if override_size else _first_pair(size_text)
        if not natural_size:
            raise RuntimeError("ADB did not report a physical phone display size")
        try:
            display_text = self.shell("dumpsys", "display")
        except RuntimeError:
            display_text = ""
        try:
            orientation_text = self.shell("dumpsys", "input")
        except RuntimeError:
            orientation_text = ""
        orientation_match = re.search(r"SurfaceOrientation:\s*([0-3])", orientation_text)
        if orientation_match is None:
            orientation_match = re.search(r"mCurrentOrientation=([0-3])", display_text)
        if orientation_match is None:
            orientation_match = re.search(r"\brotation\s+([0-3])\b", display_text)
        orientation = int(orientation_match.group(1)) if orientation_match else 0
        size = list(natural_size)
        if orientation % 2:
            size = [size[1], size[0]]
        physical_dpi = None
        active_dpi = re.search(
            r"mActiveSfDisplayMode=.*?xDpi=([0-9]+(?:\.[0-9]+)?),\s*"
            r"yDpi=([0-9]+(?:\.[0-9]+)?)",
            display_text,
            re.DOTALL,
        )
        if active_dpi is None:
            active_dpi = re.search(
                r"density\s+\d+(?:\.\d+)?\s*\("
                r"([0-9]+(?:\.[0-9]+)?)\s*x\s*"
                r"([0-9]+(?:\.[0-9]+)?)\)\s*dpi",
                display_text,
                re.IGNORECASE,
            )
        if active_dpi:
            physical_dpi = [float(active_dpi.group(1)), float(active_dpi.group(2))]
            if not all(50.0 <= value <= 2000.0 for value in physical_dpi):
                physical_dpi = None
            elif orientation % 2:
                physical_dpi.reverse()
        app_size = None
        override_info = re.search(
            r"mOverrideDisplayInfo=.*?\bapp\s+(\d+)\s*x\s*(\d+)",
            display_text,
            re.DOTALL,
        )
        if override_info:
            app_size = [int(override_info.group(1)), int(override_info.group(2))]
        state_match = re.search(
            r"mOverrideDisplayInfo=.*?\bstate\s+(ON|OFF|DOZE|DOZE_SUSPEND)",
            display_text,
            re.DOTALL,
        )
        density_text = self.shell("wm", "density")
        densities = re.findall(r"(\d+)", density_text)
        if not densities:
            raise RuntimeError("ADB did not report display density")
        refresh = refresh_hz_override
        if refresh is None:
            try:
                latency = self.shell("dumpsys", "SurfaceFlinger", "--latency")
            except RuntimeError:
                latency = ""
            period_match = re.search(r"^\s*(\d{6,9})\s*$", latency, re.MULTILINE)
            if period_match:
                refresh = 1.0e9 / float(period_match.group(1))
        if refresh is None:
            active_mode = re.search(
                r"mActiveSfDisplayMode=.*?refreshRate=([0-9]+(?:\.[0-9]+)?)",
                display_text,
            )
            if active_mode:
                refresh = float(active_mode.group(1))
        if refresh is None:
            candidates = [
                float(value)
                for value in re.findall(
                    r"(?:refreshRate|fps|peakRefreshRate)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
                    display_text,
                    flags=re.IGNORECASE,
                )
                if 20.0 <= float(value) <= 500.0
            ]
            if candidates:
                refresh = candidates[0]
        if refresh is None or float(refresh) <= 0:
            raise RuntimeError("Cannot determine panel refresh rate; pass --refresh-hz")
        return PhoneMetrics(
            serial=self.serial,
            manufacturer=self.shell("getprop", "ro.product.manufacturer"),
            model=self.shell("getprop", "ro.product.model"),
            android_version=self.shell("getprop", "ro.build.version.release"),
            screen_size_px=size,
            density_dpi=int(densities[-1]),
            refresh_hz=float(refresh),
            orientation_quarter_turns=orientation,
            natural_screen_size_px=list(natural_size),
            active_app_size_px=app_size,
            display_state=(state_match.group(1) if state_match else "unknown"),
            physical_dpi_xy=physical_dpi,
            physical_scale_source=(
                "android_active_display_mode_xdpi_ydpi" if physical_dpi else "unknown"
            ),
        )

    def wake_and_hold(
        self,
        local_target_port: int,
        screen_size_px: Sequence[int],
        rotation_quarter_turns: int = 0,
    ) -> None:
        self.wake_and_hold_display(rotation_quarter_turns)
        try:
            port = int(local_target_port)
            self.run("reverse", "tcp:{}".format(port), "tcp:{}".format(port))
            self._reverse_port = port
            resolved = self.shell(
                "cmd",
                "package",
                "resolve-activity",
                "--brief",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                "http://127.0.0.1:{}/".format(port),
            )
            self.viewer_activity = resolved.splitlines()[-1].strip() if resolved else None
            self.shell(
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                "http://127.0.0.1:{}/?autostart=1".format(port),
            )
            width, height = map(int, screen_size_px)
            self.sleeper(1.0)
            # A real ADB input event is a trusted user gesture, allowing the
            # target button to enter browser fullscreen where scripted clicks cannot.
            self.request_fullscreen([width, height])
        except Exception:
            self.cleanup(turn_display_off=True)
            raise

    def wake_and_hold_display(self, rotation_quarter_turns: int = 0) -> None:
        """Wake and lock the display without choosing or launching a viewer."""

        if self._active:
            raise RuntimeError("Phone session is already active")
        keys = (
            ("system", "screen_off_timeout"),
            ("system", "screen_brightness_mode"),
            ("system", "screen_brightness"),
            ("global", "stay_on_while_plugged_in"),
            ("global", "policy_control"),
            ("system", "accelerometer_rotation"),
            ("system", "user_rotation"),
        )
        self._saved = {
            "{}:{}".format(namespace, name): self._get_setting(namespace, name)
            for namespace, name in keys
        }
        try:
            self.shell("input", "keyevent", "KEYCODE_WAKEUP")
            self.shell("wm", "dismiss-keyguard")
            self.shell("settings", "put", "system", "screen_off_timeout", "2147483647")
            self.shell("settings", "put", "system", "screen_brightness_mode", "0")
            self.shell("settings", "put", "system", "screen_brightness", "255")
            self.shell("settings", "put", "global", "stay_on_while_plugged_in", "3")
            self.shell("settings", "put", "global", "policy_control", "immersive.full=*")
            self.shell("settings", "put", "system", "accelerometer_rotation", "0")
            self.shell(
                "settings",
                "put",
                "system",
                "user_rotation",
                str(int(rotation_quarter_turns) % 4),
            )
            self._active = True
            self.ensure_display_on()
            # Panel output and camera auto loops both respond over multiple frames.
            self.sleeper(0.75)
        except Exception:
            self.cleanup(turn_display_off=True)
            raise

    def request_fullscreen(
        self,
        screen_size_px: Sequence[int],
        viewport_size_px: Optional[Sequence[int]] = None,
    ) -> None:
        """Tap the center fullscreen gate using a trusted Android input event."""

        width, height = map(int, screen_size_px)
        viewport_height = (
            int(viewport_size_px[1]) if viewport_size_px is not None else height
        )
        # Chrome places its non-fullscreen viewport below the address bar. Its
        # excluded pixels are therefore above the document on the tested rig.
        center_y = height - max(1, viewport_height) // 2
        self.shell("input", "tap", str(width // 2), str(center_y))

    def cleanup(self, turn_display_off: bool = True) -> None:
        errors = []
        managed_display = self._active or bool(self._saved) or self._reverse_port is not None
        if self._reverse_port is not None:
            try:
                self.run("reverse", "--remove", "tcp:{}".format(self._reverse_port))
            except Exception as exc:
                errors.append(str(exc))
            self._reverse_port = None
        for key, value in self._saved.items():
            namespace, name = key.split(":", 1)
            try:
                self._restore_setting(namespace, name, value)
            except Exception as exc:
                errors.append(str(exc))
        self._saved = {}
        if turn_display_off and managed_display:
            try:
                self.shell("input", "keyevent", "KEYCODE_SLEEP")
            except Exception as exc:
                errors.append(str(exc))
        self._active = False
        if errors:
            raise RuntimeError("Phone cleanup was incomplete: {}".format("; ".join(errors)))

    def __enter__(self) -> "AdbPhoneSession":
        return self

    def __exit__(self, *_exc) -> None:
        self.cleanup(turn_display_off=True)
