"""ADB-backed Android built-in image viewer target presenter."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from ..app.phone_target import PhoneTargetAdapter, Presentation
from ..geometry import CharucoLayout, generate_charuco_target
from .phone import AdbPhoneSession


class AdbDisplayTarget(PhoneTargetAdapter):
    """Push PNG targets and show them with Android's built-in image viewer."""

    adapter_id = "android_builtin_display"

    def __init__(
        self,
        phone: AdbPhoneSession,
        component: Optional[str] = None,
        settle_seconds: float = 0.25,
        presentation_timeout_seconds: float = 8.0,
        minimum_screenshot_correlation: float = 0.98,
        minimum_matching_pixel_fraction: float = 0.995,
        minimum_stable_frame_fraction: float = 0.9995,
        minimum_ui_settle_seconds: float = 1.0,
        stable_probe_count: int = 3,
    ) -> None:
        self.phone = phone
        self.component = component
        self.settle_seconds = max(0.1, float(settle_seconds))
        self.presentation_timeout_seconds = max(
            self.settle_seconds, float(presentation_timeout_seconds)
        )
        self.minimum_screenshot_correlation = float(minimum_screenshot_correlation)
        self.minimum_matching_pixel_fraction = float(minimum_matching_pixel_fraction)
        self.minimum_stable_frame_fraction = float(minimum_stable_frame_fraction)
        self.minimum_ui_settle_seconds = max(0.0, float(minimum_ui_settle_seconds))
        self.stable_probe_count = max(1, int(stable_probe_count))
        self._layout: Optional[CharucoLayout] = None
        self._charuco: Optional[np.ndarray] = None
        self._temporary: Optional[tempfile.TemporaryDirectory] = None
        self._revision = 0
        self._remote_files = []
        self._acknowledgements = []
        self._viewer: Dict[str, Any] = {}
        self.last_target: Optional[np.ndarray] = None
        self.last_screenshot: Optional[np.ndarray] = None

    def _resolve_component(self) -> str:
        if self.component:
            return str(self.component)
        text = self.phone.shell(
            "cmd",
            "package",
            "query-activities",
            "--brief",
            "-a",
            "android.intent.action.VIEW",
            "-c",
            "android.intent.category.DEFAULT",
            "-t",
            "image/png",
        )
        components = re.findall(r"(?m)^\s*([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)\s*$", text)
        if not components:
            raise RuntimeError("Android has no activity capable of displaying image/png")
        preferred = next(
            (
                value
                for value in components
                if value.startswith(("com.sec.android.gallery3d/", "com.android.gallery3d/"))
            ),
            None,
        )
        if preferred is None and len(components) != 1:
            raise RuntimeError(
                "Multiple Android image viewers are available; pass --display-component: {}"
                .format(components)
            )
        return preferred or components[0]

    @staticmethod
    def _rotation_match(target: np.ndarray, screenshot: np.ndarray) -> Dict[str, Any]:
        candidates = [
            target,
            cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(target, cv2.ROTATE_180),
            cv2.rotate(target, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        observed = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        rows = []
        for quarter_turns, candidate in enumerate(candidates):
            gray = (
                cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
                if candidate.ndim == 3
                else candidate
            )
            fitted = cv2.resize(
                gray, (observed.shape[1], observed.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            if float(np.std(fitted)) < 1.0e-6 or float(np.std(observed)) < 1.0e-6:
                score = 1.0 - float(
                    np.mean(np.abs(fitted.astype(np.float32) - observed.astype(np.float32)))
                ) / 255.0
            else:
                score = float(np.corrcoef(fitted.reshape(-1), observed.reshape(-1))[0, 1])
            difference = np.abs(fitted.astype(np.int16) - observed.astype(np.int16))
            rows.append(
                {
                    "correlation": score,
                    "viewer_rotation_quarter_turns": quarter_turns,
                    "matching_pixel_fraction": float(np.mean(difference <= 8)),
                    "mean_absolute_error_dn": float(np.mean(difference)),
                }
            )
        return max(rows, key=lambda row: row["correlation"])

    def _capture_screenshot(self, revision: int) -> np.ndarray:
        if self._temporary is None:
            raise RuntimeError("Display target is not started")
        remote = "/sdcard/Download/aria_trace_display_probe_{}.png".format(revision)
        local = Path(self._temporary.name) / "screenshot_{}.png".format(revision)
        self.phone.shell("screencap", "-p", remote)
        self.phone.run("pull", remote, str(local))
        self.phone.shell("rm", "-f", remote)
        screenshot = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if screenshot is None or screenshot.size == 0:
            raise RuntimeError("Android Display screenshot was empty")
        return screenshot

    def _show(self, image: np.ndarray, mode: str, label: str, token: str) -> Presentation:
        if self._temporary is None or self._layout is None or not self.component:
            raise RuntimeError("Display target is not started")
        self._revision += 1
        revision = self._revision
        issued = time.monotonic_ns()
        local = Path(self._temporary.name) / "target_{}.png".format(revision)
        remote = "/sdcard/Download/aria_trace_calibration_target_{}.png".format(revision)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Cannot encode Android Display target")
        local.write_bytes(encoded.tobytes())
        self.phone.run("push", str(local), remote)
        self._remote_files.append(remote)
        self.phone.shell(
            "am",
            "start",
            "-W",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            "file://" + remote,
            "-t",
            "image/png",
            "-n",
            self.component,
        )
        self.phone.sleeper(self.settle_seconds)
        width, height = map(int, self._layout.screen_size_px)
        self.phone.shell("input", "tap", str(width // 2), str(height // 2))
        tap_time = time.monotonic()
        physical_display = self.phone.ensure_display_on(
            timeout_seconds=self.presentation_timeout_seconds
        )
        deadline = time.monotonic() + self.presentation_timeout_seconds
        screenshot = None
        match = {
            "correlation": -1.0,
            "viewer_rotation_quarter_turns": 0,
            "matching_pixel_fraction": 0.0,
            "mean_absolute_error_dn": 255.0,
        }
        probes = 0
        stable_probes = 0
        previous_screenshot = None
        stable_frame_fraction = 0.0
        while time.monotonic() < deadline:
            self.phone.sleeper(self.settle_seconds)
            screenshot = self._capture_screenshot(revision)
            probes += 1
            match = self._rotation_match(image, screenshot)
            if previous_screenshot is not None and previous_screenshot.shape == screenshot.shape:
                frame_difference = np.max(
                    np.abs(
                        previous_screenshot.astype(np.int16)
                        - screenshot.astype(np.int16)
                    ),
                    axis=2,
                )
                stable_frame_fraction = float(np.mean(frame_difference <= 2))
            else:
                stable_frame_fraction = 0.0
            previous_screenshot = screenshot.copy()
            complete = (
                match["correlation"] >= self.minimum_screenshot_correlation
                and match["matching_pixel_fraction"] >= self.minimum_matching_pixel_fraction
                and stable_frame_fraction >= self.minimum_stable_frame_fraction
                and time.monotonic() - tap_time >= self.minimum_ui_settle_seconds
            )
            stable_probes = stable_probes + 1 if complete else 0
            if stable_probes >= self.stable_probe_count:
                break
        if screenshot is None or stable_probes < self.stable_probe_count:
            self.last_target = image.copy()
            self.last_screenshot = None if screenshot is None else screenshot.copy()
            raise RuntimeError(
                "Android Display did not show target {} within {:.1f}s: "
                "last correlation {:.4f}, matching pixels {:.4%}, stable frame {:.4%}, "
                "stable probes {}/{}"
                .format(
                    label,
                    self.presentation_timeout_seconds,
                    match["correlation"],
                    match["matching_pixel_fraction"],
                    stable_frame_fraction,
                    stable_probes,
                    self.stable_probe_count,
                )
            )
        self.last_target = image.copy()
        self.last_screenshot = screenshot.copy()
        for old_remote in self._remote_files[:-1]:
            try:
                self.phone.shell("rm", "-f", old_remote)
            except Exception:
                pass
        self._remote_files = self._remote_files[-1:]
        elapsed_ms = (time.monotonic_ns() - issued) / 1.0e6
        self.phone.viewer_activity = self.component
        self._viewer = {
            "adapter_id": self.adapter_id,
            "activity": self.component,
            "canvas_width": int(screenshot.shape[1]),
            "canvas_height": int(screenshot.shape[0]),
            "fullscreen": True,
            "presentation_elapsed_ms": elapsed_ms,
            "screenshot_probe_count": probes,
            "stable_screenshot_probe_count": stable_probes,
            "required_stable_screenshot_probe_count": self.stable_probe_count,
            "stable_frame_fraction": stable_frame_fraction,
            "minimum_ui_settle_seconds": self.minimum_ui_settle_seconds,
            "physical_display": physical_display,
            **match,
        }
        self._acknowledgements.append(
            {
                "revision": revision,
                "token": token,
                "painted": True,
                "canvas_width": int(screenshot.shape[1]),
                "canvas_height": int(screenshot.shape[0]),
                "fullscreen": True,
                "presentation_elapsed_ms": elapsed_ms,
                "screenshot_probe_count": probes,
                "stable_screenshot_probe_count": stable_probes,
                "required_stable_screenshot_probe_count": self.stable_probe_count,
                "stable_frame_fraction": stable_frame_fraction,
                "minimum_ui_settle_seconds": self.minimum_ui_settle_seconds,
                "physical_display": physical_display,
                "server_receive_time_ns": time.monotonic_ns(),
                **match,
            }
        )
        return Presentation(token, mode, issued, revision)

    def start(self, layout: CharucoLayout) -> str:
        if self._temporary is not None:
            raise RuntimeError("Display target is already started")
        self._layout = layout
        self._charuco = generate_charuco_target(layout)
        self.component = self._resolve_component()
        self._temporary = tempfile.TemporaryDirectory(prefix="aria-hik-display-")
        self._show(self._charuco, "image", "ChArUco screen atlas", "charuco-initial")
        return self.component

    def present_charuco(self) -> Presentation:
        if self._charuco is None:
            raise RuntimeError("Display target is not started")
        return self._show(self._charuco, "image", "ChArUco screen atlas", "charuco")

    def present_image(self, image: np.ndarray, label: str) -> Presentation:
        return self._show(image, "image", str(label), "image-{}".format(self._revision + 1))

    def present_signal(self, state: str, token: str) -> Presentation:
        if self._layout is None or state not in ("black", "white"):
            raise ValueError("Display signal must be black or white")
        width, height = map(int, self._layout.screen_size_px)
        value = 255 if state == "white" else 0
        image = np.full((height, width, 3), value, dtype=np.uint8)
        return self._show(image, state, state, token)

    def telemetry(self) -> Mapping[str, Any]:
        return {
            "viewer": dict(self._viewer),
            "acknowledgements": list(self._acknowledgements),
        }

    def stop(self) -> None:
        for remote in self._remote_files:
            try:
                self.phone.shell("rm", "-f", remote)
            except Exception:
                pass
        self._remote_files = []
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._layout = None
        self._charuco = None
