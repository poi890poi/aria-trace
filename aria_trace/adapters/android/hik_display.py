"""ADB-backed Android built-in image viewer target presenter."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from aria_trace.adapters.android.display import PhoneTargetAdapter, Presentation
from aria_trace.services.calibration.rig.geometry import CharucoLayout, generate_charuco_target
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
        minimum_ui_settle_seconds: float = 2.0,
        stable_probe_count: int = 3,
        strict_screenshot_verification: bool = False,
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
        self.strict_screenshot_verification = bool(strict_screenshot_verification)
        self._layout: Optional[CharucoLayout] = None
        self._charuco: Optional[np.ndarray] = None
        self._temporary: Optional[tempfile.TemporaryDirectory] = None
        self._revision = 0
        self._remote_files = []
        self._acknowledgements = []
        self._warnings = []
        self._viewer: Dict[str, Any] = {}
        self.last_target: Optional[np.ndarray] = None
        self.last_screenshot: Optional[np.ndarray] = None
        self._canonical_orientation_quarter_turns = 0
        self._configured_canonical_orientation_quarter_turns: Optional[int] = None

    def configure_canonical_orientation(self, quarter_turns: int) -> None:
        """Bind the target raster to the orientation in which it was generated."""

        if self._temporary is not None:
            raise RuntimeError(
                "Canonical orientation must be configured before Display starts"
            )
        self._configured_canonical_orientation_quarter_turns = (
            int(quarter_turns) % 4
        )

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

    def _presentation_is_stable(
        self,
        match: Mapping[str, Any],
        stable_frame_fraction: float,
        elapsed_seconds: float,
    ) -> bool:
        """Gate composition without treating fixed system pixels as failure.

        Exact matching remains diagnostic. Cutouts, rounded corners, and stable
        SystemUI pixels can lower it even when the requested raster is clearly
        composed. Physical presentation is verified later from HIK ChArUco.
        """

        return bool(
            float(match["correlation"]) >= self.minimum_screenshot_correlation
            and float(stable_frame_fraction) >= self.minimum_stable_frame_fraction
            and float(elapsed_seconds) >= self.minimum_ui_settle_seconds
        )

    def _complete_post_change_quiet_period(self, last_ui_action_time: float) -> float:
        """Always allow transient viewer/SystemUI chrome to clear after an image change."""

        remaining = max(
            0.0,
            self.minimum_ui_settle_seconds
            - max(0.0, time.monotonic() - float(last_ui_action_time)),
        )
        if remaining > 0.0:
            self.phone.sleeper(remaining)
        return float(remaining)

    @staticmethod
    def _rotate_quarter_turns_clockwise(
        image: np.ndarray, quarter_turns: int
    ) -> np.ndarray:
        turns = int(quarter_turns) % 4
        if turns == 0:
            return image.copy()
        if turns == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if turns == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _logical_target(
        self, image: np.ndarray, display_orientation_quarter_turns: int
    ) -> Tuple[np.ndarray, int]:
        rotation = (
            int(display_orientation_quarter_turns)
            - int(self._canonical_orientation_quarter_turns)
        ) % 4
        return self._rotate_quarter_turns_clockwise(image, rotation), rotation

    def _launch_target(
        self, image: np.ndarray, revision: int, orientation_attempt: int
    ) -> None:
        local = Path(self._temporary.name) / "target_{}_{}.png".format(
            revision, orientation_attempt
        )
        remote = "/sdcard/Download/aria_trace_calibration_target_{}_{}.png".format(
            revision, orientation_attempt
        )
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

    def _show(self, image: np.ndarray, mode: str, label: str, token: str) -> Presentation:
        if self._temporary is None or self._layout is None or not self.component:
            raise RuntimeError("Display target is not started")
        self._revision += 1
        revision = self._revision
        issued = time.monotonic_ns()
        physical_display = self.phone.ensure_display_on(
            timeout_seconds=self.presentation_timeout_seconds
        )
        screenshot = None
        displayed_image = image
        display_orientation = self.phone.display_orientation_quarter_turns()
        display_rotation = 0
        orientation_attempts = []
        match = {
            "correlation": -1.0,
            "viewer_rotation_quarter_turns": 0,
            "matching_pixel_fraction": 0.0,
            "mean_absolute_error_dn": 255.0,
        }
        probes = 0
        stable_probes = 0
        stable_frame_fraction = 0.0
        orientation_changed = False
        verification_error = None
        last_ui_action_time = time.monotonic()
        for orientation_attempt in range(1, 5):
            intended_orientation = int(display_orientation) % 4
            displayed_image, display_rotation = self._logical_target(
                image, intended_orientation
            )
            self._launch_target(displayed_image, revision, orientation_attempt)
            last_ui_action_time = time.monotonic()
            self.phone.sleeper(self.settle_seconds)
            try:
                screenshot = self._capture_screenshot(revision)
            except Exception as exc:
                verification_error = "{}: {}".format(type(exc).__name__, exc)
                break
            probes += 1
            observed_orientation = self.phone.display_orientation_quarter_turns()
            orientation_attempts.append(
                {
                    "attempt": orientation_attempt,
                    "intended_orientation_quarter_turns": intended_orientation,
                    "observed_orientation_quarter_turns": observed_orientation,
                    "logical_target_size_px": [
                        int(displayed_image.shape[1]),
                        int(displayed_image.shape[0]),
                    ],
                    "screenshot_size_px": [
                        int(screenshot.shape[1]),
                        int(screenshot.shape[0]),
                    ],
                }
            )
            if observed_orientation != intended_orientation:
                display_orientation = observed_orientation
                continue

            width, height = int(screenshot.shape[1]), int(screenshot.shape[0])
            self.phone.shell("input", "tap", str(width // 2), str(height // 2))
            tap_time = time.monotonic()
            last_ui_action_time = tap_time
            deadline = time.monotonic() + self.presentation_timeout_seconds
            previous_screenshot = None
            stable_probes = 0
            orientation_changed = False
            while time.monotonic() < deadline:
                self.phone.sleeper(self.settle_seconds)
                try:
                    screenshot = self._capture_screenshot(revision)
                except Exception as exc:
                    verification_error = "{}: {}".format(type(exc).__name__, exc)
                    break
                probes += 1
                observed_orientation = self.phone.display_orientation_quarter_turns()
                if observed_orientation != intended_orientation:
                    display_orientation = observed_orientation
                    orientation_changed = True
                    break
                match = self._rotation_match(displayed_image, screenshot)
                if (
                    previous_screenshot is not None
                    and previous_screenshot.shape == screenshot.shape
                ):
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
                complete = self._presentation_is_stable(
                    match,
                    stable_frame_fraction,
                    time.monotonic() - tap_time,
                )
                stable_probes = stable_probes + 1 if complete else 0
                if stable_probes >= self.stable_probe_count:
                    break
            if stable_probes >= self.stable_probe_count:
                break
            if verification_error is not None:
                break
            if not orientation_changed:
                break
        additional_ui_settle_seconds = self._complete_post_change_quiet_period(
            last_ui_action_time
        )
        presentation_verified = bool(
            screenshot is not None and stable_probes >= self.stable_probe_count
        )
        warning = None
        if not presentation_verified:
            warning = (
                "Android Display did not show target {} within {:.1f}s: "
                "last correlation {:.4f}, matching pixels {:.4%}, stable frame {:.4%}, "
                "stable probes {}/{}, orientation attempts {}, probe error {}"
                .format(
                    label,
                    self.presentation_timeout_seconds,
                    match["correlation"],
                    match["matching_pixel_fraction"],
                    stable_frame_fraction,
                    stable_probes,
                    self.stable_probe_count,
                    orientation_attempts,
                    verification_error or "none",
                )
            )
            if self.strict_screenshot_verification:
                self.last_target = displayed_image.copy()
                self.last_screenshot = (
                    None if screenshot is None else screenshot.copy()
                )
                raise RuntimeError(warning)
            self._warnings.append(warning)
            print(
                "Warning: {} Continuing; camera-observed calibration evidence "
                "remains authoritative.".format(warning),
                flush=True,
            )
        self.last_target = displayed_image.copy()
        self.last_screenshot = None if screenshot is None else screenshot.copy()
        for old_remote in self._remote_files[:-1]:
            try:
                self.phone.shell("rm", "-f", old_remote)
            except Exception:
                pass
        self._remote_files = self._remote_files[-1:]
        elapsed_ms = (time.monotonic_ns() - issued) / 1.0e6
        self.phone.viewer_activity = self.component
        canvas = screenshot if screenshot is not None else displayed_image
        self._viewer = {
            "adapter_id": self.adapter_id,
            "activity": self.component,
            "canvas_width": int(canvas.shape[1]),
            "canvas_height": int(canvas.shape[0]),
            "fullscreen": True,
            "presentation_elapsed_ms": elapsed_ms,
            "screenshot_probe_count": probes,
            "stable_screenshot_probe_count": stable_probes,
            "required_stable_screenshot_probe_count": self.stable_probe_count,
            "stable_frame_fraction": stable_frame_fraction,
            "minimum_ui_settle_seconds": self.minimum_ui_settle_seconds,
            "additional_ui_settle_seconds": additional_ui_settle_seconds,
            "screenshot_verification_strict": self.strict_screenshot_verification,
            "screenshot_presentation_verified": presentation_verified,
            "screenshot_verification_warning": warning,
            "physical_display": physical_display,
            "canonical_orientation_quarter_turns": int(
                self._canonical_orientation_quarter_turns
            ),
            "display_orientation_quarter_turns": int(display_orientation),
            "canonical_to_display_rotation_quarter_turns": int(display_rotation),
            "canonical_target_size_px": [int(image.shape[1]), int(image.shape[0])],
            "logical_target_size_px": [
                int(displayed_image.shape[1]),
                int(displayed_image.shape[0]),
            ],
            "orientation_attempts": orientation_attempts,
            **match,
        }
        self._acknowledgements.append(
            {
                "revision": revision,
                "token": token,
                "painted": True,
                "canvas_width": int(canvas.shape[1]),
                "canvas_height": int(canvas.shape[0]),
                "fullscreen": True,
                "presentation_elapsed_ms": elapsed_ms,
                "screenshot_probe_count": probes,
                "stable_screenshot_probe_count": stable_probes,
                "required_stable_screenshot_probe_count": self.stable_probe_count,
                "stable_frame_fraction": stable_frame_fraction,
                "minimum_ui_settle_seconds": self.minimum_ui_settle_seconds,
                "additional_ui_settle_seconds": additional_ui_settle_seconds,
                "screenshot_verification_strict": self.strict_screenshot_verification,
                "screenshot_presentation_verified": presentation_verified,
                "screenshot_verification_warning": warning,
                "physical_display": physical_display,
                "canonical_orientation_quarter_turns": int(
                    self._canonical_orientation_quarter_turns
                ),
                "display_orientation_quarter_turns": int(display_orientation),
                "canonical_to_display_rotation_quarter_turns": int(display_rotation),
                "canonical_target_size_px": [int(image.shape[1]), int(image.shape[0])],
                "logical_target_size_px": [
                    int(displayed_image.shape[1]),
                    int(displayed_image.shape[0]),
                ],
                "orientation_attempts": orientation_attempts,
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
        configured_orientation = self._configured_canonical_orientation_quarter_turns
        self._canonical_orientation_quarter_turns = (
            self.phone.display_orientation_quarter_turns()
            if configured_orientation is None
            else int(configured_orientation)
        )
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
            "warnings": list(self._warnings),
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
        self._canonical_orientation_quarter_turns = 0
        self._configured_canonical_orientation_quarter_turns = None
