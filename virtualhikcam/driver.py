"""Persistent Android-camera virtual implementation of the HIK adapter API.

The adapter is intended for exercising the real IRIS rig pipeline without
claiming physical HIK behavior.  Android Camera2 supplies full decoded frames;
zoom and ROI are deterministic software operations whose spaces are explicit in
every FrameSample.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse

import cv2
import numpy as np

from androidcam.driver import AndroidCamera
from rig_runtime.adapters.rig.devices import (
    CameraAdapter,
    CameraConfiguration,
    CameraDevice,
)
from rig_runtime.services.calibration.rig.contracts import FrameSample


STATE_SCHEMA_VERSION = 2
ADAPTER_ID = "virtual_hik_android_camera"
DISPLACEMENT_BORDER_BGR = (255, 0, 255)


def _state_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    configured = os.environ.get("IRIS_VIRTUAL_HIK_STATE_ROOT")
    if configured:
        return Path(configured).resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return (Path(local) / "IRIS" / "virtual-cameras").resolve()
    return (Path.home() / ".iris" / "virtual-cameras").resolve()


def _number(query: Mapping[str, Sequence[str]], name: str, default: float) -> float:
    values = query.get(name)
    return float(values[-1]) if values else float(default)


def _integer(query: Mapping[str, Sequence[str]], name: str, default: int) -> int:
    return int(round(_number(query, name, default)))


def _camera_spec(device_id: str, configuration: CameraConfiguration) -> dict[str, Any]:
    parsed = urlparse(str(device_id))
    if parsed.scheme not in ("androidcam", "virtual-hik"):
        raise ValueError(
            "Virtual HIK device ID must use androidcam:// or virtual-hik://"
        )
    serial = parsed.netloc.strip()
    if not serial:
        raise ValueError("Virtual HIK device ID requires an Android serial")
    query = parse_qs(parsed.query)
    parts = [part for part in parsed.path.split("/") if part]
    camera_id = None
    if len(parts) >= 2 and parts[0] == "camera":
        camera_id = parts[1]
    elif parts:
        camera_id = parts[-1]
    facing = str((query.get("facing") or ["front"])[-1]).lower()
    if camera_id is None and facing not in ("front", "back", "external"):
        raise ValueError("Virtual camera facing must be front, back, or external")
    width = _integer(query, "width", configuration.width_px)
    height = _integer(query, "height", configuration.height_px)
    fps = _integer(query, "fps", int(round(configuration.fps)))
    zoom = _number(query, "zoom", 1.0)
    bit_rate = _integer(query, "bit_rate", 12_000_000)
    if min(width, height, fps, bit_rate) <= 0 or zoom < 1.0:
        raise ValueError("Virtual camera mode and zoom must be positive")
    path = "/camera/{}".format(camera_id) if camera_id is not None else "/facing/{}".format(facing)
    canonical = "virtual-hik://{}{}?{}".format(
        serial,
        path,
        urlencode(
            [
                ("width", width),
                ("height", height),
                ("fps", fps),
                ("zoom", "{:.6g}".format(zoom)),
                ("bit_rate", bit_rate),
            ]
        ),
    )
    return {
        "serial": serial,
        "camera_id": camera_id,
        "camera_facing": facing,
        "width_px": width,
        "height_px": height,
        "fps": fps,
        "zoom": zoom,
        "bit_rate": bit_rate,
        "canonical_device_id": canonical,
    }


class _DeviceLease:
    """One-byte OS lock so concurrent opens fail like a claimed camera."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            handle.close()
            raise RuntimeError("Virtual camera is already open: {}".format(self.path.parent.name))
        self.handle = handle

    def release(self) -> None:
        handle, self.handle = self.handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class VirtualHikCameraAdapter(CameraAdapter):
    """HIK-compatible virtual driver backed by one Android Camera2 source."""

    adapter_id = ADAPTER_ID

    def __init__(
        self,
        *,
        state_root: Optional[Path] = None,
        source_factory: Callable[..., AndroidCamera] = AndroidCamera,
        sleep_source_on_close: bool = True,
    ) -> None:
        self.state_root = _state_root(state_root)
        self.source_factory = source_factory
        self.sleep_source_on_close = bool(sleep_source_on_close)
        self.configuration: Optional[CameraConfiguration] = None
        self.metadata: dict[str, Any] = {}
        self._spec: Optional[dict[str, Any]] = None
        self._source = None
        self._full_size: Optional[list[int]] = None
        self._state: dict[str, Any] = {}
        self._state_path: Optional[Path] = None
        self._lease: Optional[_DeviceLease] = None
        self._last_frame_metadata: dict[str, Any] = {}
        self._once_frames_remaining = 0
        self._displacement_map_x: Optional[np.ndarray] = None
        self._displacement_map_y: Optional[np.ndarray] = None
        self._displacement_forward_3x3 = np.eye(3, dtype=np.float64)
        self._displacement_inverse_3x3 = np.eye(3, dtype=np.float64)
        self._displacement_map_build_ms = 0.0
        self._displacement_map_generation = 0

    def devices(self, probe: bool = False) -> Sequence[CameraDevice]:
        # Camera2 enumeration is intentionally not performed implicitly.  A
        # complete virtual model URI is the selectable device identity.
        return ()

    @staticmethod
    def _identity_directory(root: Path, canonical_device_id: str) -> Path:
        digest = hashlib.sha256(canonical_device_id.encode("utf-8")).hexdigest()[:20]
        return Path(root) / digest

    @property
    def full_size(self) -> list[int]:
        if self._full_size is None:
            raise RuntimeError("Virtual camera is not open")
        return list(self._full_size)

    @property
    def active_roi(self) -> list[int]:
        if not self._state:
            raise RuntimeError("Virtual camera is not open")
        return list(map(int, self._state["effective_roi_xywh"]))

    def _default_state(self, model_id: str, full_size: Sequence[int]) -> dict[str, Any]:
        width, height = map(int, full_size)
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "adapter_id": self.adapter_id,
            "camera_model_id": str(model_id),
            "sensor_size_px": [width, height],
            "effective_roi_xywh": [0, 0, width, height],
            "state_generation": 0,
            "imaging": {
                "exposure_us": 8333.0,
                "gain": 0.0,
                "ratio_red": 1024,
                "ratio_green": 1024,
                "ratio_blue": 1024,
                "black_level": 0,
                "gamma": 1.0,
                "ccm_rgb_3x3": np.eye(3, dtype=float).tolist(),
                "exposure_auto": "Continuous",
                "gain_auto": "Continuous",
                "balance_white_auto": "Continuous",
            },
            "auto_limits": {"maximum_exposure_us": 33333.0, "maximum_gain": 24.0},
            "auto_function_roi_xywh": [0, 0, width, height],
            "simulated_displacement": {
                "x_px": 0.0,
                "y_px": 0.0,
                "rotation_deg_clockwise": 0.0,
                "mode": "canonical",
                "seed": None,
            },
        }

    @staticmethod
    def _validated_displacement(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        value = dict(value or {})
        x_px = float(value.get("x_px", 0.0))
        y_px = float(value.get("y_px", 0.0))
        rotation = float(value.get("rotation_deg_clockwise", 0.0))
        if not np.all(np.isfinite([x_px, y_px, rotation])):
            raise ValueError("Virtual displacement must contain finite values")
        return {
            "x_px": x_px,
            "y_px": y_px,
            "rotation_deg_clockwise": rotation,
            "mode": str(value.get("mode") or (
                "canonical" if max(abs(x_px), abs(y_px), abs(rotation)) <= 1.0e-12
                else "explicit"
            )),
            "seed": value.get("seed"),
        }

    def _load_state(self, model_id: str, full_size: Sequence[int]) -> dict[str, Any]:
        default = self._default_state(model_id, full_size)
        if self._state_path is None or not self._state_path.is_file():
            return default
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
            schema_version = int(value.get("schema_version", -1))
            if schema_version not in (1, STATE_SCHEMA_VERSION):
                raise ValueError("unsupported state schema")
            if str(value.get("camera_model_id")) != str(model_id):
                raise ValueError("camera model identity changed")
            if list(map(int, value.get("sensor_size_px") or [])) != list(map(int, full_size)):
                raise ValueError("effective sensor mode changed")
            roi = self._align_roi(value.get("effective_roi_xywh"), full_size)
            value["effective_roi_xywh"] = roi
            value["simulated_displacement"] = self._validated_displacement(
                value.get("simulated_displacement")
            )
            value["schema_version"] = STATE_SCHEMA_VERSION
            return value
        except Exception as exc:
            default["state_recovery_warning"] = "{}: {}".format(type(exc).__name__, exc)
            return default

    def _persist(self) -> None:
        if self._state_path is None:
            raise RuntimeError("Virtual camera state path is unavailable")
        self._state["updated_unix_time"] = time.time()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".json.tmp-{}".format(os.getpid()))
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(str(temporary), str(self._state_path))

    def _mutate(self, update: Callable[[dict[str, Any]], None]) -> None:
        update(self._state)
        self._state["state_generation"] = int(self._state.get("state_generation", 0)) + 1
        self._persist()

    def _rebuild_displacement_map(self) -> None:
        """Precompute the inverse dense map for the persistent virtual pose."""

        width, height = self.full_size
        displacement = self._validated_displacement(
            self._state.get("simulated_displacement")
        )
        x_px = float(displacement["x_px"])
        y_px = float(displacement["y_px"])
        angle = np.deg2rad(float(displacement["rotation_deg_clockwise"]))
        center_x = (float(width) - 1.0) / 2.0
        center_y = (float(height) - 1.0) / 2.0
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        to_origin = np.asarray(
            [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        from_origin = np.asarray(
            [
                [1.0, 0.0, center_x + x_px],
                [0.0, 1.0, center_y + y_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        forward = from_origin.dot(rotation).dot(to_origin)
        inverse = np.linalg.inv(forward)
        started = time.perf_counter_ns()
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        map_x = (
            inverse[0, 0] * grid_x
            + inverse[0, 1] * grid_y
            + inverse[0, 2]
        ).astype(np.float32)
        map_y = (
            inverse[1, 0] * grid_x
            + inverse[1, 1] * grid_y
            + inverse[1, 2]
        ).astype(np.float32)
        completed = time.perf_counter_ns()
        self._state["simulated_displacement"] = displacement
        self._displacement_forward_3x3 = forward
        self._displacement_inverse_3x3 = inverse
        self._displacement_map_x = map_x
        self._displacement_map_y = map_y
        self._displacement_map_build_ms = (completed - started) / 1.0e6
        self._displacement_map_generation += 1

    def simulated_displacement(self) -> Mapping[str, Any]:
        if not self._state:
            raise RuntimeError("Virtual camera is not open")
        return copy.deepcopy(self._state["simulated_displacement"])

    def set_simulated_displacement(
        self,
        x_px: float,
        y_px: float,
        rotation_deg_clockwise: float,
        *,
        mode: str = "explicit",
        seed: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Persist one physical-pose simulation in full-sensor coordinates.

        Positive X moves scene content right, positive Y moves it down, and a
        positive angle rotates it clockwise around the sensor center.
        """

        displacement = self._validated_displacement(
            {
                "x_px": x_px,
                "y_px": y_px,
                "rotation_deg_clockwise": rotation_deg_clockwise,
                "mode": mode,
                "seed": seed,
            }
        )
        width, height = self.full_size
        if abs(displacement["x_px"]) > width * 0.1:
            raise ValueError("Virtual X displacement is limited to 10% of sensor width")
        if abs(displacement["y_px"]) > height * 0.1:
            raise ValueError("Virtual Y displacement is limited to 10% of sensor height")
        if abs(displacement["rotation_deg_clockwise"]) > 10.0:
            raise ValueError("Virtual rotation displacement is limited to 10 degrees")
        self._mutate(
            lambda state: state.update(simulated_displacement=copy.deepcopy(displacement))
        )
        self._rebuild_displacement_map()
        self.metadata["state_generation"] = int(self._state["state_generation"])
        self.metadata["simulated_displacement"] = self.simulated_displacement()
        self.metadata["displacement_map_build_ms"] = self._displacement_map_build_ms
        return self.simulated_displacement()

    def randomize_simulated_displacement(
        self,
        *,
        max_x_px: Optional[float] = None,
        max_y_px: Optional[float] = None,
        max_rotation_deg: float = 2.0,
        seed: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Generate and persist a small repeatable random physical displacement."""

        width, height = self.full_size
        max_x = float(max_x_px if max_x_px is not None else width * 0.02)
        max_y = float(max_y_px if max_y_px is not None else height * 0.02)
        max_rotation = float(max_rotation_deg)
        if min(max_x, max_y, max_rotation) < 0.0:
            raise ValueError("Random displacement limits must be non-negative")
        generator = random.Random(seed)
        return self.set_simulated_displacement(
            generator.uniform(-max_x, max_x),
            generator.uniform(-max_y, max_y),
            generator.uniform(-max_rotation, max_rotation),
            mode="random",
            seed=seed,
        )

    def reset_simulated_displacement(self) -> Mapping[str, Any]:
        """Return content to canonical pose without changing ROI or imaging state."""

        return self.set_simulated_displacement(
            0.0, 0.0, 0.0, mode="canonical", seed=None
        )

    def open(self, configuration: CameraConfiguration) -> Mapping[str, Any]:
        self.close()
        spec = _camera_spec(configuration.device_id, configuration)
        directory = self._identity_directory(self.state_root, spec["canonical_device_id"])
        lease = _DeviceLease(directory / "device.lock")
        lease.acquire()
        source = None
        try:
            source = self.source_factory(
                spec["serial"],
                camera_id=spec["camera_id"],
                camera_facing=spec["camera_facing"],
                width_px=spec["width_px"],
                height_px=spec["height_px"],
                fps=spec["fps"],
                bit_rate=spec["bit_rate"],
            )
            source.open()
            effective = dict(source.effective_configuration)
            full_size = [
                int(effective["effective"]["width_px"]),
                int(effective["effective"]["height_px"]),
            ]
            self._state_path = directory / "state.json"
            self._state = self._load_state(spec["canonical_device_id"], full_size)
            self._persist()
            self._source = source
            self._lease = lease
            self._spec = spec
            self._full_size = full_size
            self.configuration = configuration
            self._rebuild_displacement_map()
            self.metadata = {
                "adapter_id": self.adapter_id,
                "device_id": spec["canonical_device_id"],
                "transport": "Android Camera2 via scrcpy",
                "model": "Android camera {} zoom {} at {}x{}".format(
                    spec["camera_id"] if spec["camera_id"] is not None else spec["camera_facing"],
                    spec["zoom"],
                    full_size[0],
                    full_size[1],
                ),
                "serial": spec["serial"],
                "camera_id": spec["camera_id"],
                "camera_facing": spec["camera_facing"],
                "width_px": full_size[0],
                "height_px": full_size[1],
                "fps": float(spec["fps"]),
                "zoom": float(spec["zoom"]),
                "virtual_camera": True,
                "roi_implementation": "software_crop_after_full_frame_decode",
                "zoom_implementation": (
                    "identity" if float(spec["zoom"]) == 1.0 else "software_center_crop_resize"
                ),
                "state_file": str(self._state_path),
                "state_generation": int(self._state["state_generation"]),
                "simulated_displacement": self.simulated_displacement(),
                "displacement_map_build_ms": self._displacement_map_build_ms,
                "physical_hik_claims": "none",
            }
            return copy.deepcopy(self.metadata)
        except Exception:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            lease.release()
            self._state_path = None
            self._state = {}
            raise

    @staticmethod
    def _zoom(image: np.ndarray, zoom: float) -> np.ndarray:
        if zoom <= 1.0 + 1.0e-9:
            return image
        height, width = image.shape[:2]
        crop_width = max(2, int(round(width / zoom)))
        crop_height = max(2, int(round(height / zoom)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        crop = image[top : top + crop_height, left : left + crop_width]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    def read(self) -> FrameSample:
        if self._source is None or self._spec is None:
            raise RuntimeError("Virtual camera is not open")
        started = time.perf_counter_ns()
        source = self._source.read_sample()
        full = self._zoom(source.image, float(self._spec["zoom"]))
        if [int(full.shape[1]), int(full.shape[0])] != self.full_size:
            raise RuntimeError("Virtual camera full raster changed during acquisition")
        displacement_started = time.perf_counter_ns()
        displacement = self.simulated_displacement()
        displacement_active = max(
            abs(float(displacement["x_px"])),
            abs(float(displacement["y_px"])),
            abs(float(displacement["rotation_deg_clockwise"])),
        ) > 1.0e-12
        if displacement_active:
            if self._displacement_map_x is None or self._displacement_map_y is None:
                raise RuntimeError("Virtual displacement map is unavailable")
            full = cv2.remap(
                full,
                self._displacement_map_x,
                self._displacement_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=DISPLACEMENT_BORDER_BGR,
            )
        displacement_completed = time.perf_counter_ns()
        x, y, width, height = self.active_roi
        image = full[y : y + height, x : x + width].copy()
        if image.shape[:2] != (height, width):
            raise RuntimeError("Virtual camera ROI shape differs from effective state")
        if self._once_frames_remaining > 0:
            self._once_frames_remaining -= 1
            if self._once_frames_remaining == 0:
                self._state["imaging"].update(
                    exposure_auto="Off", gain_auto="Off", balance_white_auto="Off"
                )
        completed = time.perf_counter_ns()
        image_space = {
            "schema_version": "1.0",
            "space_id": "virtual_hik_camera_acquisition_pixels",
            "stored_size_px": [width, height],
            "parent_space_id": "virtual_hik_full_sensor_pixels",
            "parent_size_px": self.full_size,
            "roi_in_parent_xywh": [x, y, width, height],
            "local_to_parent_3x3": [
                [1.0, 0.0, float(x)],
                [0.0, 1.0, float(y)],
                [0.0, 0.0, 1.0],
            ],
            "orientation": "scrcpy_camera_locked_initial_as_delivered",
            "mirroring": "as_delivered_by_android_camera_transport",
            "color_order": "BGR",
            "content_from_canonical_parent_3x3": self._displacement_forward_3x3.tolist(),
            "canonical_parent_from_content_3x3": self._displacement_inverse_3x3.tolist(),
        }
        metadata = {
            **dict(source.metadata),
            "adapter_id": self.adapter_id,
            "camera_model_id": self._spec["canonical_device_id"],
            "virtual_camera": True,
            "state_generation": int(self._state["state_generation"]),
            "effective_roi_xywh": [x, y, width, height],
            "zoom": float(self._spec["zoom"]),
            "simulated_displacement": displacement,
            "simulated_displacement_active": displacement_active,
            "simulated_displacement_border_bgr": list(DISPLACEMENT_BORDER_BGR),
            "displacement_map_precomputed": True,
            "displacement_map_generation": self._displacement_map_generation,
            "displacement_map_build_ms": self._displacement_map_build_ms,
            "displacement_remap_ms": (
                displacement_completed - displacement_started
            ) / 1.0e6,
            "roi_implementation": "software_crop_after_full_frame_decode",
            "virtual_processing_ms": (completed - started) / 1.0e6,
            "image_space": image_space,
        }
        self._last_frame_metadata = copy.deepcopy(metadata)
        return FrameSample(
            image=image,
            time_ns=int(source.capture_time_ns),
            receive_time_ns=int(source.receive_time_ns),
            source_id="virtual-hik:{}".format(self._spec["canonical_device_id"]),
            metadata=metadata,
        )

    def get_iris_frame_metadata(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._last_frame_metadata)

    def get_aria_frame_metadata(self) -> Mapping[str, Any]:
        """Compatibility alias for :meth:`get_iris_frame_metadata`."""

        return self.get_iris_frame_metadata()

    @staticmethod
    def _align_roi(roi_xywh: Sequence[int], full_size: Sequence[int]) -> list[int]:
        if roi_xywh is None or len(roi_xywh) != 4:
            raise ValueError("Virtual camera ROI must be XYWH")
        x, y, width, height = map(int, roi_xywh)
        full_width, full_height = map(int, full_size)
        if width <= 0 or height <= 0:
            raise ValueError("Virtual camera ROI dimensions must be positive")
        x = min(max(0, x), full_width - 1)
        y = min(max(0, y), full_height - 1)
        width = min(width, full_width - x)
        height = min(height, full_height - y)
        if width <= 0 or height <= 0:
            raise ValueError("Virtual camera ROI is outside the sensor raster")
        return [x, y, width, height]

    def align_roi(self, roi_xywh: Sequence[int]) -> list[int]:
        return self._align_roi(roi_xywh, self.full_size)

    def set_roi(self, roi_xywh: Sequence[int]) -> list[int]:
        effective = self.align_roi(roi_xywh)
        self._mutate(lambda state: state.update(effective_roi_xywh=effective))
        return list(effective)

    def reset_full_sensor_roi(self) -> list[int]:
        width, height = self.full_size
        return self.set_roi([0, 0, width, height])

    def controls(self) -> Mapping[str, Any]:
        width, height = self.full_size
        return {
            "width": {"minimum": 1, "maximum": width, "increment": 1},
            "height": {"minimum": 1, "maximum": height, "increment": 1},
            "offset_x": {"minimum": 0, "maximum": width - 1, "increment": 1},
            "offset_y": {"minimum": 0, "maximum": height - 1, "increment": 1},
            "exposure_us": {"minimum": 100.0, "maximum": 33333.0, "available": False},
            "gain": {"minimum": 0.0, "maximum": 24.0, "available": False},
            "genicam": {
                "PixelFormat": {
                    "available": True,
                    "access": "read_only",
                    "interface": "enumeration",
                    "value": "BGR8",
                }
            },
            "virtual_control_authority": "stateful_contract_only_android_isp_uncontrolled",
        }

    def set_control(self, name: str, value: Any) -> bool:
        if name == "exposure_us":
            self.set_manual_imaging(float(value), self.imaging_state()["gain"])
            return True
        if name == "gain":
            self.set_manual_imaging(self.imaging_state()["exposure_us"], float(value))
            return True
        return False

    def imaging_state(self) -> Mapping[str, float]:
        imaging = self._state["imaging"]
        return {"exposure_us": float(imaging["exposure_us"]), "gain": float(imaging["gain"])}

    def set_auto_imaging(self) -> Mapping[str, float]:
        def update(state):
            state["imaging"].update(exposure_auto="Continuous", gain_auto="Continuous")

        self._mutate(update)
        return dict(self.imaging_state())

    def set_manual_imaging(self, exposure_us: float, gain: float) -> Mapping[str, float]:
        def update(state):
            state["imaging"].update(
                exposure_us=float(exposure_us), gain=float(gain),
                exposure_auto="Off", gain_auto="Off",
            )

        self._mutate(update)
        return {**dict(self.imaging_state()), "fps": float(self.metadata.get("fps", 0.0))}

    def auto_imaging_modes(self) -> Mapping[str, str]:
        imaging = self._state["imaging"]
        return {
            "ExposureAuto": str(imaging["exposure_auto"]),
            "GainAuto": str(imaging["gain_auto"]),
            "BalanceWhiteAuto": str(imaging["balance_white_auto"]),
        }

    def set_once_auto_imaging(self) -> Mapping[str, Any]:
        def update(state):
            state["imaging"].update(
                exposure_auto="Once", gain_auto="Once", balance_white_auto="Once"
            )

        self._mutate(update)
        self._once_frames_remaining = 4
        return {**dict(self.imaging_state()), "modes": dict(self.auto_imaging_modes())}

    def configure_auto_function_roi(self, camera_roi_xywh: Sequence[int]) -> Mapping[str, Any]:
        effective = self.align_roi(camera_roi_xywh)
        self._mutate(lambda state: state.update(auto_function_roi_xywh=effective))
        selector = {
            "xywh": effective,
            "usage_intensity": True,
            "usage_white_balance": True,
        }
        return {
            "requested_camera_roi_xywh": list(map(int, camera_roi_xywh)),
            "selectors": {"AOI1": dict(selector), "AOI2": dict(selector)},
            "source": "virtual_stateful_contract",
        }

    def configure_once_auto_limits(self, maximum_exposure_us: float, maximum_gain: float) -> Mapping[str, Any]:
        def update(state):
            state["auto_limits"] = {
                "maximum_exposure_us": float(maximum_exposure_us),
                "maximum_gain": float(maximum_gain),
            }

        self._mutate(update)
        return {
            "exposure_upper_us": float(maximum_exposure_us),
            "gain_upper": float(maximum_gain),
            "source": "virtual_stateful_contract",
        }

    def white_balance_state(self) -> Mapping[str, int]:
        imaging = self._state["imaging"]
        return {
            "ratio_red": int(imaging["ratio_red"]),
            "ratio_green": int(imaging["ratio_green"]),
            "ratio_blue": int(imaging["ratio_blue"]),
        }

    def set_white_balance(self, red: int, green: int, blue: int) -> Mapping[str, int]:
        def update(state):
            state["imaging"].update(
                ratio_red=int(red), ratio_green=int(green), ratio_blue=int(blue),
                balance_white_auto="Off",
            )

        self._mutate(update)
        return dict(self.white_balance_state())

    def black_level_control(self) -> Mapping[str, Any]:
        return {"available": False, "error": "Android Camera2 black level is not controlled"}

    def set_black_level(self, value: int) -> int:
        raise RuntimeError("Virtual Android camera does not control physical black level")

    def set_bayer_conversion(self, gamma: float, ccm_rgb_3x3: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        matrix = np.asarray(ccm_rgb_3x3, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Virtual CCM must be a finite 3x3 matrix")

        def update(state):
            state["imaging"].update(gamma=float(gamma), ccm_rgb_3x3=matrix.tolist())

        self._mutate(update)
        return {
            "gamma": float(gamma),
            "ccm_rgb_3x3": matrix.tolist(),
            "implementation": "persistent_metadata_only_no_stream_transform",
        }

    def close(self) -> None:
        source, self._source = self._source, None
        lease, self._lease = self._lease, None
        serial = None if self._spec is None else self._spec.get("serial")
        adb = getattr(source, "adb", None) if source is not None else None
        try:
            if source is not None:
                source.close()
        finally:
            if self.sleep_source_on_close and serial and adb:
                try:
                    subprocess.run(
                        [str(adb), "-s", str(serial), "shell", "input", "keyevent", "223"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if lease is not None:
                lease.release()
            self.configuration = None
            self.metadata = {}
            self._spec = None
            self._full_size = None
            self._state = {}
            self._state_path = None
            self._last_frame_metadata = {}
            self._once_frames_remaining = 0
            self._displacement_map_x = None
            self._displacement_map_y = None
            self._displacement_forward_3x3 = np.eye(3, dtype=np.float64)
            self._displacement_inverse_3x3 = np.eye(3, dtype=np.float64)
            self._displacement_map_build_ms = 0.0
            self._displacement_map_generation = 0


HikMvsCameraAdapter = VirtualHikCameraAdapter


def create_camera_adapter() -> VirtualHikCameraAdapter:
    """Factory consumed by the real rig application's adapter plugin option."""

    return VirtualHikCameraAdapter()
