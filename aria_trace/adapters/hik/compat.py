"""HikCamera-compatible facade backed by a saved rectified calibration.

Usage is intentionally shaped like the common ``hik_camera.hik_camera`` module::

    import aria_trace.adapters.hik.compat as hikcam

    with hikcam.HikCamera(config={"game_id": "genshin-impact"}) as cam:
        rgb = cam.get_frame()

The production profile registry resolves the connected camera and requested
game/mode once during construction. Arbitrary paths and ``latest artifact``
selection are intentionally unsupported. A direct file may be supplied only as
``diagnostic_calibration_override`` for an explicit diagnostic run. The context
manager (or ``open``) owns the camera lifecycle; registry resolution never runs
per frame.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from .driver import (
    HikMvsCameraAdapter,
    RectifiedHikCamera,
    rotate_quarter_turns_clockwise,
)
from aria_trace.services.calibration.rig.hik.spaces import RigCalibratedSpaceConverter


CalibrationPath = Union[str, Path]


def _diagnostic_calibration_override(config: Mapping[str, Any]) -> Optional[Path]:
    if config.get("calibration") is not None:
        raise ValueError(
            "config['calibration'] is obsolete; production selection is automatic. "
            "Use config['diagnostic_calibration_override'] only for diagnostics."
        )
    configured = config.get("diagnostic_calibration_override")
    if configured is None:
        return None
    path = Path(configured).resolve()
    if not path.is_file():
        raise FileNotFoundError("HIK calibration does not exist: {}".format(path))
    return path


def _camera_id_for_registry(
    ip: Optional[str], config: Mapping[str, Any]
) -> str:
    configured = config.get("camera_id")
    if configured:
        return str(configured)
    if ip:
        candidate = Path(str(ip))
        if not candidate.is_file() and candidate.suffix.lower() != ".json":
            return str(ip)
    adapter = HikMvsCameraAdapter(sdk_python_path=config.get("mvs_python_path"))
    devices = list(adapter.devices(probe=True))
    if not devices:
        raise RuntimeError("No connected HIK camera was found for profile resolution")
    if len(devices) != 1:
        raise RuntimeError(
            "Multiple HIK cameras are connected; set config['camera_id']: {}".format(
                ", ".join(str(device.device_id) for device in devices)
            )
        )
    return str(devices[0].device_id)


def _registry_configuration(
    ip: Optional[str], config: Mapping[str, Any]
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    from aria_trace.adapters.filesystem.profile_registry import (
        AdapterRequest,
        ProfileContext,
        ProfileRegistry,
    )
    from aria_trace.adapters.filesystem.system_configuration import (
        load_system_configuration,
    )

    profile_root = Path(config["profile_root"]) if config.get("profile_root") else None
    settings = load_system_configuration(profile_root)
    configured = dict(config)
    for name, value in (
        ("camera_id", settings["devices"].get("camera_id")),
        ("phone_id", settings["devices"].get("phone_id")),
        ("mvs_python_path", settings["tools"].get("mvs_python_path")),
    ):
        if not configured.get(name) and value:
            configured[name] = value
    camera_id = _camera_id_for_registry(ip, configured)
    game_id = (
        configured.get("game_id")
        or os.environ.get("IRIS_GAME_ID")
        or settings["game"].get("game_id")
    )
    normalization = configured.get("normalization")
    if normalization is None:
        normalization = "auto" if bool(configured.get("rectify", True)) else "none"
    if normalization not in ("auto", "none"):
        raise ValueError(
            "HikCamera supports normalization='auto' or 'none'; explicit "
            "dense_remap/homography selection is not implemented"
        )
    context = ProfileContext(
        game_id=str(game_id) if game_id else None,
        platform=str(configured.get("platform", "android")),
        package=configured.get("package"),
        game_version=configured.get("game_version"),
        camera_adapter="hik_mvs",
        camera_id=camera_id,
        phone_id=configured.get("phone_id") or configured.get("phone_serial"),
        phone_model=configured.get("phone_model"),
        panel_display=configured.get("panel_display") or {},
        game_display=configured.get("game_display") or {},
    )
    request = AdapterRequest(
        purpose=str(configured.get("purpose", "application")),
        mode=str(configured.get("mode", "full")),
        normalization=str(normalization),
        color_order=str(configured.get("color_order", "RGB")),
        color_policy=str(configured.get("color_policy", "auto")),
        roi_policy=str(configured.get("roi_policy", "auto")),
        mask_policy=str(configured.get("mask_policy", "none")),
        minimap_margin_px=int(configured.get("minimap_margin_px", 6)),
        frame_rate_policy=str(configured.get("frame_rate_policy", "calibrated")),
        frame_rate=(
            float(configured["frame_rate"])
            if configured.get("frame_rate") is not None
            else None
        ),
    )
    registry = ProfileRegistry(profile_root)
    resolved = registry.resolve_adapter(context, request)
    effective = dict(configured)
    effective.update(
        calibration=resolved["paths"]["rig_calibration"],
        mode=resolved["adapter_plan"]["mode"],
        rectify=bool(resolved["adapter_plan"]["rectify"]),
        color_order=resolved["adapter_plan"]["color_order"],
        color_policy=resolved["adapter_plan"]["color_policy"],
        minimap_margin_px=resolved["adapter_plan"]["minimap_margin_px"],
        mask_policy=resolved["adapter_plan"]["mask_policy"],
        game_model=resolved["adapter_plan"]["game_model"],
        game_upright_quarter_turns_clockwise=resolved["adapter_plan"].get(
            "game_upright_quarter_turns_clockwise", 0
        ),
    )
    if resolved["paths"]["rig_game_profile"]:
        effective["minimap_calibration"] = resolved["paths"]["rig_game_profile"]
    if resolved["paths"].get("game_color_profile"):
        effective["game_color_calibration"] = resolved["paths"][
            "game_color_profile"
        ]
    return Path(effective["calibration"]).resolve(), effective, resolved


class HikCamera:
    """Drop-in high-level HIK camera shape returning rectified display frames.

    The supported compatibility surface is deliberately high-level. It does not
    impersonate the vendor's ctypes structures or status-code based ``MV_CC_*``
    ABI. Methods raise Python exceptions on failure.
    """

    TIMEOUT_MS = 40000

    def __init__(
        self,
        ip: Optional[str] = None,
        host_ip: Optional[str] = None,
        setting_items: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        if ip and (Path(str(ip)).is_file() or Path(str(ip)).suffix.lower() == ".json"):
            raise ValueError(
                "Passing a calibration path as HikCamera(ip) is obsolete; use the "
                "profile registry or diagnostic_calibration_override"
            )
        explicit_calibration = _diagnostic_calibration_override(self.config)
        self.resolved_config: Dict[str, Any]
        if explicit_calibration is None:
            self.calibration_path, self.config, self.resolved_config = (
                _registry_configuration(ip, self.config)
            )
        else:
            self.calibration_path = explicit_calibration
            self.resolved_config = {
                "schema_version": "explicit-path",
                "selection": "diagnostic_calibration_override",
                "paths": {
                    "rig_calibration": str(explicit_calibration),
                    "rig_game_profile": (
                        str(Path(self.config["minimap_calibration"]).resolve())
                        if self.config.get("minimap_calibration")
                        else None
                    ),
                    "game_color_profile": (
                        str(Path(self.config["game_color_calibration"]).resolve())
                        if self.config.get("game_color_calibration")
                        else None
                    ),
                },
                "adapter_plan": {
                    "mode": str(self.config.get("mode", "full")),
                    "rectify": bool(self.config.get("rectify", True)),
                    "color_order": str(self.config.get("color_order", "RGB")).upper(),
                    "mask_policy": str(self.config.get("mask_policy", "none")),
                    "game_model": {
                        "cursor_follows": "character",
                        "cursor_behavior_by_acquisition": {
                            "zigzag": "static",
                            "micro_movement": "rotating",
                        },
                        "minimap_orientation": "unspecified",
                        "source": "iris_default",
                    },
                    "registry_reads_per_frame": 0,
                    "phone_operations": "none",
                },
            }
        self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        camera = self.calibration["camera"]
        self._ip = str(camera.get("device_id", ip or ""))
        if ip and str(ip) != self._ip:
            raise ValueError(
                "Requested camera {!r} does not match calibrated camera {!r}".format(
                    ip, self._ip
                )
            )
        self.host_ip = host_ip
        self.setting_items = dict(setting_items or {})
        self.is_open = False
        self.last_time_get_frame = 0.0
        self._reader = None
        self._last_frame = None
        self._last_frame_by_stream: Dict[str, np.ndarray] = {}
        self.last_frame_metadata: Dict[str, Any] = {}
        self._last_frame_metadata_by_stream: Dict[str, Dict[str, Any]] = {}
        self._color_order = str(self.config.get("color_order", "RGB")).upper()
        self._rectify_enabled = bool(self.config.get("rectify", True))
        self._game_upright_turns = int(
            self.config.get("game_upright_quarter_turns_clockwise", 0)
        ) % 4
        if self._color_order not in ("RGB", "BGR"):
            raise ValueError("config['color_order'] must be RGB or BGR")
        imaging = self.calibration["imaging"]
        self._exposure_us = float(imaging["exposure_us"])
        self._gain = float(imaging["gain"])
        self._black_level = (
            int(imaging["black_level"]) if imaging.get("black_level") is not None else None
        )
        self._white_balance = dict(imaging["white_balance"])
        self._balance_selector = "Red"
        self._fps = float(camera["full_sensor_mode"]["fps"])
        if self._rectify_enabled:
            output_width, output_height = map(
                int, self.calibration["normalization"]["output_size_px"]
            )
        else:
            _, _, output_width, output_height = map(
                int, camera["hardware_roi_xywh"]
            )
        if self._game_upright_turns % 2:
            output_width, output_height = output_height, output_width
        self.shape = (output_height, output_width, 3)
        self.bit = 24
        self.pixel_format = "RGB8Packed" if self._color_order == "RGB" else "BGR8Packed"

    @property
    def orientation(self) -> Dict[str, Any]:
        """Return the ChArUco orientation evidence applied to every frame."""

        return dict(self.calibration.get("normalization", {}).get("orientation", {}))

    def space_converter(
        self, adb_surface_quarter_turns_clockwise_from_natural: int = 0
    ) -> RigCalibratedSpaceConverter:
        """Return coordinate conversion matching this adapter's rectified frames."""

        if not self._rectify_enabled:
            raise RuntimeError(
                "Adapter/ADB conversion requires config['rectify'] to be true"
            )
        return RigCalibratedSpaceConverter(
            self.calibration,
            adb_surface_quarter_turns_clockwise_from_natural,
            self._game_upright_turns,
        )

    def camera_adapter_to_adb_points(
        self,
        points_xy: Sequence[Sequence[float]],
        adb_surface_quarter_turns_clockwise_from_natural: int = 0,
    ) -> np.ndarray:
        return self.space_converter(
            adb_surface_quarter_turns_clockwise_from_natural
        ).camera_adapter_to_adb_points(points_xy)

    def adb_to_camera_adapter_points(
        self,
        points_xy: Sequence[Sequence[float]],
        adb_surface_quarter_turns_clockwise_from_natural: int = 0,
    ) -> np.ndarray:
        return self.space_converter(
            adb_surface_quarter_turns_clockwise_from_natural
        ).adb_to_camera_adapter_points(points_xy)

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def is_raw(self) -> bool:
        return False

    def is_calibrated(self) -> bool:
        """Return whether this facade has a complete saved rig calibration."""

        return RectifiedHikCamera(
            self.calibration_path,
            rectify=self._rectify_enabled,
            output_quarter_turns_clockwise=self._game_upright_turns,
        ).is_calibrated()

    @classmethod
    def get_all_ips(cls, sdk_python_path: Optional[str] = None) -> list[str]:
        """Return HIK device identifiers without opening any camera."""

        adapter = HikMvsCameraAdapter(sdk_python_path=sdk_python_path)
        return [device.device_id for device in adapter.devices(probe=True)]

    @classmethod
    def get_cam(cls) -> "HikCamera":
        return cls()

    @classmethod
    def get_cams(cls, ips: Optional[Sequence[str]] = None) -> "MultiHikCamera":
        selected = list(ips or cls.get_all_ips())
        return MultiHikCamera({camera_id: cls(camera_id) for camera_id in selected})

    get_all_cams = get_cams

    def _new_reader(self):
        factory = self.config.get("reader_factory")
        if factory is not None:
            return factory(self.calibration_path)
        minimap_calibration = self.config.get("minimap_calibration")
        game_color = {}
        game_color_path = self.config.get("game_color_calibration")
        use_game_color = str(self.config.get("color_policy", "auto")) not in (
            "rig_locked",
            "unadjusted",
        )
        if game_color_path and use_game_color:
            document = json.loads(
                Path(game_color_path).read_text(encoding="utf-8")
            )
            payload = document.get("payload")
            if isinstance(payload, Mapping):
                document = {**document, **dict(payload)}
            game_color = dict(document.get("hik_bayer_conversion") or {})
        if minimap_calibration is not None:
            from .game_camera import ProfiledHikGameCamera

            options = {
                "mode": str(self.config.get("mode", "minimap")),
                "rectify_minimap": bool(self.config.get("rectify", True)),
                "minimap_margin_px": int(self.config.get("minimap_margin_px", 6)),
                "apply_game_color": use_game_color,
                "output_quarter_turns_clockwise": self._game_upright_turns,
                "mask_policy": str(self.config.get("mask_policy", "none")),
            }
            if game_color:
                options["bayer_conversion"] = game_color
            return ProfiledHikGameCamera(
                self.calibration_path, minimap_calibration, **options
            )
        options = {
            "rectify": self._rectify_enabled,
            "output_quarter_turns_clockwise": self._game_upright_turns,
        }
        if game_color and use_game_color:
            options["bayer_conversion"] = game_color
        return RectifiedHikCamera(self.calibration_path, **options)

    def open(self) -> "HikCamera":
        if self.is_open:
            return self
        reader = self._new_reader()
        try:
            opened = reader.open()
            self._reader = opened if opened is not None else reader
            self.is_open = True
            self.setting()
        except Exception:
            try:
                reader.release()
            except Exception:
                pass
            self._reader = None
            self.is_open = False
            raise
        return self

    def close(self) -> None:
        reader, self._reader = self._reader, None
        self.is_open = False
        if reader is not None:
            reader.release()
        self.last_frame_metadata = {}
        self._last_frame_metadata_by_stream = {}
        self._last_frame_by_stream = {}

    release = close

    def __enter__(self) -> "HikCamera":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _require_reader(self):
        if not self.is_open or self._reader is None:
            raise RuntimeError("HikCamera is not open; use `with cam:` or cam.open()")
        return self._reader

    def _convert_output(self, bgr: np.ndarray) -> np.ndarray:
        if self._color_order == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr

    def _public_frame_metadata(
        self, metadata: Mapping[str, Any], frame: np.ndarray
    ) -> Dict[str, Any]:
        result = copy.deepcopy(dict(metadata))
        image_space = dict(result.get("image_space") or {})
        image_space.update(
            stored_size_px=[int(frame.shape[1]), int(frame.shape[0])],
            color_order=self._color_order,
        )
        result["image_space"] = image_space
        return result

    def get_frame(self) -> np.ndarray:
        reader = self._require_reader()
        if hasattr(reader, "read_sample"):
            sample = reader.read_sample()
            bgr = sample.image
            source_metadata = {
                **dict(sample.metadata),
                "host_capture_time_ns": int(sample.time_ns),
                "host_receive_time_ns": int(
                    sample.receive_time_ns or sample.time_ns
                ),
                "host_timestamp_clock_id": str(sample.clock_id),
            }
        else:
            ok, bgr = reader.read()
            if not ok or bgr is None:
                raise RuntimeError("HIK camera returned no rectified frame")
            source_metadata = {}
        frame = self._convert_output(bgr)
        self.last_frame_metadata = self._public_frame_metadata(
            source_metadata, frame
        )
        self._last_frame_metadata_by_stream = {
            str(self.last_frame_metadata.get("stream_id", "default")): dict(
                self.last_frame_metadata
            )
        }
        stream_id = str(self.last_frame_metadata.get("stream_id", "default"))
        self._last_frame_by_stream = {stream_id: frame}
        if str(self.config.get("mode", "full")) == "full":
            self._last_frame_by_stream["full"] = frame
        self._last_frame = frame
        self.last_time_get_frame = time.time()
        self.shape = tuple(frame.shape)
        self.bit = int(frame.dtype.itemsize * 8 * (frame.shape[2] if frame.ndim == 3 else 1))
        return frame

    def get_frames(self) -> Dict[str, np.ndarray]:
        """Return synchronized products when configured for dual streaming."""

        reader = self._require_reader()
        if not hasattr(reader, "read_streams"):
            return {"full": self.get_frame()}
        frame_set = reader.read_streams()
        declared = getattr(frame_set, "stream_metadata", {}) or {}
        frames = {
            name: self._convert_output(frame)
            for name, frame in frame_set.streams.items()
        }
        self._last_frame_by_stream = dict(frames)
        self._last_frame_metadata_by_stream = {
            str(name): self._public_frame_metadata(
                {
                    **dict(declared.get(name) or frame_set.metadata),
                    "host_capture_time_ns": int(frame_set.time_ns),
                    "host_receive_time_ns": int(frame_set.receive_time_ns),
                    "host_timestamp_clock_id": "host_perf_counter_ns",
                },
                frames[name],
            )
            for name in frame_set.streams
        }
        self.last_frame_metadata = {
            **dict(frame_set.metadata),
            "streams": copy.deepcopy(self._last_frame_metadata_by_stream),
        }
        preferred = frames.get("minimap")
        if preferred is None:
            preferred = frames.get("full")
        if preferred is not None:
            self._last_frame = preferred
            self.last_time_get_frame = time.time()
            self.shape = tuple(preferred.shape)
            self.bit = int(
                preferred.dtype.itemsize
                * 8
                * (preferred.shape[2] if preferred.ndim == 3 else 1)
            )
        return frames

    def correct_game_orientation(
        self,
        adb_image: np.ndarray,
        hik_full_image: Optional[np.ndarray] = None,
        *,
        preferred_confidence: float = 0.50,
        preferred_margin: float = 0.08,
    ) -> Dict[str, Any]:
        """Correct game-up from one current ADB/HIK image pair.

        This proprietary IRIS operation tries all four discrete orientations
        with the existing cross-source evidence matcher. It never operates the
        phone: callers must provide the canonical full ADB screenshot. On a
        confident match an open adapter is reopened once, rebuilding its dense
        maps with the selected quarter-turn and adding no per-frame overhead.
        """

        if not self._rectify_enabled:
            raise RuntimeError(
                "Runtime game-orientation correction requires rectification; "
                "an unrectified hardware-ROI frame is not calibration-display space"
            )
        if adb_image is None or not isinstance(adb_image, np.ndarray):
            raise ValueError("adb_image must be a decoded full ADB screenshot")
        if hik_full_image is None:
            hik_full_image = self._last_frame_by_stream.get("full")
        if hik_full_image is None:
            raise RuntimeError(
                "Runtime game-orientation correction needs a current full HIK "
                "frame; open the adapter in full or dual mode"
            )

        current_turns = int(self._game_upright_turns) % 4
        public_bgr = (
            cv2.cvtColor(hik_full_image, cv2.COLOR_RGB2BGR)
            if self._color_order == "RGB"
            else hik_full_image
        )
        calibration_display_bgr = rotate_quarter_turns_clockwise(
            public_bgr, -current_turns
        )
        from aria_trace.services.calibration.rig.cross_source import (
            match_game_camera_orientation,
        )

        summary, _evidence = match_game_camera_orientation(
            adb_image,
            calibration_display_bgr,
            self.calibration_path,
            preferred_confidence=float(preferred_confidence),
            preferred_margin=float(preferred_margin),
        )
        selected_turns = int(
            summary[
                "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
            ]
        ) % 4
        result = {
            **summary,
            "previous_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": (
                current_turns
            ),
            "applied": False,
            "adapter_reopened": False,
        }
        if summary.get("status") != "selected":
            result["application_status"] = "not_applied_ambiguous_image_evidence"
            return result

        application = self._apply_game_orientation_turns(selected_turns)
        result["applied"] = True
        result.update(application)
        return result

    def _apply_game_orientation_turns(self, selected_turns: int) -> Dict[str, Any]:
        selected_turns = int(selected_turns) % 4
        previous_turns = int(self._game_upright_turns) % 4
        was_open = bool(self.is_open)
        if was_open:
            self.close()
        self._game_upright_turns = selected_turns
        self.config["game_upright_quarter_turns_clockwise"] = selected_turns
        adapter_plan = self.resolved_config.setdefault("adapter_plan", {})
        adapter_plan["game_upright_quarter_turns_clockwise"] = selected_turns
        if (previous_turns - selected_turns) % 2 and len(self.shape) >= 2:
            self.shape = (self.shape[1], self.shape[0]) + tuple(self.shape[2:])
        if was_open:
            self.open()
        return {
            "adapter_reopened": was_open,
            "application_status": (
                "applied_dense_maps_rebuilt"
                if was_open
                else "applied_for_next_open"
            ),
        }

    def correct_game_orientation_from_android_surface(
        self,
        android_surface_quarter_turns_clockwise_from_natural: int,
        *,
        foreground_package: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compose game-up from Android surface telemetry and the saved rig.

        This deterministic operation assumes the foreground game renders its
        top edge at the top of Android's current logical surface. It is separate
        from :meth:`correct_game_orientation`, which verifies real pixels by
        trying all four ADB/HIK image orientations.
        """

        surface_turns = int(
            android_surface_quarter_turns_clockwise_from_natural
        ) % 4
        phone = dict(self.calibration.get("phone") or {})
        viewer = dict(phone.get("viewer") or {})
        rig_display_turns = int(
            phone.get(
                "orientation_quarter_turns",
                viewer.get("canonical_orientation_quarter_turns", 0),
            )
        ) % 4
        selected_turns = (surface_turns - rig_display_turns) % 4
        expected_package = str(
            (((self.resolved_config.get("context") or {}).get("game") or {}).get(
                "package"
            ) or "")
        ).strip()
        observed_package = str(foreground_package or "").strip()
        warnings = []
        if expected_package and observed_package and expected_package != observed_package:
            warnings.append(
                "Foreground package {!r} differs from profile package {!r}; "
                "orientation was composed from the observed Android surface as "
                "requested, but image-evidence verification is recommended."
                .format(observed_package, expected_package)
            )
        application = self._apply_game_orientation_turns(selected_turns)
        return {
            "schema_version": 1,
            "status": "applied_surface_composition",
            "selection_basis": "foreground_game_android_surface_and_saved_rig_relation",
            "assumption": "foreground_game_top_matches_android_logical_surface_top",
            "foreground_package": observed_package or None,
            "profile_package": expected_package or None,
            "android_surface_quarter_turns_clockwise_from_phone_natural": surface_turns,
            "rig_calibration_display_quarter_turns_clockwise_from_phone_natural": (
                rig_display_turns
            ),
            "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": (
                selected_turns
            ),
            "selected_camera_adapter_image_degrees_clockwise_from_calibration_display": (
                selected_turns * 90
            ),
            "warnings": warnings,
            "image_evidence_used": False,
            "image_evidence_override_available": True,
            "applied": True,
            **application,
        }

    def get_iris_frame_metadata(
        self, stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return IRIS space/provenance metadata for the last frame.

        HIK-compatible methods such as :meth:`get_frame`, :meth:`get_frames`,
        and :meth:`read` deliberately keep their established image-only return
        values. This proprietary additive method exposes metadata without
        changing that native-facing surface.
        """

        if stream_id is not None:
            return copy.deepcopy(
                self._last_frame_metadata_by_stream.get(str(stream_id), {})
            )
        if len(self._last_frame_metadata_by_stream) == 1:
            return copy.deepcopy(
                next(iter(self._last_frame_metadata_by_stream.values()))
            )
        if self._last_frame_metadata_by_stream:
            return copy.deepcopy({
                "streams": {
                    name: dict(value)
                    for name, value in self._last_frame_metadata_by_stream.items()
                }
            })
        return copy.deepcopy(self.last_frame_metadata)

    def get_game_model(self) -> Dict[str, Any]:
        """Return the resolved game-behavior model without reading a frame."""

        return copy.deepcopy(
            dict(
                self.config.get("game_model")
                or {
                    "cursor_follows": "character",
                    "cursor_behavior_by_acquisition": {
                        "zigzag": "static",
                        "micro_movement": "rotating",
                    },
                    "minimap_orientation": "unspecified",
                    "source": "iris_default",
                }
            )
        )

    def get_cursor_geometry(
        self, stream_id: str = "minimap"
    ) -> Dict[str, Any]:
        """Return calibrated cursor center and size with explicit space metadata."""

        reader = self._require_reader()
        method = getattr(reader, "get_cursor_geometry", None)
        if method is None:
            return {}
        return copy.deepcopy(dict(method(stream_id)))

    def get_minimap_geometry(
        self, stream_id: str = "minimap"
    ) -> Dict[str, Any]:
        """Return the mini-map boundary with explicit stream-space metadata."""

        reader = self._require_reader()
        method = getattr(reader, "get_minimap_geometry", None)
        if method is None:
            return {}
        return copy.deepcopy(dict(method(stream_id)))

    def get_aria_frame_metadata(
        self, stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compatibility alias for :meth:`get_iris_frame_metadata`."""

        return self.get_iris_frame_metadata(stream_id)

    def get_frame_with_config(self) -> None:
        self._last_frame = self.get_frame()

    def robust_get_frame(self) -> np.ndarray:
        try:
            return self.get_frame()
        except Exception:
            self.reset()
            return self.get_frame()

    def read(self):
        """OpenCV-compatible alias returning ``(ok, frame)``."""

        try:
            return True, self.get_frame()
        except Exception:
            return False, None

    def get_shape(self):
        return self.shape

    def reset(self) -> None:
        self.close()
        self.open()

    def waite(self, timeout: int = 20) -> None:
        """Compatibility spelling: verify the configured camera can be opened."""

        deadline = time.monotonic() + float(timeout)
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.open()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        raise TimeoutError("Cannot open calibrated HIK camera: {}".format(last_error))

    def set_rgb(self) -> None:
        self._color_order = "RGB"
        self.pixel_format = "RGB8Packed"

    def set_bgr(self) -> None:
        self._color_order = "BGR"
        self.pixel_format = "BGR8Packed"

    def setting(self) -> None:
        """Apply the configured high-level settings after opening."""

        for key, value in self.setting_items.items():
            self.setitem(key, value)

    def set_raw(self, *args, **kwargs) -> None:
        raise NotImplementedError("Rectified calibrated output cannot be exposed as Bayer/raw")

    def get_exposure(self) -> float:
        return float(self._exposure_us)

    def set_exposure(self, exposure_us: float) -> None:
        if abs(float(exposure_us) - self._exposure_us) > max(2.0, self._exposure_us * 0.001):
            raise RuntimeError("Exposure is locked by the saved rig calibration")

    def get_exposure_by_second(self) -> float:
        return self.get_exposure() * 1.0e-6

    def set_exposure_by_second(self, seconds: float) -> None:
        self.set_exposure(float(seconds) * 1.0e6)

    def get_gain(self) -> float:
        return float(self._gain)

    def set_gain(self, gain: float) -> None:
        if abs(float(gain) - self._gain) > 1.0e-6:
            raise RuntimeError("Gain is locked by the saved rig calibration")

    def set_white_balance(self, red: int, green: int, blue: int) -> None:
        requested = {
            "ratio_red": int(red),
            "ratio_green": int(green),
            "ratio_blue": int(blue),
        }
        if requested != {
            key: int(self._white_balance[key]) for key in requested
        }:
            raise RuntimeError("White balance is locked by the saved rig calibration")

    def getitem(self, key: str) -> Any:
        normalized = str(key)
        height, width = self.shape[:2]
        values = {
            "ExposureTime": self.get_exposure(),
            "Gain": self.get_gain(),
            "Width": int(width),
            "Height": int(height),
            "OffsetX": 0,
            "OffsetY": 0,
            "PixelFormat": self.pixel_format,
            "AcquisitionFrameRate": self._fps,
            "AcquisitionFrameRateEnable": True,
            "ExposureAuto": "Off",
            "GainAuto": "Off",
            "BalanceWhiteAuto": "Off",
            "BalanceRatioSelector": self._balance_selector,
            "BalanceRatio": int(
                self._white_balance[
                    "ratio_{}".format(self._balance_selector.lower())
                ]
            ),
            "TriggerMode": "Off",
        }
        if self._black_level is not None:
            values["BlackLevelEnable"] = True
            values["BlackLevel"] = self._black_level
        if normalized not in values:
            raise KeyError("Unsupported calibrated-camera setting {}".format(normalized))
        return values[normalized]

    def setitem(self, key: str, value: Any) -> None:
        normalized = str(key)
        if normalized == "ExposureTime":
            self.set_exposure(float(value))
            return
        if normalized == "Gain":
            self.set_gain(float(value))
            return
        if normalized == "BlackLevel":
            if self._black_level is None:
                raise KeyError("Saved calibration has no black-level setting")
            if int(value) != self._black_level:
                raise RuntimeError("Black level is locked by the saved rig calibration")
            return
        if normalized == "AcquisitionFrameRate":
            reader = self._require_reader()
            if not reader.adapter.set_control("frame_rate", float(value)):
                raise RuntimeError("HIK camera does not expose frame-rate control")
            self._fps = float(value)
            return
        if normalized == "BalanceRatioSelector":
            selector = str(value).title()
            if selector not in ("Red", "Green", "Blue"):
                raise ValueError("BalanceRatioSelector must be Red, Green, or Blue")
            self._balance_selector = selector
            return
        if normalized == "BalanceRatio":
            ratios = {
                color: int(self._white_balance["ratio_{}".format(color)])
                for color in ("red", "green", "blue")
            }
            ratios[self._balance_selector.lower()] = int(value)
            self.set_white_balance(ratios["red"], ratios["green"], ratios["blue"])
            return
        if normalized == "PixelFormat":
            if str(value) == "RGB8Packed":
                self.set_rgb()
                return
            if str(value) in ("BGR8Packed", "BGR8"):
                self.set_bgr()
                return
            raise ValueError("Rectified HikCamera supports RGB8Packed or BGR8Packed")
        if normalized in ("Width", "Height", "OffsetX", "OffsetY"):
            raise RuntimeError(
                "Calibrated ROI/shape is immutable; changing it invalidates rectification"
            )
        fixed_modes = {
            "AcquisitionFrameRateEnable": True,
            "ExposureAuto": "Off",
            "GainAuto": "Off",
            "BalanceWhiteAuto": "Off",
            "TriggerMode": "Off",
        }
        if self._black_level is not None:
            fixed_modes["BlackLevelEnable"] = True
        if normalized in fixed_modes:
            if value != fixed_modes[normalized]:
                raise RuntimeError(
                    "Calibrated HikCamera requires {}={!r}".format(
                        normalized, fixed_modes[normalized]
                    )
                )
            return
        raise KeyError("Unsupported calibrated-camera setting {}".format(normalized))

    __getitem__ = getitem
    __setitem__ = setitem


class MultiHikCamera(dict):
    """Small synchronous compatibility container for multiple HikCamera objects."""

    def __enter__(self):
        opened = []
        try:
            for camera in self.values():
                camera.open()
                opened.append(camera)
        except Exception:
            for camera in reversed(opened):
                camera.close()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        for camera in self.values():
            camera.close()

    def get_frame(self):
        return {key: camera.get_frame() for key, camera in self.items()}

    def robust_get_frame(self):
        return {key: camera.robust_get_frame() for key, camera in self.items()}


Camera = HikCamera
get_all_ips = HikCamera.get_all_ips
get_cam = HikCamera.get_cam
get_cams = HikCamera.get_cams
get_all_cams = HikCamera.get_all_cams

__all__ = [
    "Camera",
    "HikCamera",
    "MultiHikCamera",
    "get_all_cams",
    "get_all_ips",
    "get_cam",
    "get_cams",
]
