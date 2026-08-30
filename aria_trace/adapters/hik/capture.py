"""Acquisition-recorder sources for native and calibrated HIK streams."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Mapping, Optional

import cv2

from aria_trace.domain.packets import FramePacket
from aria_trace.adapters.rig.devices import CameraConfiguration
from aria_trace.services.calibration.rig.hik.driver import HikMvsCameraAdapter
from aria_trace.services.calibration.rig.hik.driver import RectifiedHikCamera
from aria_trace.adapters.sources import FrameSource


def rotate_quarter_turns_clockwise(image, quarter_turns: int):
    """Rotate an image between two explicitly declared raster spaces."""

    turns = int(quarter_turns) % 4
    if turns == 0:
        return image
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


class CalibratedHikFrameSource(FrameSource):
    """Expose normalized HIK frames with raw device timing kept as metadata."""

    def __init__(
        self,
        calibration_file: Path,
        stream_id: str = "hik_phone",
        rectify: bool = True,
        output_quarter_turns_clockwise: int = 0,
        reader: Optional[RectifiedHikCamera] = None,
    ) -> None:
        self.calibration_file = Path(calibration_file)
        self.stream_id = str(stream_id)
        self.rectify = bool(rectify)
        self.output_quarter_turns_clockwise = (
            int(output_quarter_turns_clockwise) % 4
        )
        self.reader = reader
        self._owns_reader = reader is None
        self._started = False
        self._orientation_lock = threading.Lock()
        self._orientation_evidence = None

    def set_output_orientation(
        self,
        quarter_turns_clockwise: int,
        evidence: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Set output rotation relative to the rig-calibration display space."""

        with self._orientation_lock:
            self.output_quarter_turns_clockwise = (
                int(quarter_turns_clockwise) % 4
            )
            self._orientation_evidence = (
                dict(evidence) if evidence is not None else None
            )

    def alignment_evidence_image(self, packet: FramePacket):
        """Return this packet in rig-normalized calibration-display space.

        A minimum-latency (rectify=False) stream stays unrectified for normal
        reads.  Only this explicit diagnostic conversion performs the saved
        rig warp so its orientation can be compared with an ADB image.
        """

        image = packet.image
        padding = packet.metadata.get(
            "video_encoding_padding_right_bottom_px", [0, 0]
        )
        padding_right, padding_bottom = map(int, padding)
        if padding_right:
            image = image[:, :-padding_right]
        if padding_bottom:
            image = image[:-padding_bottom, :]
        turns = int(
            packet.metadata.get(
                "output_quarter_turns_clockwise_from_calibration_display", 0
            )
        ) % 4
        calibration_display = rotate_quarter_turns_clockwise(image, -turns)
        if self.rectify:
            return calibration_display
        rectify_for_evidence = getattr(self.reader, "rectify_for_evidence", None)
        if not callable(rectify_for_evidence):
            raise RuntimeError(
                "The HIK reader cannot rectify a hardware-ROI frame for "
                "cross-source orientation evidence"
            )
        return rectify_for_evidence(calibration_display)

    def start(self) -> None:
        if self._started:
            return
        if self.reader is None:
            self.reader = RectifiedHikCamera(
                self.calibration_file, rectify=self.rectify
            )
        self.reader.open()
        self._started = True

    def read(self) -> Optional[FramePacket]:
        if not self._started or self.reader is None:
            return None
        sample = self.reader.read_sample()
        with self._orientation_lock:
            output_turns = self.output_quarter_turns_clockwise
            orientation_evidence = (
                dict(self._orientation_evidence)
                if self._orientation_evidence is not None
                else None
            )
        image = rotate_quarter_turns_clockwise(sample.image, output_turns)
        content_height, content_width = image.shape[:2]
        padding_right = int(image.shape[1] % 2)
        padding_bottom = int(image.shape[0] % 2)
        if padding_right or padding_bottom:
            image = cv2.copyMakeBorder(
                image,
                0,
                padding_bottom,
                0,
                padding_right,
                cv2.BORDER_REPLICATE,
            )
        metadata = dict(sample.metadata)
        raw_device_time = metadata.get("device_timestamp")
        if raw_device_time is None:
            high = metadata.get("device_timestamp_high")
            low = metadata.get("device_timestamp_low")
            if high is not None and low is not None:
                raw_device_time = (int(high) << 32) | int(low)
        metadata.update(
            {
                "source": "hik_mvs_calibrated",
                "coordinate_space": "hik_session_aligned_visible_phone_pixels",
                "source_coordinate_space": "hik_rig_rectified_visible_phone_pixels",
                "rig_calibration": str(self.calibration_file.resolve()),
                "timestamp_timebase": "host perf_counter_ns at frame receive",
                "device_timestamp_raw": raw_device_time,
                "calibrated_source_size_px": [
                    int(content_width),
                    int(content_height),
                ],
                "output_quarter_turns_clockwise_from_calibration_display": (
                    output_turns
                ),
                "output_orientation_evidence": orientation_evidence,
                "video_encoding_padding_right_bottom_px": [
                    padding_right,
                    padding_bottom,
                ],
            }
        )
        return FramePacket(
            self.stream_id,
            image,
            int(sample.time_ns),
            int(sample.receive_time_ns or sample.time_ns),
            source_time_ns=(int(raw_device_time) if raw_device_time is not None else None),
            metadata=metadata,
        )

    def stop(self) -> None:
        if self.reader is not None and self._started:
            self.reader.release()
        self._started = False
        if self._owns_reader:
            self.reader = None

    def describe(self) -> dict:
        with self._orientation_lock:
            output_turns = self.output_quarter_turns_clockwise
            orientation_evidence = (
                dict(self._orientation_evidence)
                if self._orientation_evidence is not None
                else None
            )
        return {
            "type": type(self).__name__,
            "stream_id": self.stream_id,
            "calibration": str(self.calibration_file.resolve()),
            "rectified": self.rectify,
            "coordinate_space": "hik_session_aligned_visible_phone_pixels",
            "source_coordinate_space": "hik_rig_rectified_visible_phone_pixels",
            "output_quarter_turns_clockwise_from_calibration_display": (
                output_turns
            ),
            "output_orientation_evidence": orientation_evidence,
            "host_timestamp": "perf_counter_ns_at_frame_receive",
            "device_timestamp": "raw_hik_counter_in_frame_metadata",
            "video_encoding_padding": "replicate at right/bottom only when a dimension is odd",
        }


class NativeHikFrameSource(FrameSource):
    """Record the native HIK sensor view without requiring a rig calibration."""

    def __init__(
        self,
        camera_id: str,
        stream_id: str = "hik_full",
        *,
        width_px: int = 2448,
        height_px: int = 2048,
        fps: float = 30.0,
        sdk_python_path: Optional[str] = None,
        adapter: Optional[HikMvsCameraAdapter] = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.stream_id = str(stream_id)
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.fps = float(fps)
        self.adapter = adapter or HikMvsCameraAdapter(
            sdk_python_path=sdk_python_path
        )
        self._owns_adapter = adapter is None
        self._started = False
        self.metadata = {}
        self.controls = {}
        self.full_sensor_roi = None

    @staticmethod
    def _feature_value(features: dict, name: str):
        value = features.get(name) or {}
        return value.get("value") if isinstance(value, dict) else None

    def start(self) -> None:
        if self._started:
            return
        self.metadata = dict(
            self.adapter.open(
                CameraConfiguration(
                    device_id=self.camera_id,
                    width_px=self.width_px,
                    height_px=self.height_px,
                    fps=self.fps,
                    backend="hik_mvs",
                )
            )
        )
        try:
            self.controls = dict(self.adapter.controls())
            features = self.controls.get("genicam") or {}
            sensor_width = self._feature_value(features, "SensorWidth")
            sensor_height = self._feature_value(features, "SensorHeight")
            if sensor_width is None:
                sensor_width = (self.controls.get("width") or {}).get(
                    "maximum", self.metadata.get("width_px", self.width_px)
                )
            if sensor_height is None:
                sensor_height = (self.controls.get("height") or {}).get(
                    "maximum", self.metadata.get("height_px", self.height_px)
                )
            self.full_sensor_roi = list(
                self.adapter.set_roi(
                    [0, 0, int(sensor_width), int(sensor_height)]
                )
            )
            self.metadata["full_sensor_roi_xywh"] = list(self.full_sensor_roi)
            self._started = True
        except Exception:
            self.adapter.close()
            raise

    def read(self) -> Optional[FramePacket]:
        if not self._started:
            return None
        sample = self.adapter.read()
        image = sample.image
        padding_right = int(image.shape[1] % 2)
        padding_bottom = int(image.shape[0] % 2)
        if padding_right or padding_bottom:
            image = cv2.copyMakeBorder(
                image,
                0,
                padding_bottom,
                0,
                padding_right,
                cv2.BORDER_REPLICATE,
            )
        metadata = dict(sample.metadata)
        raw_device_time = metadata.get("device_timestamp")
        if raw_device_time is None:
            high = metadata.get("device_timestamp_high")
            low = metadata.get("device_timestamp_low")
            if high is not None and low is not None:
                raw_device_time = (int(high) << 32) | int(low)
        metadata.update(
            {
                "source": "hik_mvs_native_full_sensor",
                "coordinate_space": "native_hik_sensor_bgr_pixels",
                "timestamp_timebase": "host perf_counter_ns at frame receive",
                "device_timestamp_raw": raw_device_time,
                "full_sensor_roi_xywh": list(self.full_sensor_roi),
                "native_source_size_px": [
                    int(sample.image.shape[1]),
                    int(sample.image.shape[0]),
                ],
                "video_encoding_padding_right_bottom_px": [
                    padding_right,
                    padding_bottom,
                ],
            }
        )
        return FramePacket(
            self.stream_id,
            image,
            int(sample.time_ns),
            int(sample.receive_time_ns or sample.time_ns),
            source_time_ns=(
                int(raw_device_time) if raw_device_time is not None else None
            ),
            metadata=metadata,
        )

    def stop(self) -> None:
        if self._started:
            self.adapter.close()
        self._started = False

    def describe(self) -> dict:
        return {
            "type": type(self).__name__,
            "stream_id": self.stream_id,
            "camera_id": self.camera_id,
            "coordinate_space": "native_hik_sensor_bgr_pixels",
            "full_sensor_roi_xywh": self.full_sensor_roi,
            "effective_camera": dict(self.metadata),
            "camera_controls": dict(self.controls),
            "host_timestamp": "perf_counter_ns_at_frame_receive",
            "device_timestamp": "raw_hik_counter_in_frame_metadata",
            "video_encoding_padding": (
                "replicate at right/bottom only when a dimension is odd"
            ),
        }
