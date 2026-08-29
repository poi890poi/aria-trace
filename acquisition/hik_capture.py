"""Acquisition-recorder source backed by a saved HIK rig calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from .models import FramePacket
from .rig_calibration.hik.driver import RectifiedHikCamera
from .sources import FrameSource


class CalibratedHikFrameSource(FrameSource):
    """Expose normalized HIK frames with raw device timing kept as metadata."""

    def __init__(
        self,
        calibration_file: Path,
        stream_id: str = "hik_phone",
        rectify: bool = True,
        reader: Optional[RectifiedHikCamera] = None,
    ) -> None:
        self.calibration_file = Path(calibration_file)
        self.stream_id = str(stream_id)
        self.rectify = bool(rectify)
        self.reader = reader
        self._owns_reader = reader is None
        self._started = False

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
                "source": "hik_mvs_calibrated",
                "rig_calibration": str(self.calibration_file.resolve()),
                "timestamp_timebase": "host perf_counter_ns at frame receive",
                "device_timestamp_raw": raw_device_time,
                "calibrated_source_size_px": [
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
        return {
            "type": type(self).__name__,
            "stream_id": self.stream_id,
            "calibration": str(self.calibration_file.resolve()),
            "rectified": self.rectify,
            "host_timestamp": "perf_counter_ns_at_frame_receive",
            "device_timestamp": "raw_hik_counter_in_frame_metadata",
            "video_encoding_padding": "replicate at right/bottom only when a dimension is odd",
        }
