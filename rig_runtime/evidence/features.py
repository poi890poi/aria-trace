"""Online feature evidence extracted from raw frames before video encoding."""

import hashlib
import json
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


class OnlineSiftRecorder:
    """Persist the SIFT observations made from raw capture frames.

    Descriptors are stored as uint8 only when that conversion is numerically
    lossless. OpenCV returns SIFT descriptors as float32 integer values, so this
    normally cuts their uncompressed size by four without changing values.
    """

    def __init__(
        self,
        rate_hz: float = 1.0,
        max_features: int = 4096,
        streams: Optional[Sequence[str]] = None,
        save_lossless_frames: bool = False,
        artifact_name: str = "online_sift_v1",
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("Feature rate must be positive")
        self.rate_hz = rate_hz
        self.period_ns = int(1.0e9 / rate_hz)
        self.max_features = max_features
        self.streams = set(streams) if streams else None
        self.save_lossless_frames = save_lossless_frames
        self.artifact_name = artifact_name
        self.extractor = cv2.SIFT_create(nfeatures=max_features)
        self.last_sample_ns = {}
        self.record_counts = {}
        self._connection = None
        self._path = None
        self._manifest_path = None
        self._manifest = None

    def start(self, session_path: Path, session_id: str, origin_ns: int) -> None:
        self._path = Path(session_path) / "evidence" / self.artifact_name
        self._path.mkdir(parents=True, exist_ok=False)
        self._manifest_path = self._path / "manifest.json"
        self._manifest = {
            "schema_version": "1.0",
            "status": "recording",
            "evidence_class": "online_raw_frame_features",
            "source_session_id": session_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "extractor": {
                "name": "opencv_sift",
                "opencv_version": cv2.__version__,
                "max_features": self.max_features,
                "input": "raw_capture_bgr_converted_to_gray",
            },
            "sampling": {"rate_hz": self.rate_hz, "streams": sorted(self.streams) if self.streams else "all"},
            "lossless_frames_included": self.save_lossless_frames,
            "database": "features.sqlite3",
        }
        _write_json_atomic(self._manifest_path, self._manifest)
        self._connection = sqlite3.connect(str(self._path / "features.sqlite3"))
        self._connection.execute(
            """
            CREATE TABLE observations (
                stream_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                session_time_ns INTEGER NOT NULL,
                host_capture_time_ns INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                frame_sha256 TEXT NOT NULL,
                keypoint_count INTEGER NOT NULL,
                keypoints_shape TEXT NOT NULL,
                keypoints_zlib BLOB NOT NULL,
                descriptors_shape TEXT,
                descriptors_stored_dtype TEXT,
                descriptors_original_dtype TEXT,
                descriptors_zlib BLOB,
                lossless_png BLOB,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (stream_id, frame_index)
            )
            """
        )
        self._connection.commit()

    def _should_sample(self, stream_id: str, capture_ns: int) -> bool:
        if self.streams is not None and stream_id not in self.streams:
            return False
        previous = self.last_sample_ns.get(stream_id)
        if previous is not None and capture_ns - previous < self.period_ns:
            return False
        self.last_sample_ns[stream_id] = capture_ns
        return True

    def process(self, packet, frame_index: int, session_time_ns: int) -> None:
        if not self._should_sample(packet.stream_id, packet.host_capture_time_ns):
            return
        gray = cv2.cvtColor(packet.image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.extractor.detectAndCompute(gray, None)
        keypoint_array = np.asarray(
            [
                (
                    point.pt[0], point.pt[1], point.size, point.angle,
                    point.response, float(point.octave), float(point.class_id),
                )
                for point in keypoints
            ],
            dtype=np.float32,
        ).reshape((-1, 7))

        descriptor_shape = None
        stored_dtype = None
        original_dtype = None
        descriptor_blob = None
        if descriptors is not None:
            descriptors = np.ascontiguousarray(descriptors)
            original_dtype = str(descriptors.dtype)
            rounded = np.rint(descriptors)
            if (
                descriptors.size == 0
                or (
                    float(descriptors.min()) >= 0.0
                    and float(descriptors.max()) <= 255.0
                    and np.array_equal(descriptors, rounded)
                )
            ):
                stored = rounded.astype(np.uint8)
            else:
                stored = descriptors
            descriptor_shape = json.dumps(list(stored.shape))
            stored_dtype = str(stored.dtype)
            descriptor_blob = zlib.compress(stored.tobytes(), level=6)

        png = None
        if self.save_lossless_frames:
            ok, encoded = cv2.imencode(".png", packet.image)
            if not ok:
                raise RuntimeError("Cannot encode lossless feature frame")
            png = encoded.tobytes()

        height, width = packet.image.shape[:2]
        self._connection.execute(
            """
            INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet.stream_id,
                frame_index,
                session_time_ns,
                packet.host_capture_time_ns,
                width,
                height,
                hashlib.sha256(packet.image.tobytes()).hexdigest(),
                len(keypoints),
                json.dumps(list(keypoint_array.shape)),
                sqlite3.Binary(zlib.compress(keypoint_array.tobytes(), level=6)),
                descriptor_shape,
                stored_dtype,
                original_dtype,
                sqlite3.Binary(descriptor_blob) if descriptor_blob is not None else None,
                sqlite3.Binary(png) if png is not None else None,
                json.dumps(packet.metadata, separators=(",", ":")),
            ),
        )
        self._connection.commit()
        self.record_counts[packet.stream_id] = self.record_counts.get(packet.stream_id, 0) + 1

    def close(self, status: str = "complete") -> None:
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None
        self._manifest.update(
            {
                "status": status,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "record_counts": self.record_counts,
            }
        )
        _write_json_atomic(self._manifest_path, self._manifest)

    def describe(self) -> dict:
        return {
            "type": "OnlineSiftRecorder",
            "path": "evidence/{}/features.sqlite3".format(self.artifact_name),
            "evidence_class": "online_raw_frame_features",
            "rate_hz": self.rate_hz,
            "max_features": self.max_features,
            "save_lossless_frames": self.save_lossless_frames,
            "record_counts": self.record_counts,
        }
