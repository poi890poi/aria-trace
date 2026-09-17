"""UVC input bridge for the virtual HIK contract; no camera is opened on import."""
from __future__ import annotations

import math
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

from rig_runtime.adapters.rig.devices import CameraConfiguration, OpenCvCameraAdapter


def device_uri(device, width, height, fps, zoom=1.0, backend="dshow"):
    return "virtual-hik+uvc://{}?{}".format(device, urlencode([
        ("width", width), ("height", height), ("fps", fps),
        ("zoom", zoom), ("backend", backend),
    ]))


def camera_spec(device_id, configuration):
    parsed = urlparse(device_id)
    if (
        not parsed.netloc.isascii() or not parsed.netloc.isdigit()
        or parsed.path not in ("", "/") or parsed.fragment
    ):
        raise ValueError("UVC virtual camera requires a non-negative camera index")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if set(query) - {"width", "height", "fps", "zoom", "backend"}:
        raise ValueError("Unsupported UVC virtual-camera option")
    def get(name, default):
        return (query.get(name) or [default])[-1]

    width = int(get("width", configuration.width_px))
    height = int(get("height", configuration.height_px))
    fps = float(get("fps", configuration.fps))
    zoom = float(get("zoom", 1.0))
    backend = str(get("backend", "dshow")).lower()
    if (
        width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0
        or not math.isfinite(zoom) or zoom < 1
    ):
        raise ValueError("UVC dimensions/FPS must be positive and zoom must be at least one")
    if backend not in ("auto", "dshow", "msmf"):
        raise ValueError("UVC backend must be auto, dshow, or msmf")
    device = str(int(parsed.netloc))
    canonical = device_uri(device, width, height, "{:.12g}".format(fps), "{:.12g}".format(zoom), backend)
    return dict(source_kind="uvc", device=device, backend=backend, width_px=width,
                height_px=height, fps=fps, zoom=zoom, canonical_device_id=canonical,
                serial=None, camera_id=device, camera_facing=None)


class UvcSource:
    """Present the existing OpenCV camera boundary as full decoded samples.

    The first real frame establishes geometry and is retained for the first read.
    Timestamps describe host reception, not an unavailable sensor exposure clock.
    """
    def __init__(self, device, *, width_px, height_px, fps, backend):
        self.configuration = CameraConfiguration(device, width_px, height_px, fps, backend)
        self.adapter = OpenCvCameraAdapter()
        self.effective_configuration = {}
        self._first = None

    def open(self):
        try:
            reported = self.adapter.open(self.configuration)
            self._first = self.adapter.read()
            height, width = self._first.image.shape[:2]
            self.effective_configuration = dict(
                effective=dict(width_px=width, height_px=height, fps=reported["fps"]),
                reported=reported,
            )
            return self
        except Exception:
            self.close()
            raise

    def read_sample(self):
        sample, self._first = self._first, None
        if sample is None:
            sample = self.adapter.read()
        return SimpleNamespace(image=sample.image, capture_time_ns=sample.time_ns,
                               receive_time_ns=sample.receive_time_ns,
                               metadata=dict(sample.metadata, timestamp_basis="host_receive_no_sensor_timestamp"))

    def close(self):
        self._first = None
        self.adapter.close()
