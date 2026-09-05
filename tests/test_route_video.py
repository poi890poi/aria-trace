import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from aria_trace.evidence.route_video import RouteTraceVideoRecorder
from aria_trace.services.tracking.runtime import render_route_trace_video_frame


class RouteTraceVideoTests(unittest.TestCase):
    @staticmethod
    def _calibration():
        return {
            "coordinate_space": {
                "space_id": "minimap-crop",
                "space_type": "raster",
                "dimensions_wh": [40, 40],
                "origin": "top-left",
                "x_axis": "right",
                "y_axis": "down",
                "pixel_convention": "pixel-center",
            },
            "outer_boundary": {"center_x": 20, "center_y": 20, "radius": 18},
            "rotation_center": {"x": 20, "y": 20},
        }

    @staticmethod
    def _state():
        return {
            "mode": "TRACK",
            "pose": {
                "x": 50.0,
                "y": 50.0,
                "map_alignment_deg": 0.0,
                "player_heading_map_deg": 0.0,
            },
            "map_scale": 1.0,
            "trail": [[48.0, 50.0], [49.0, 50.0]],
        }

    def test_composition_preserves_shape_and_changes_overlay_regions(self):
        frame = np.full((180, 320, 3), 80, np.uint8)
        mosaic = np.full((120, 120, 3), 130, np.uint8)
        rendered = render_route_trace_video_frame(
            frame,
            mosaic,
            self._state(),
            [[50, 50], [60, 50], [70, 55]],
            self._calibration(),
            [10, 10, 40, 40],
        )
        self.assertEqual(frame.shape, rendered.shape)
        self.assertFalse(np.array_equal(frame[10:50, 10:50], rendered[10:50, 10:50]))
        self.assertFalse(np.array_equal(frame[10:130, 134:314], rendered[10:130, 134:314]))
        self.assertTrue(np.array_equal(frame[150:170, 10:40], rendered[150:170, 10:40]))

    def test_async_recorder_writes_video_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = RouteTraceVideoRecorder(
                root,
                lambda image, state: image,
                fps=10.0,
                encoding="mjpeg",
                queue_capacity=8,
            )
            recorder.update_state({"pose": {"x": 1}})
            for index in range(4):
                recorder.submit(
                    np.full((48, 64, 3), index * 30, np.uint8),
                    1_000_000_000 + index * 100_000_000,
                )
                time.sleep(0.01)
            summary = recorder.close()
            self.assertEqual("complete", summary["status"])
            self.assertEqual(4, summary["source_frames"])
            self.assertGreaterEqual(summary["written_frames"], 4)
            video = root / summary["video_file"]
            self.assertTrue(video.is_file())
            capture = cv2.VideoCapture(str(video))
            try:
                self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            finally:
                capture.release()
            manifest = json.loads(
                (root / "route_trace_video.json").read_text(encoding="utf-8")
            )
            self.assertEqual("complete", manifest["status"])
            self.assertFalse(
                manifest["composition"]["desktop_or_workbench_included"]
            )


if __name__ == "__main__":
    unittest.main()
