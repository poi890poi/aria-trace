import tempfile
import threading
import time
import unittest
import urllib.request
import json
import sqlite3
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from acquisition.models import FramePacket, InputPacket
from acquisition.features import OnlineSiftRecorder
from acquisition.annotations import AnnotationStore
from acquisition.extract_portal_init import extract_initialization
from acquisition.review import ReviewState, make_handler
from acquisition.session import SessionReader, SessionWriter
from acquisition.sources import estimate_clock_offset, parse_getevent_line
from acquisition.video import find_ffmpeg
from poc.evaluate_portal_initialization import first_consistent_window


try:
    TEST_FFMPEG = find_ffmpeg()
except RuntimeError:
    TEST_FFMPEG = None


class DescribedSource:
    def __init__(self, identifier, frame=True):
        self.identifier = identifier
        self.frame = frame

    def describe(self):
        return {
            "type": "test",
            "stream_id" if self.frame else "source_id": self.identifier,
        }


class AcquisitionTests(unittest.TestCase):
    def test_portal_confirmation_requires_consistent_consecutive_poses(self):
        rotation = np.eye(3)
        estimates = {
            0: (rotation, np.array([0.0, 0.0, 0.0])),
            1: (rotation, np.array([0.1, 0.0, 0.0])),
            2: (rotation, np.array([0.2, 0.0, 0.0])),
        }
        accepted = first_consistent_window(
            [0, 1, 2], estimates, np.zeros(3), portal_limit_m=1.0
        )
        self.assertEqual(accepted[1], [0, 1, 2])
        estimates[1] = (rotation, np.array([5.0, 0.0, 0.0]))
        self.assertIsNone(
            first_consistent_window(
                [0, 1, 2], estimates, np.zeros(3), portal_limit_m=1.0
            )
        )

    def test_parses_getevent_line(self):
        parsed = parse_getevent_line(
            "[  123.456789] /dev/input/event4: EV_ABS ABS_X 0000007f"
        )
        self.assertEqual(parsed["device"], "/dev/input/event4")
        self.assertEqual(parsed["event_type"], "EV_ABS")
        self.assertEqual(parsed["code"], "ABS_X")
        self.assertEqual(parsed["value"], "0000007f")
        self.assertEqual(parsed["device_time_ns"], 123456789000)

    def test_clock_mapping_uses_lowest_round_trip_sample(self):
        offset, rtt = estimate_clock_offset(
            [
                (1_000, 1_500, 50),
                (2_000, 2_100, 1_000),
                (3_000, 3_300, 1_900),
            ]
        )
        self.assertEqual(rtt, 100)
        self.assertEqual(offset, 1_050)

    def test_writes_reads_and_decodes_multistream_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session"
            writer = SessionWriter(
                path,
                [DescribedSource("main"), DescribedSource("reference")],
                [DescribedSource("gamepad", frame=False)],
                video_encoding="mjpeg",
            )
            origin = writer.origin_ns
            for index in range(3):
                for stream_id, level in (("main", 40), ("reference", 180)):
                    image = np.full((48, 64, 3), level + index, dtype=np.uint8)
                    timestamp = origin + index * 33_000_000
                    writer.write_frame(
                        FramePacket(
                            stream_id,
                            image,
                            timestamp,
                            timestamp + 1_000_000,
                            source_time_ns=index * 33_000_000,
                        )
                    )
            writer.write_input(
                InputPacket(
                    "gamepad",
                    "gamepad_state",
                    origin + 35_000_000,
                    {"axes": {"left_y": -1.0}, "buttons": {}},
                )
            )
            writer.close()

            reader = SessionReader(path)
            summary = reader.summary()
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["streams"]["main"]["frames"], 3)
            self.assertEqual(summary["streams"]["reference"]["frames"], 3)
            self.assertEqual(summary["input_events"], 1)
            self.assertEqual(len(reader.nearby_inputs(33_000_000)), 1)

            review = ReviewState(path)
            try:
                jpeg = review.frame_jpeg("main", 1)
                decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape[:2], (48, 64))
                self.assertEqual(review.frame_info("main", 1)["frame"]["frame_index"], 1)

                server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(review))
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                try:
                    base = "http://127.0.0.1:{}".format(server.server_address[1])
                    descriptor = json.loads(urllib.request.urlopen(base + "/api/session").read())
                    self.assertEqual(descriptor["frame_counts"]["main"], 3)
                    response = urllib.request.urlopen(base + "/api/frame.jpg?stream=main&index=1")
                    self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                    self.assertGreater(len(response.read()), 100)
                    body = json.dumps(
                        {
                            "kind": "world_ready",
                            "stream_id": "main",
                            "frame_index": 1,
                            "portal_id": "portal-test",
                            "route_id": "route-test",
                        }
                    ).encode("utf-8")
                    request = urllib.request.Request(
                        base + "/api/annotations",
                        data=body,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    created = json.loads(urllib.request.urlopen(request).read())
                    annotations = json.loads(
                        urllib.request.urlopen(base + "/api/annotations").read()
                    )
                    self.assertEqual(annotations[0]["frame_index"], 1)
                    delete = urllib.request.Request(
                        base + "/api/annotations?id=" + created["annotation_id"],
                        method="DELETE",
                    )
                    urllib.request.urlopen(delete).read()
                    annotations = json.loads(
                        urllib.request.urlopen(base + "/api/annotations").read()
                    )
                    self.assertEqual(annotations, [])
                finally:
                    server.shutdown()
                    server.server_close()
                    server_thread.join(timeout=2)
            finally:
                review.close()

    @unittest.skipIf(TEST_FFMPEG is None, "FFmpeg is not installed")
    def test_h264_session_decodes_by_frame_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session"
            writer = SessionWriter(
                path,
                [DescribedSource("main")],
                [],
                video_encoding="h264",
                ffmpeg=TEST_FFMPEG,
            )
            for index in range(4):
                timestamp = writer.origin_ns + index * 33_000_000
                image = np.full((48, 64, 3), 40 + index * 30, dtype=np.uint8)
                writer.write_frame(FramePacket("main", image, timestamp, timestamp))
            writer.close()

            reader = SessionReader(path)
            self.assertEqual(reader.manifest["video_streams"]["main"]["encoding"], "h264")
            self.assertEqual(reader.summary()["streams"]["main"]["frames"], 4)
            review = ReviewState(path)
            try:
                jpeg = review.frame_jpeg("main", 3)
                decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape[:2], (48, 64))
            finally:
                review.close()

    def test_online_sift_is_sampled_from_raw_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session"
            processor = OnlineSiftRecorder(
                rate_hz=1.0,
                max_features=128,
                save_lossless_frames=True,
            )
            writer = SessionWriter(
                path,
                [DescribedSource("main")],
                [],
                video_encoding="mjpeg",
                frame_processors=[processor],
            )
            random = np.random.RandomState(7)
            images = [random.randint(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(3)]
            for index, offset_ns in enumerate((0, 500_000_000, 1_100_000_000)):
                timestamp = writer.origin_ns + offset_ns
                writer.write_frame(FramePacket("main", images[index], timestamp, timestamp))
            writer.close()

            database = path / "evidence" / "online_sift_v1" / "features.sqlite3"
            connection = sqlite3.connect(str(database))
            try:
                rows = connection.execute(
                    "SELECT frame_index, lossless_png, descriptors_stored_dtype "
                    "FROM observations ORDER BY frame_index"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual([row[0] for row in rows], [0, 2])
            decoded = cv2.imdecode(np.frombuffer(rows[1][1], dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(decoded, images[2]))
            self.assertIn(rows[1][2], ("uint8", None))
            manifest = SessionReader(path).manifest
            self.assertEqual(
                manifest["online_frame_artifacts"][0]["evidence_class"],
                "online_raw_frame_features",
            )
            reader = SessionReader(path)
            feature_info = reader.online_features_for_frame("main", 2)
            self.assertEqual(feature_info[0]["keypoint_count"] > 0, True)
            self.assertEqual(feature_info[0]["has_lossless_frame"], True)

            annotations = AnnotationStore(path)
            frames = reader.frames_by_stream["main"]
            annotations.add(
                "world_ready", frames[0]["session_time_ns"], "main", 0,
                portal_id="portal-test", route_id="route-test",
            )
            annotations.add(
                "route_start", frames[2]["session_time_ns"], "main", 2,
                portal_id="portal-test", route_id="route-test",
            )
            extracted = extract_initialization(
                path,
                Path(temporary) / "portal-init",
                "portal-test",
                "route-test",
                require_lossless=True,
            )
            self.assertEqual(extracted["frame_count"], 2)
            self.assertEqual(extracted["source_quality"], ["raw_lossless_evidence"])
            extracted_image = cv2.imread(
                str(Path(temporary) / "portal-init" / extracted["frames"][1]["image"])
            )
            self.assertTrue(np.array_equal(extracted_image, images[2]))


if __name__ == "__main__":
    unittest.main()
