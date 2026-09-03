import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.request
import json
import queue
import sqlite3
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from acquisition.models import FramePacket, InputPacket
from acquisition.recorder import AcquisitionRecorder
from acquisition.record import default_adb
from acquisition.features import OnlineSiftRecorder
from acquisition.annotations import AnnotationStore
from acquisition.extract_portal_init import extract_initialization
from acquisition.review import ReviewState, make_handler
from acquisition.session import SessionReader, SessionWriter
from acquisition.sources import (
    AdbClockMapper,
    AdbScreenshotFrameSource,
    estimate_clock_offset,
    parse_getevent_line,
)
from acquisition.android_capture import (
    AndroidRoiFrameSource,
    parse_android_roi,
)
from acquisition.video import find_ffmpeg
from acquisition.windows import (
    WindowsKeyboardMouseSource,
    WindowsWindowFrameSource,
    select_window,
)
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


class ContinuousFrameSource:
    stream_id = "main"

    def __init__(self):
        self.running = False
        self.index = 0

    def start(self):
        self.running = True

    def read(self):
        if not self.running:
            return None
        time.sleep(0.002)
        now = time.perf_counter_ns()
        self.index += 1
        return FramePacket(
            "main",
            np.full((32, 32, 3), self.index % 255, dtype=np.uint8),
            now,
            now,
        )

    def stop(self):
        self.running = False

    def describe(self):
        return {"type": "continuous-test", "stream_id": "main"}


class DiagnosticInputSource:
    source_id = "diagnostic-input"

    def __init__(self):
        self.finalized = False

    def start(self, emit):
        now = time.perf_counter_ns()
        emit(InputPacket(self.source_id, "test_input", now, {"active": True}))

    def stop(self):
        self.finalized = True

    def describe(self):
        return {
            "type": "diagnostic-test-input",
            "source_id": self.source_id,
            "finalized": self.finalized,
        }


class DelayedInputSource:
    source_id = "delayed-input"

    def __init__(self, delay_s=0.05):
        self.delay_s = float(delay_s)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self, emit):
        def run():
            if not self.stop_event.wait(self.delay_s):
                now = time.perf_counter_ns()
                emit(
                    InputPacket(
                        self.source_id,
                        "pc_raw_keyboard",
                        now,
                        {"pressed": True, "foreground": True},
                    )
                )

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def describe(self):
        return {"type": "delayed-test-input", "source_id": self.source_id}


class FakeWindowsApi:
    def __init__(self):
        self.snapshots = [
            {
                "foreground": False,
                "keys": [],
                "buttons": [],
                "cursor_client": (5, 7),
                "cursor_normalized": (0.05, 0.07),
            },
            {
                "foreground": True,
                "keys": [(87, "W")],
                "buttons": ["left"],
                "cursor_client": (20, 30),
                "cursor_normalized": (0.2, 0.3),
            },
        ]
        self.snapshot_index = 0

    def list_windows(self):
        return [(17, "AriaTrace Test Game")]

    def capture_client(self, hwnd):
        if hwnd != 17:
            raise RuntimeError("unexpected window")
        return np.full((48, 64, 3), 91, dtype=np.uint8), (100, 200, 64, 48)

    def input_snapshot(self, hwnd):
        if hwnd != 17:
            raise RuntimeError("unexpected window")
        index = min(self.snapshot_index, len(self.snapshots) - 1)
        self.snapshot_index += 1
        return self.snapshots[index]


class AcquisitionTests(unittest.TestCase):
    def test_parses_android_roi_with_target_and_quality(self):
        spec = parse_android_roi("minimap=20,30,400,360,256,224,17")
        self.assertEqual(spec.stream_id, "minimap")
        self.assertEqual((spec.x, spec.y, spec.width, spec.height), (20, 30, 400, 360))
        self.assertEqual((spec.output_width, spec.output_height), (256, 224))
        self.assertEqual(spec.crf, 17)
        with self.assertRaisesRegex(ValueError, "4, 6, or 7"):
            parse_android_roi("bad=1,2,3")
        with self.assertRaisesRegex(ValueError, "between 0 and 51"):
            parse_android_roi("bad=0,0,10,10,8,8,60")

    def test_android_roi_preserves_shared_timestamp_and_resizes(self):
        class FakeHub:
            def __init__(self):
                self.items = {}

            def register(self, stream_id):
                result = queue.Queue()
                self.items[stream_id] = result
                return result

            def start(self):
                pass

            def stop(self):
                pass

            def take_drops(self, stream_id):
                return 2

            def describe(self):
                return {"type": "fake"}

        hub = FakeHub()
        source = AndroidRoiFrameSource(
            hub,
            parse_android_roi("roi=2,3,8,6,4,2,20"),
            image_space_context={
                "natural_size_px": [12, 16],
                "quarter_turns_clockwise_from_natural": 1,
                "source": "unit_test",
            },
        )
        source.start()
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape((12, 16, 3))
        hub.items["roi"].put(("frame", 7, image, 111, 222, 333))
        packet = source.read()
        self.assertEqual(packet.image.shape, (2, 4, 3))
        self.assertEqual(packet.source_time_ns, 111)
        self.assertEqual(packet.host_capture_time_ns, 222)
        self.assertEqual(packet.host_receive_time_ns, 333)
        self.assertEqual(packet.metadata["source_sequence"], 7)
        self.assertEqual(
            "android_phone_natural_display_pixels",
            packet.metadata["image_space"]["canonical_space_id"],
        )
        np.testing.assert_allclose(
            packet.metadata["image_space"]["local_to_canonical_3x3"],
            [[0, 3, 4], [-2, 0, 12.5], [0, 0, 1]],
        )
        self.assertEqual(packet.dropped_before, 2)

    def test_deferred_recording_starts_at_first_qualifying_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            started = []
            manifest = AcquisitionRecorder(
                Path(temporary) / "first-input-start",
                [ContinuousFrameSource()],
                [DelayedInputSource()],
                video_encoding="mjpeg",
            ).run(
                duration_s=0.05,
                start_on_input=True,
                input_start_predicate=lambda packet: packet.kind
                == "pc_raw_keyboard",
                on_recording_started=lambda packet: started.append(packet.kind),
            )
            reader = SessionReader(Path(temporary) / "first-input-start")
            self.assertEqual(started, ["pc_raw_keyboard"])
            self.assertTrue(manifest["recording_start"]["started"])
            self.assertEqual(
                manifest["recording_start"]["first_input_kind"],
                "pc_raw_keyboard",
            )
            self.assertEqual(reader.inputs[0]["session_time_ns"], 0)
            self.assertTrue(reader.frames_by_stream["main"])
            self.assertGreaterEqual(
                reader.frames_by_stream["main"][0]["session_time_ns"], 0
            )
            self.assertTrue(
                all(
                    frame["session_time_ns"] >= 0
                    for frame in reader.frames_by_stream["main"]
                )
            )

    def test_deferred_recording_can_start_on_timer_without_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            started = []
            manifest = AcquisitionRecorder(
                Path(temporary) / "timer-start",
                [ContinuousFrameSource()],
                [],
                video_encoding="mjpeg",
            ).run(
                duration_s=0.04,
                start_after_delay_s=0.02,
                on_recording_started=lambda packet: started.append(packet),
            )
            reader = SessionReader(Path(temporary) / "timer-start")
            self.assertEqual(started, [None])
            self.assertEqual(
                manifest["recording_start"]["policy"], "settled_timer"
            )
            self.assertEqual(
                manifest["recording_start"]["automatic_start_delay_s"], 0.02
            )
            self.assertTrue(reader.frames_by_stream["main"])
            self.assertFalse(reader.inputs)

    def test_recorder_persists_final_input_source_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_source = DiagnosticInputSource()
            manifest = AcquisitionRecorder(
                Path(temporary) / "diagnostics",
                [ContinuousFrameSource()],
                [input_source],
                video_encoding="mjpeg",
            ).run(duration_s=0.02)
            self.assertTrue(manifest["input_sources"][0]["finalized"])
            self.assertEqual(
                manifest["input_counts"]["diagnostic-input:test_input"],
                1,
            )

    def test_recorder_signals_sources_ready_before_external_take_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            started = threading.Event()
            external_stop = threading.Event()
            recorder = AcquisitionRecorder(
                Path(temporary) / "prearmed",
                [ContinuousFrameSource()],
                [],
                video_encoding="mjpeg",
            )
            result = {}

            def run():
                result["manifest"] = recorder.run(
                    external_stop=external_stop,
                    started_event=started,
                )

            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(started.wait(2))
            external_stop.set()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["manifest"]["status"], "complete")
            self.assertGreater(
                result["manifest"]["frame_counts"].get("main", 0), 0
            )

    def test_selects_one_window_and_rejects_ambiguous_titles(self):
        windows = [(1, "Game - Alpha"), (2, "Game - Beta")]
        self.assertEqual(select_window(windows, "Game - Alpha"), windows[0])
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            select_window(windows, "Game")
        with self.assertRaisesRegex(RuntimeError, "No visible window"):
            select_window(windows, "Missing")

    def test_windows_frame_and_input_sources_emit_session_packets(self):
        api = FakeWindowsApi()
        frames = WindowsWindowFrameSource(
            "Test Game", fps=1000.0, api=api
        )
        frames.start()
        try:
            packet = frames.read()
        finally:
            frames.stop()
        self.assertEqual(packet.stream_id, "main")
        self.assertEqual(packet.image.shape, (48, 64, 3))
        self.assertEqual(packet.metadata["client_screen_rect"], [100, 200, 64, 48])
        self.assertEqual(frames.describe()["matched_window_title"], "AriaTrace Test Game")

        events = []
        inputs = WindowsKeyboardMouseSource(
            "Test Game", poll_hz=1000.0, api=api
        )
        inputs.start(events.append)
        deadline = time.time() + 1.0
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.005)
        inputs.stop()
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0].kind, "pc_input_state")
        self.assertFalse(events[0].payload["foreground"])
        self.assertEqual(events[1].payload["keys"][0]["name"], "W")
        self.assertEqual(events[1].payload["mouse_buttons"], ["left"])

    def test_tool_discovery_has_no_machine_specific_fallback(self):
        with mock.patch("aria_trace.apps.record.shutil.which", return_value=None):
            self.assertIsNone(default_adb())
        with mock.patch("rig_runtime.adapters.filesystem.video.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Install it, pass --ffmpeg"):
                find_ffmpeg()

    def test_adb_screenshot_source_prefers_lossless_image_series(self):
        description = AdbScreenshotFrameSource(Path("adb.exe")).describe()
        self.assertEqual("image_series", description["preferred_frame_storage"])
        self.assertEqual("png", description["preferred_image_format"])
        self.assertFalse(description["external_ffmpeg_required"])
        source = DescribedSource("adb")
        source.describe = lambda: {
            "type": "adb_screenshot",
            "stream_id": "adb",
            "preferred_frame_storage": "image_series",
            "preferred_image_format": "png",
            "external_ffmpeg_required": False,
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "rig_runtime.adapters.filesystem.video.find_ffmpeg"
        ) as find:
            writer = SessionWriter(Path(temporary) / "session", [source], [])
            now = time.perf_counter_ns()
            writer.write_frame(
                FramePacket(
                    "adb",
                    np.full((16, 16, 3), 80, np.uint8),
                    now,
                    now,
                )
            )
            writer.close()
            reader = SessionReader(Path(temporary) / "session")
            manifest = reader.manifest
            records = reader.frames_by_stream["adb"]
            decoded = reader.read_image_frames(records)
        find.assert_not_called()
        self.assertNotIn("adb", manifest["videos"])
        self.assertNotIn("adb", manifest["video_streams"])
        self.assertEqual("png", manifest["image_streams"]["adb"]["format"])
        self.assertEqual(
            "image_series",
            manifest["frame_storage"]["stream_options"]["adb"]["storage"],
        )
        self.assertEqual((1, 16, 16, 3), decoded.shape)
        with tempfile.TemporaryDirectory() as temporary:
            explicit = SessionWriter(
                Path(temporary) / "session",
                [source],
                [],
                video_stream_options={"adb": {"encoding": "h264"}},
            )
            self.assertEqual(
                "h264", explicit.video_stream_options["adb"]["encoding"]
            )
            explicit.close()
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

    def test_clock_mapping_keeps_compatible_proc_uptime_epoch(self):
        mapper = AdbClockMapper(Path("adb"), maximum_plausible_lag_ns=1_000)
        mapper.offset_ns = 900
        mapper.rtt_ns = 20
        mapper.status = "mapped-from-proc-uptime-unverified"

        self.assertEqual(1_000, mapper.to_host_time_ns(100, 1_100))
        self.assertEqual("mapped-from-proc-uptime-verified", mapper.status)
        self.assertIsNone(mapper.observed_offset_ns)

    def test_clock_mapping_rebases_incompatible_media_pts_epoch(self):
        mapper = AdbClockMapper(Path("adb"), maximum_plausible_lag_ns=1_000)
        mapper.offset_ns = -98_900_000
        mapper.rtt_ns = 20
        mapper.status = "mapped-from-proc-uptime-unverified"

        first = mapper.to_host_time_ns(25_700_000, 2_300_000)
        second = mapper.to_host_time_ns(25_734_000, 2_340_000)

        self.assertEqual(2_300_000, first)
        self.assertEqual(2_334_000, second)
        self.assertEqual("mapped-from-observed-source-time", mapper.status)
        self.assertEqual(-23_400_000, mapper.observed_offset_ns)
        self.assertGreaterEqual(second, first)

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
