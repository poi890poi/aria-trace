import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.annotations import AnnotationStore
from acquisition.models import FramePacket, InputPacket
from acquisition.session import SessionReader, SessionWriter
from replay.alignment import align_session
from replay.package import ReplayPackage, compile_replay_package


class DescribedSource:
    def __init__(self, identifier, frame=True):
        self.identifier = identifier
        self.frame = frame

    def describe(self):
        return {
            "type": "test",
            "stream_id" if self.frame else "source_id": self.identifier,
        }


def route_images(count=24):
    random = np.random.RandomState(17)
    panorama = random.randint(0, 256, (48, 64 + count * 4, 3), dtype=np.uint8)
    panorama = cv2.GaussianBlur(panorama, (5, 5), 0)
    images = []
    for index in range(count):
        image = panorama[:, index * 4 : index * 4 + 64].copy()
        cv2.line(image, (index * 3 % 64, 0), ((index * 3 + 19) % 64, 47), (255, 255, 255), 2)
        cv2.circle(image, ((index * 7 + 8) % 64, 24), 5, (20, 20, 20), -1)
        images.append(image)
    return images


def write_session(path, images, source_indices=None, distractor_index=None):
    writer = SessionWriter(
        path,
        [DescribedSource("main")],
        [DescribedSource("gamepad", frame=False)],
        video_encoding="mjpeg",
        video_fps=10.0,
    )
    source_indices = source_indices or list(range(len(images)))
    random = np.random.RandomState(29)
    for frame_index, source_index in enumerate(source_indices):
        image = images[source_index].copy()
        if source_indices != list(range(len(images))):
            image = np.clip(image.astype(np.int16) + 18, 0, 255).astype(np.uint8)
            noise = random.normal(0, 2.0, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if frame_index == distractor_index:
            image = random.randint(0, 256, image.shape, dtype=np.uint8)
        timestamp = writer.origin_ns + frame_index * 100_000_000
        writer.write_frame(FramePacket("main", image, timestamp, timestamp + 1_000_000))
        if frame_index % 3 == 0:
            writer.write_input(
                InputPacket(
                    "gamepad",
                    "gamepad_state",
                    timestamp + 10_000_000,
                    {"axes": {"left_y": -1.0, "right_x": frame_index / 100.0}},
                )
            )
    writer.close()
    return SessionReader(path)


def annotate_route(path, reader, stage_frames):
    frames = reader.frames_by_stream["main"]
    store = AnnotationStore(path)
    store.add(
        "route_start",
        frames[0]["session_time_ns"],
        "main",
        frames[0]["frame_index"],
        portal_id="portal-a",
        route_id="route-a",
        note="approach",
    )
    for frame_index, label in stage_frames:
        store.add(
            "route_stage",
            frames[frame_index]["session_time_ns"],
            "main",
            frames[frame_index]["frame_index"],
            portal_id="portal-a",
            route_id="route-a",
            note=label,
        )
    store.add(
        "route_complete",
        frames[-1]["session_time_ns"],
        "main",
        frames[-1]["frame_index"],
        portal_id="portal-a",
        route_id="route-a",
    )


class ReplayTests(unittest.TestCase):
    def test_compiles_versioned_replay_package_with_action_priors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = route_images()
            session_path = root / "demo"
            reader = write_session(session_path, images)
            annotate_route(session_path, reader, [(8, "corridor"), (16, "door")])

            manifest = compile_replay_package(
                session_path, root / "package", "main", "route-a", reference_rate_hz=10.0
            )
            package = ReplayPackage(root / "package")
            self.assertEqual(manifest["visual_source_quality"], "decoded_primary_video")
            self.assertEqual(manifest["action_semantics"], "recorded_input_prior_not_timed_macro")
            self.assertEqual(manifest["counts"]["references"], 24)
            self.assertEqual([stage["label"] for stage in package.stages], ["approach", "corridor", "door"])
            self.assertEqual(len(package.action_priors), 23)
            self.assertGreater(manifest["counts"]["input_events"], 0)
            self.assertEqual(package.descriptors.shape[0], 24)
            self.assertTrue(manifest["source_session"]["manifest_sha256"])

    def test_aligns_time_warped_route_and_rejects_distractor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = route_images()
            demo_path = root / "demo"
            demo_reader = write_session(demo_path, images)
            annotate_route(demo_path, demo_reader, [(8, "corridor"), (16, "door")])
            compile_replay_package(
                demo_path, root / "package", "main", "route-a", reference_rate_hz=10.0
            )

            mapping = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            distractor_index = 14
            query_path = root / "query"
            query_reader = write_session(
                query_path,
                images,
                source_indices=mapping,
                distractor_index=distractor_index,
            )
            annotate_route(
                query_path,
                query_reader,
                [(mapping.index(8), "corridor"), (mapping.index(16), "door")],
            )
            summary = align_session(
                root / "package",
                query_path,
                root / "alignment",
                query_rate_hz=10.0,
                distance_threshold=0.35,
            )
            records = [
                json.loads(line)
                for line in (root / "alignment" / "alignment.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            estimated = [record["reference_index"] for record in records]
            self.assertTrue(summary["monotonic"])
            self.assertEqual(summary["final_progress"], 1.0)
            self.assertLessEqual(float(np.median(np.abs(np.array(estimated) - mapping))), 1.0)
            self.assertGreaterEqual(summary["stage_label_accuracy"], 0.9)
            self.assertFalse(records[distractor_index]["accepted"])
            self.assertGreater(summary["accepted_fraction"], 0.8)

    def test_compiler_requires_observed_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = route_images(4)
            reader = write_session(root / "demo", images)
            frame = reader.frames_by_stream["main"][0]
            AnnotationStore(root / "demo").add(
                "route_start",
                frame["session_time_ns"],
                "main",
                0,
                route_id="route-a",
            )
            with self.assertRaisesRegex(RuntimeError, "route_complete"):
                compile_replay_package(
                    root / "demo", root / "package", "main", "route-a", 10.0
                )


if __name__ == "__main__":
    unittest.main()

