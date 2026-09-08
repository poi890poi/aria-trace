"""Known raster pivots, independent of the estimator's circle scores."""

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from rig_runtime.domain.spatial import bind_geometry, raster_space
from rig_runtime.services.calibration.minimap.calibration import (
    _cursor_temporal_center,
    calibrate_cursor_orbit_frames,
)


def rotating_cursor_frames(angles, pivot=(62.3, 58.7), shape="dot", size=1.0, seed=1701, noise=0):
    scale, side = 4, 128
    rng = np.random.default_rng(seed)
    background = cv2.GaussianBlur(rng.uniform(20, 50, (side, side, 3)).astype(np.float32), (5, 5), 1)
    triangle = size * np.array([[-7., -5.], [-7., 5.], [10., 0.]])
    frames = []
    for angle in angles:
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        mask = np.zeros((side * scale, side * scale), np.uint8)
        # INTER_AREA pixel centers are (scale - 1) / 2 in the larger raster.
        # Preserve the independently specified pivot when reducing resolution.
        if shape == "triangle":
            points = (triangle @ rotation.T + pivot) * scale + (scale - 1) / 2
            cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 255)
        else:
            center = np.asarray(pivot) + size * 7 * np.array([np.cos(a), np.sin(a)])
            center = np.rint(center * scale + (scale - 1) / 2).astype(int)
            cv2.circle(mask, tuple(center), round(size * 4 * scale), 255, -1)
        alpha = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)[..., None] / 255.
        frame = background * (1 - alpha) + np.array([25, 30, 235]) * alpha
        if noise:
            frame += rng.normal(0, noise, frame.shape)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return np.stack(frames)


class CursorColdCircleTests(unittest.TestCase):
    def fit(self, frames, initial=(64, 64)):
        return _cursor_temporal_center(frames, np.asarray(initial), 12)["metrics"]

    def test_incomplete_cold_disc_locates_offset_pivot_without_opposite_samples(self):
        pivot = (62.3, 58.7)
        frames = rotating_cursor_frames([0, 25, 55, 85, 125, 160, 205], pivot)
        result = self.fit(frames)
        self.assertLess(np.linalg.norm(np.array([result["x"], result["y"]]) - pivot), 1.0)
        self.assertGreater(np.linalg.norm(np.array([result["x"], result["y"]]) - [64, 64]), 3)

    def test_repeated_and_reordered_directions_do_not_move_the_pivot(self):
        frames = rotating_cursor_frames([5, 33, 69, 102, 147, 191, 238], shape="triangle")
        sparse = self.fit(frames)
        uneven = self.fit(frames[[0] * 50 + [4, 6, 2, 1, 5, 3, 0]])
        for key in ("x", "y", "cold_core_radius_px", "confidence"):
            self.assertEqual(sparse[key], uneven[key], key)

    def test_incomplete_disc_survives_public_calibration_and_saved_geometry(self):
        pivot = (62.3, 58.7)
        frames = rotating_cursor_frames([0] * 30 + [25, 55, 85, 125, 160, 205], pivot)
        space = raster_space("current_minimap_crop_pixels", [128, 128])
        boundary = bind_geometry({"center_x": 64., "center_y": 64., "radius": 55.}, "circle", space)
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_cursor_orbit_frames(frames, Path(temporary), outer_boundary=boundary, frame_space=space)
            self.assertEqual("available", result["capabilities"]["rotation_center"]["status"])
            self.assertEqual(boundary, result["outer_boundary"])
            self.assertEqual(space, result["rotation_center"]["space"])
            with np.load(Path(temporary) / "model.npz") as model:
                self.assertLess(np.linalg.norm(model["rotation_center"] - pivot), 1.0)
            self.assertIsNotNone(cv2.imread(str(Path(temporary) / "cursor_center_hough.jpg")))

    def test_fresh_pivots_sizes_and_missing_sectors(self):
        # Fixed after selecting cold-facing Hough votes on the development
        # sequence. Truth is used only here, never passed to the estimator.
        for shape in ("dot", "triangle"):
            for pivot, size, angles in (
                ((66.6, 61.2), 0.85, [17, 44, 76, 113, 159, 207, 253]),
                ((59.8, 65.4), 1.3, [83, 115, 154, 191, 226, 271, 316]),
            ):
                with self.subTest(shape=shape, pivot=pivot, size=size):
                    frames = rotating_cursor_frames(angles, pivot, shape, size, seed=8307, noise=0.6)
                    result = self.fit(frames)
                    self.assertLess(np.linalg.norm(np.array([result["x"], result["y"]]) - pivot), 1.0)

    def test_static_and_uniform_flicker_do_not_invent_a_pivot(self):
        static = rotating_cursor_frames([0] * 10)
        flicker = np.stack([np.full((128, 128, 3), 30 + index, np.uint8) for index in range(10)])
        for frames in (static, flicker):
            with self.subTest(kind="static" if frames is static else "flicker"):
                with self.assertRaisesRegex(RuntimeError, "no observable temporal signal"):
                    self.fit(frames)

    def test_noise_does_not_invent_a_center_on_static_or_empty_frames(self):
        for seed in (9013, 9029):
            for kind in ("static", "empty"):
                with self.subTest(seed=seed, kind=kind):
                    rng = np.random.default_rng(seed + 70)
                    frames = rotating_cursor_frames([17] * 126, shape="triangle", seed=seed, noise=6)
                    if kind == "empty":
                        frames = np.clip(35 + rng.normal(0, 6, frames.shape), 0, 255).astype(np.uint8)
                    for probability in (0, .003):
                        noisy = frames.copy()
                        hot = rng.random(noisy.shape[:3]) < probability
                        noisy[hot] = rng.integers(0, 256, (int(hot.sum()), 3), dtype=np.uint8)
                        with self.subTest(impulses=probability):
                            with self.assertRaisesRegex(RuntimeError, "no observable temporal signal"):
                                self.fit(noisy)

    def test_noisy_long_dwell_preserves_single_frame_rare_directions(self):
        for seed, pivot, shape in ((9013, (66.6, 61.2), "dot"), (9029, (59.8, 65.4), "triangle")):
            with self.subTest(seed=seed, shape=shape):
                frames = rotating_cursor_frames([17] * 120 + [44, 76, 113, 159, 207, 253],
                                               pivot=pivot, shape=shape, seed=seed, noise=6)
                rng = np.random.default_rng(seed + 70)
                hot = rng.random(frames.shape[:3]) < .003
                frames[hot] = rng.integers(0, 256, (int(hot.sum()), 3), dtype=np.uint8)
                result = self.fit(frames)
                self.assertLess(np.linalg.norm(np.array([result["x"], result["y"]]) - pivot), 1.0)
                reordered = self.fit(frames[::-1])
                for key in ("x", "y", "cold_core_radius_px", "confidence"):
                    self.assertEqual(result[key], reordered[key], key)

    def test_sparse_impulses_at_static_edges_are_not_a_rotation_circle(self):
        for shape in ("dot", "triangle"):
            for seed, pivot, size, dwell, angle in ((19031, (61.1, 66.8), .9, 200, 61),
                                                   (41047, (59.1, 62.2), 1.1, 30, 53)):
                with self.subTest(shape=shape, seed=seed):
                    frames = rotating_cursor_frames([angle] * (dwell + 6), pivot=pivot,
                                                   shape=shape, size=size, seed=seed, noise=2)
                    rng = np.random.default_rng(seed + 99)
                    hot = rng.random(frames.shape[:3]) < .003
                    frames[hot] = rng.integers(0, 256, (int(hot.sum()), 3), dtype=np.uint8)
                    with self.assertRaisesRegex(RuntimeError, "no observable temporal signal"):
                        self.fit(frames)

    def test_straight_temporal_edge_does_not_constrain_a_center(self):
        frames = np.full((8, 128, 128, 3), 30, np.uint8)
        for index in range(8):
            frames[index, :, 58 + index:] = 220
        with self.assertRaisesRegex(RuntimeError, "cold-core|cold-core arc"):
            self.fit(frames)

    def test_translation_without_rotation_does_not_invent_a_pivot(self):
        frames = np.full((12, 128, 128, 3), 30, np.uint8)
        for index, frame in enumerate(frames):
            cv2.circle(frame, (58 + index, 62), 4, (25, 30, 235), -1)
        with self.assertRaisesRegex(RuntimeError, "no supported cold-core circle"):
            self.fit(frames)


if __name__ == "__main__":
    unittest.main()
