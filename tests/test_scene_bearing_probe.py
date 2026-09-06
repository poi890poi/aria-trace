import unittest

import cv2
import numpy as np

from benchmarks.localization.scene_bearing_probe import assay, BearingTracker, scene_mask


class SceneBearingTests(unittest.TestCase):
    def test_every_tracking_path_rejects_hud_and_avatar_locations(self):
        rng = np.random.default_rng(418)
        gray = rng.integers(0, 256, (360, 640), dtype=np.uint8)
        points = np.float32([[620, 120], [320, 170], [400, 120]])
        allowed = scene_mask(gray)
        for method, reacquire in [('lk', False), ('template', False), ('lk', True)]:
            tracker = BearingTracker(gray, points, method, reacquire, mask_scene=True)
            result = tracker.update(gray)
            self.assertNotIn(0, result['active_ids'])
            self.assertNotIn(1, result['active_ids'])
            self.assertIn(2, result['active_ids'])
            for point in result['points']:
                x, y = np.rint(point).astype(int)
                self.assertTrue(np.all(allowed[y-8:y+9, x-8:x+9]))

    def test_known_motion_loss_and_same_feature_recovery(self):
        rng = np.random.default_rng(912)
        source = cv2.GaussianBlur(rng.integers(0, 256, (180, 320), dtype=np.uint8), (5, 5), .8)
        times = np.arange(31) / 10
        shifts = np.column_stack([3 * times, times])
        frames = [cv2.warpAffine(source, np.float32([[1, 0, x], [0, 1, y]]), (320, 180)) for x, y in shifts]
        for i, t in enumerate(times):
            if 1 <= t <= 1.4:
                frames[i][:] = 0
        for method, reacquire in [('lk', False), ('template', False), ('lk', True)]:
            with self.subTest(method=method, reacquire=reacquire):
                result = assay(frames, times, method, reacquire, shifts, (1., 1.4))
                self.assertGreaterEqual(result['initial_features'], 3)
                self.assertEqual(result['blackout_false_fresh_features'], 0)
                self.assertLess(result['synthetic_error_px']['p95'], .5)
                if reacquire:
                    self.assertGreaterEqual(result['final_features'], 3)
                    self.assertLessEqual(result['recovery_after_blackout_s'], .4)
                else:
                    self.assertEqual(result['final_features'], 0)
                    self.assertIsNone(result['recovery_after_blackout_s'])


if __name__ == '__main__':
    unittest.main()
