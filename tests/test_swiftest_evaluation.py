"""Pixels-only decisions, cache invalidation and frame-driven wait behavior."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import cv2
import numpy as np

from swiftest import Session, Template, Text, Box, AmbiguousTargetError
from swiftest.evaluation import VisualCache
from swiftest.session import Observation


class Frames:
    repeat = False
    def __init__(self, image):
        self.image, self.reads = image, 0
    def read(self):
        self.reads += 1
        return self.image
    def close(self):
        pass


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.image = np.zeros((120, 160, 3), np.uint8)
        self.tile = np.random.default_rng(341).integers(20, 230, (12, 16, 3), dtype=np.uint8)
        self.image[30:42, 40:56] = self.tile
        (self.base / 'button.png').write_bytes(cv2.imencode('.png', self.tile)[1].tobytes())
        self.frames = Frames(self.image)
        self.session = Session(device=SimpleNamespace(close=lambda: None),
                               capture=self.frames, base=self.base, timeout=0)
        self.addCleanup(self.session.close)

    def test_exact_pixels_reuse_even_without_backend_repeat_flag(self):
        target = Template('button.png')
        with patch('swiftest.targets.cv2.matchTemplate', wraps=cv2.matchTemplate) as matcher:
            a = self.session.wait(target)
            b = self.session.wait(target)
            self.assertEqual(a, b)
            self.assertEqual(matcher.call_count, 1)
        self.assertEqual(self.frames.reads, 2)  # Recognition reuse never replaces acquisition.

    def test_region_reuses_while_outside_animates_but_invalidates_inside(self):
        target = Template('button.png', region=(0, 0, .5, .5))
        with patch('swiftest.targets.cv2.matchTemplate', wraps=cv2.matchTemplate) as matcher:
            self.session.wait(target)
            self.image[100, 100] = 255
            self.session.wait(target)
            self.assertEqual(matcher.call_count, 1)
            self.image[30:42, 40:56] = 0
            self.assertFalse(self.session.exists(target))
            self.assertEqual(matcher.call_count, 2)

    def test_cache_does_not_hide_movement_duplicates_or_setting_changes(self):
        target = Template('button.png')
        self.session.wait(target)
        self.image[60:72, 90:106] = self.tile
        with self.assertRaises(AmbiguousTargetError):
            self.session.wait(target)
        self.image[30:42, 40:56] = 0
        self.assertEqual(self.session.wait(target), (98, 66))
        target.region = (0, 0, .5, .5)
        self.assertFalse(self.session.exists(target))

    def test_template_file_replacement_invalidates_results(self):
        target = Template('button.png')
        self.session.wait(target)
        other = np.random.default_rng(99).integers(20, 230, (14, 18, 3), dtype=np.uint8)
        (self.base / 'button.png').write_bytes(cv2.imencode('.png', other)[1].tobytes())
        self.assertFalse(self.session.exists(target))

    def test_cache_result_lists_are_not_shared_and_memory_is_bounded(self):
        cache = VisualCache(max_bytes=20, max_entries=2)
        pixels = np.zeros((2, 2), np.uint8)
        first = cache.evaluate('a', pixels, lambda: [Box(1, 2, 3, 4)])
        first.clear()
        self.assertEqual(len(cache.evaluate('a', pixels, lambda: self.fail('cache miss'))), 1)
        for i in range(10):
            cache.evaluate(i, pixels, lambda: [])
        self.assertLessEqual(len(cache.entries), 2)
        self.assertLessEqual(cache.bytes, 20)
        with self.assertRaises(RuntimeError):
            cache.evaluate('failure', pixels, lambda: (_ for _ in ()).throw(RuntimeError('bad')))
        self.assertEqual(cache.evaluate('failure', pixels, lambda: [1]), [1])

    def test_regional_ocr_crops_translates_and_shares_different_text_conditions(self):
        shapes = []
        def recognize(image):
            shapes.append(image.shape)
            return [('Ready', Box(2, 3, 20, 10, .95)), ('Go', Box(25, 3, 10, 10, .98))]
        self.session.ocr = recognize
        region = (.25, .25, .75, .75)
        a = Text('Ready', region=region, padding=0)
        b = Text('Go', region=region, padding=0)
        obs = Observation(self.session)
        self.assertEqual(a.matches(obs), [Box(42, 33, 20, 10, .95)])
        self.assertEqual(b.matches(obs), [Box(65, 33, 10, 10, .98)])
        self.assertEqual(shapes, [(60, 80, 3)])
        self.image[110, 150] = 255
        a.matches(Observation(self.session))
        self.assertEqual(len(shapes), 1)
        self.image[35, 45] = 255
        a.matches(Observation(self.session))
        self.assertEqual(len(shapes), 2)

    def test_padding_keeps_context_but_rejects_text_outside_requested_region(self):
        shapes = []
        def recognize(image):
            shapes.append(image.shape)
            return [('Ready', Box(0, 0, 5, 5)), ('Ready', Box(8, 8, 20, 10))]
        self.session.ocr = recognize
        target = Text('Ready', region=(.25, .25, .75, .75))
        self.assertEqual(target.matches(Observation(self.session)), [Box(40, 30, 20, 10)])
        self.assertEqual(shapes, [(76, 96, 3)])
        with self.assertRaises(ValueError):
            Text('Ready', padding=-1)

    def test_wait_any_shares_capture_and_reports_named_condition(self):
        self.session.ocr = lambda image: [('Ready', Box(10, 10, 20, 10))]
        result = self.session.wait_any({'error': Text('Error'), 'ready': Text('Ready')})
        self.assertEqual(result, {'condition': 'ready', 'point': (20, 15)})
        self.assertEqual(self.frames.reads, 1)
        self.assertEqual(self.session.stats['ocr_calls'], 1)

    def test_capture_failure_never_returns_a_cached_match(self):
        target = Template('button.png')
        self.session.wait(target)
        with patch.object(self.frames, 'read', side_effect=RuntimeError('disconnected')):
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                self.session.wait(target)

    def test_ordinary_polling_has_no_growing_backoff(self):
        clock = [0.0]
        class Capture:
            repeat = False
            def read(self):
                return np.full((2, 2, 3), int(clock[0] >= 2.0), np.uint8)
        class Ready:
            def matches(self, obs):
                return [Box(0, 0, 1, 1)] if obs.image[0, 0, 0] else []
        self.session._backend = Capture()
        self.session._backend.close = lambda: None
        with patch('swiftest.session.time.monotonic', side_effect=lambda: clock[0]), \
             patch('swiftest.session.time.sleep', side_effect=lambda n: clock.__setitem__(0, clock[0] + n)):
            self.session.wait(Ready(), timeout=4)
        self.assertGreaterEqual(clock[0], 2)
        self.assertLessEqual(clock[0], 2.100001)

    def test_frame_driven_backend_supplies_pacing_without_extra_sleep(self):
        clock = [0.0]
        class Stream:
            frame_driven = True
            repeat = False
            def read(self):
                clock[0] += .05  # Represents waiting for a producer notification.
                return np.full((2, 2, 3), int(clock[0] >= .2), np.uint8)
            def close(self):
                pass
        class Ready:
            def matches(self, obs):
                return [Box(0, 0, 1, 1)] if obs.image[0, 0, 0] else []
        self.session._backend = Stream()
        with patch('swiftest.session.time.monotonic', side_effect=lambda: clock[0]), \
             patch('swiftest.session.time.sleep', side_effect=AssertionError('extra polling sleep')):
            self.session.wait(Ready(), timeout=1)
        self.assertAlmostEqual(clock[0], .2)


if __name__ == '__main__':
    unittest.main()
