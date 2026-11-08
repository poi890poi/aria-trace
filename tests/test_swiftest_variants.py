"""Observable variant and interruption contracts, including negative cases."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from swiftest import (Absent, AmbiguousTargetError, Box, Dialog, Node,
                           PopupHandlingError, Position, Session, Template, Text,
                           TextPattern, Variants)
from swiftest.intent import execute, load_catalog, target


class FakeDevice:
    serial = "fixture"
    package = None

    def __init__(self, title="notice", close="Close"):
        self.title, self.close_text = title, close
        self.popup = True
        self.stuck = False
        self.done = False
        self.taps = []
        self.trees = 0
        self.duplicate = False
        self.outside_close = False

    def hierarchy(self):
        self.trees += 1
        root = ET.Element("hierarchy")
        bounds = "[4,4][40,24]" if self.popup else "[120,80][180,110]"
        ET.SubElement(root, "node", {"resource-id": "app:go", "text": "Done" if self.done else "Go",
                                    "bounds": bounds, "enabled": "true"})
        if self.popup:
            parent = ET.SubElement(root, "node", {"resource-id": self.title,
                                                  "bounds": "[20,20][180,100]"})
            ET.SubElement(parent, "node", {"text": self.close_text, "bounds": "[140,40][170,60]"})
            if self.duplicate:
                ET.SubElement(parent, "node", {"text": self.close_text, "bounds": "[50,40][80,60]"})
        if self.outside_close:
            ET.SubElement(root, "node", {"text": self.close_text, "bounds": "[1,1][18,18]"})
        return root

    def display_size(self):
        return 200, 120

    def tap(self, x, y, duration=0):
        self.taps.append((x, y))
        if (x, y) == (155, 50):
            if not self.stuck:
                self.popup = False
        elif (x, y) == (150, 95):
            self.done = True
        else:
            raise AssertionError("Incorrect or stale touch: " + str((x, y)))

    def close(self):
        pass


class VariantTests(unittest.TestCase):
    def session(self, device=None, **kwargs):
        session = Session(device=device or FakeDevice(), timeout=.5, **kwargs)
        self.addCleanup(session.close)
        return session

    def test_aliases_unicode_and_exact_native_compatibility(self):
        pattern = TextPattern(["Continue", "繼續", "继续"])
        for value in (" ＣＯＮＴＩＮＵＥ ", "繼 續", "继续"):
            self.assertTrue(pattern.matches(value))
        for value in ("Do not continue", "Cancel", ""):
            self.assertFalse(pattern.matches(value))
        d = FakeDevice(close="ＣＬＯＳＥ")
        s = self.session(d)
        self.assertFalse(s.exists(Node(text="Close")))  # Old native equality stays exact.
        self.assertTrue(s.exists(Node(text=TextPattern("Close"))))
        self.assertEqual(s.stats["captures"], 0)

    def test_regex_is_whole_label_and_explicit(self):
        pattern = TextPattern(regex=r"download\d+%")
        self.assertTrue(pattern.matches("Download 42%"))
        self.assertFalse(pattern.matches("Do not download 42%"))
        self.assertFalse(pattern.matches("Download failed"))

    def test_fuzzy_missing_glyph_is_bounded_and_scoped(self):
        observation = SimpleNamespace(words=[("輕按幕進入", Box(110, 95, 40, 10, .95)),
                                             ("輕按螢幕進入", Box(10, 10, 40, 10, .95))],
                                      screen=Box(0, 0, 200, 120))
        selector = Text("輕按螢幕進入", max_edits=1, region=(.4, .7, 1, 1), min_confidence=.8)
        self.assertEqual(selector.matches(observation), [observation.words[0][1]])
        self.assertFalse(TextPattern("輕按螢幕進入", max_edits=1).matches("輕按入"))
        with self.assertRaises(ValueError):
            Text("輕按螢幕進入", max_edits=1).matches(observation)
        with self.assertRaises(ValueError):
            TextPattern("Pay", max_edits=1)
        with self.assertRaises(ValueError):
            Text("Continue", exact=False, max_edits=1)
        low = SimpleNamespace(words=[("輕按幕進入", Box(110, 95, 40, 10, .2))], screen=observation.screen)
        self.assertFalse(selector.matches(low))

    def test_variants_do_not_hide_two_different_controls(self):
        d = FakeDevice(close="Close")
        d.outside_close = True
        s = self.session(d)
        with self.assertRaises(AmbiguousTargetError):
            s.touch(Variants(Node(text="Close"), Node(resource_id="app:go")), timeout=0)
        self.assertEqual(d.taps, [])

    def test_duplicate_aliases_are_one_control_and_read_tree_once(self):
        d = FakeDevice()
        s = self.session(d)
        self.assertEqual(s.wait(Variants(Node(text="Go"), Node(resource_id="app:go"))), (22, 14))
        self.assertEqual(d.trees, 1)
        self.assertEqual(s.stats["captures"], 0)

    def test_conflicting_template_anchors_stay_ambiguous(self):
        class Fixed:
            def __init__(self, point):
                self.input = point
            def matches(self, observation, scope=None):
                return [Box(20, 20, 100, 40)]
            def point(self, box):
                return self.input
        self.assertEqual(len(Variants(Fixed((20, 20)), Fixed((100, 50))).matches(None)), 2)

    def test_theme_variants_share_frame_and_keep_region_and_mask_constraints(self):
        rng = np.random.default_rng(164)
        light = rng.integers(20, 235, (12, 18, 3), dtype=np.uint8)
        dark = 255-light
        mask = np.full((12, 18), 255, np.uint8)
        mask[:, :4] = 0
        class Frames:
            repeat = False
            mode = "fixture"
            bytes = 0
            def read(self):
                return screen
            def close(self):
                pass
        with TemporaryDirectory() as temp:
            base = Path(temp)
            for name, image in (("light.png", light), ("dark.png", dark), ("mask.png", mask)):
                cv2.imwrite(str(base/name), image)
            selector = Variants(*(Template(name, mask="mask.png", scales=(1, 2),
                                           region=(.4, .2, 1, .8)) for name in ("light.png", "dark.png")))
            for variant in (light, dark):
                screen = np.zeros((120, 200, 3), np.uint8)
                changed = variant.copy()
                changed[:, :4] = 253
                screen[40:64, 100:136] = cv2.resize(changed, (36, 24))
                screen[5:17, 5:23] = variant  # Identical-looking control outside allowed region.
                s = self.session(capture=Frames(), base=base)
                self.assertEqual(s.wait(selector), (118, 52))
                self.assertEqual(s.stats["captures"], 1)


class PopupTests(unittest.TestCase):
    def session(self, device, **opts):
        s = Session(device=device, timeout=.5, **opts)
        self.addCleanup(s.close)
        return s

    def test_catalog_covers_localized_popup_and_reacquires_moved_button(self):
        for title, close in (("notice", "Close"), ("notice.v2", "關閉"), ("notice.v2", "关闭")):
            with self.subTest(title=title, close=close), TemporaryDirectory() as temp:
                d = FakeDevice(title, close)
                d.outside_close = True
                s = self.session(d)
                path = Path(temp)/"ui.json"
                path.write_text(json.dumps({"elements": {
                    "go": {"node": {"resource_id": "app:go"}},
                    "done": {"node": {"text": "Done"}},
                    "notice": {"variants": [{"node": {"resource_id": "notice"}},
                                               {"node": {"resource_id": "notice.v2"}}]},
                    "close": {"node": {"text": {"values": ["Close", "關閉", "关闭"]}}}},
                    "popups": [{"name": "notice", "when": "notice", "dismiss": "close"}]}), encoding="utf-8")
                load_catalog(s, path)
                execute(s, {"steps": [{"touch": "go", "expect": "done"}]})
                self.assertEqual(d.taps, [(155, 50), (150, 95)])
                self.assertEqual(s.stats["captures"], 0)
                self.assertTrue(any(e.get("status") == "dismissed" for e in s.events))

    def test_stuck_popup_is_tapped_once_and_never_clicks_through(self):
        d = FakeDevice()
        d.stuck = True
        s = self.session(d)
        s.on_popup("notice", Node(resource_id="notice"), Node(text="Close"), timeout=.02)
        with self.assertRaises(PopupHandlingError):
            s.touch(Node(resource_id="app:go"))
        self.assertEqual(d.taps, [(155, 50)])

    def test_popup_after_main_touch_is_handled_before_postcondition(self):
        d = FakeDevice()
        d.popup = False
        original = d.tap

        def tap(x, y, duration=0):
            original(x, y, duration)
            if (x, y) == (150, 95):
                d.popup = True

        d.tap = tap
        s = self.session(d)
        s.on_popup("notice", Node(resource_id="notice"), Node(text="Close"))
        s.touch(Node(resource_id="app:go"), expect=Node(text="Done"))
        self.assertEqual(d.taps, [(150, 95), (155, 50)])
        self.assertFalse(d.popup)

    def test_button_outside_popup_cannot_be_used_for_dismissal(self):
        d = FakeDevice()
        original = d.hierarchy

        def hierarchy():
            tree = original()
            ET.SubElement(tree, "node", {"text": "Dismiss", "bounds": "[1,1][18,18]"})
            return tree

        d.hierarchy = hierarchy
        s = self.session(d)
        s.on_popup("notice", Node(resource_id="notice"), Node(text="Dismiss"), timeout=.01)
        with self.assertRaises(PopupHandlingError):
            s.touch(Node(resource_id="app:go"))
        self.assertEqual(d.taps, [])

    def test_zero_time_queries_do_not_dismiss_or_click_through(self):
        d = FakeDevice()
        s = self.session(d)
        s.on_popup("notice", Node(resource_id="notice"), Node(text="Close"))
        self.assertFalse(s.exists(Node(resource_id="app:go")))
        self.assertEqual(d.taps, [])

    def test_unknown_popup_is_not_dismissed(self):
        d = FakeDevice(title="unregistered")
        s = self.session(d)
        s.on_popup("known", Node(resource_id="notice"), Node(text="Close"))
        self.assertFalse(s.exists(Node(text="Missing"), timeout=.01))
        self.assertEqual(d.taps, [])

    def test_ambiguous_dismissal_sends_no_input(self):
        d = FakeDevice()
        d.duplicate = True
        s = self.session(d)
        s.on_popup("notice", Node(resource_id="notice"), Node(text="Close"))
        with self.assertRaises(AmbiguousTargetError):
            s.touch(Node(resource_id="app:go"))
        self.assertEqual(d.taps, [])

    def test_multiple_handlers_matching_block_input(self):
        d = FakeDevice()
        s = self.session(d)
        for name in ("a", "b"):
            s.on_popup(name, Node(resource_id="notice"), Node(text="Close"))
        with self.assertRaises(AmbiguousTargetError):
            s.touch(Node(resource_id="app:go"))
        self.assertEqual(d.taps, [])

    def test_recurring_popup_has_session_wide_limit(self):
        d = FakeDevice()
        s = self.session(d)
        s.on_popup("notice", Node(resource_id="notice"), Node(text="Close"))
        s.wait(Node(resource_id="app:go"))
        d.popup = True
        with self.assertRaises(PopupHandlingError):
            s.touch(Node(resource_id="app:go"))
        self.assertEqual(d.taps, [(155, 50)])

    def test_condition_cannot_be_popup_or_dismissal(self):
        s = self.session(FakeDevice())
        for guard, dismiss in ((Absent(Node(text="None")), Position(.5, .5)),
                               (Node(resource_id="notice"), Absent(Node(text="None")))):
            with self.assertRaises(ValueError):
                s.on_popup("bad", guard, dismiss)


if __name__ == "__main__":
    unittest.main()
