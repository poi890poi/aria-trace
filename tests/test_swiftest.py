"""Public behavior contracts for SwifTest, independent of an installed app."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from swiftest import (
    Absent,
    AmbiguousTargetError,
    Dialog,
    Node,
    Position,
    Session,
    TargetNotFoundError,
    Template,
)
from swiftest import compat
from swiftest.convert import convert
from swiftest.intent import execute, load_catalog
from swiftest.cli import run_script


class Frames:
    mode = "replay"
    repeat = False
    bytes = 0

    def __init__(self, image):
        self.image = image

    def read(self):
        return self.image

    def close(self):
        pass


class Device:
    serial = "fixture"
    package = None

    def __init__(self):
        self.taps = []
        self.page = "Open"
        self.switch_on = 1

    def hierarchy(self):
        return ET.fromstring(
            '<hierarchy><node resource-id="demo:id/action" text="'
            + self.page
            + '" bounds="[20,10][60,40]" enabled="true"/></hierarchy>'
        )

    def display_size(self):
        return 200, 120

    def tap(self, x, y, duration=0):
        self.taps.append((x, y))
        if len(self.taps) == self.switch_on:
            self.page = "Done"

    def launch(self, package, activity=None):
        self.package = package

    def close(self):
        pass


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.device = Device()
        self.screen = np.zeros((120, 200, 3), np.uint8)
        self.session = Session(
            device=self.device,
            capture=Frames(self.screen),
            base=self.base,
            timeout=0.01,
        )
        self.addCleanup(self.session.close)

    def asset(self, name, image):
        cv2.imwrite(str(self.base / name), image)
        return name

    def test_named_catalog_and_busy_overlay_defer_the_tap(self):
        catalog = self.base / "ui.json"
        catalog.write_text(
            '{"elements":{"go":{"node":{"text":"Open"}},"progress":{"node":{"text":"Loading"}}},"busy":[{"while":"progress","allowance":0.5}]}'
        )
        load_catalog(self.session, catalog)
        original = self.device.hierarchy
        calls = []

        def hierarchy():
            tree = original()
            calls.append(tree)
            if len(calls) == 1:
                tree.append(
                    ET.Element("node", {"text": "Loading", "bounds": "[0,0][20,20]"})
                )
            return tree

        self.device.hierarchy = hierarchy
        execute(self.session, {"steps": [{"touch": "go"}]})
        self.assertEqual(self.device.taps, [(40, 25)])
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.session.stats["captures"], 0)

    def test_failed_case_does_not_hide_later_case(self):
        path = self.base / "test_cases.py"
        path.write_text(
            "def test_one():\n    raise AssertionError('first')\ndef test_two(session):\n    assert session.device.serial == 'fixture'\n"
        )
        with self.session:
            cases = run_script(path, self.session)
        self.assertEqual([c["status"] for c in cases], ["failed", "passed"])

    def test_native_intent_without_pixels_or_app_profile(self):
        execute(
            self.session,
            {
                "app": "example.reader",
                "steps": [
                    {
                        "touch": {"node": {"text": "Open"}},
                        "expect": {"node": {"text": "Done"}},
                    },
                    {"assert_absent": {"node": {"text": "Open"}}},
                ],
            },
        )
        self.assertEqual(self.device.taps, [(40, 25)])
        self.assertEqual(self.session.report()["captures"], 0)

    def test_coordinates_and_offset_do_not_start_image_backend(self):
        self.session.register("missing", Absent(Node(text="Missing")))
        with self.assertRaises(ValueError):
            self.session.touch("missing")
        self.assertEqual(self.device.taps, [])
        self.assertEqual(self.session.touch(Position(1, 0), offset=(-10, 0)), (189, 0))
        self.assertEqual(self.session.stats["captures"], 0)
        with self.assertRaises(ValueError):
            self.session.touch(Position(1, 0), offset=(1, 0))

    def test_delivered_touch_is_not_implicitly_repeated(self):
        self.device.switch_on = 2
        self.assertFalse(
            self.session.touch(
                Node(text="Open"), expect=Node(text="Done"), required=False
            )
        )
        self.assertEqual(self.device.taps, [(40, 25)])
        with self.assertRaises(ValueError):
            self.session.touch(Node(text="Open"), expect=Node(text="Done"), retries=1)
        self.device.taps.clear()
        self.assertEqual(
            self.session.touch(
                Node(text="Open"), expect=Node(text="Done"), retries=1, retry_safe=True
            ),
            (40, 25),
        )
        self.assertEqual(len(self.device.taps), 2)

    def test_mask_scale_position_size_and_region_are_hard_constraints(self):
        rng = np.random.default_rng(42)
        original = rng.integers(30, 220, (12, 14, 3), dtype=np.uint8)
        mask = np.full((12, 14), 255, np.uint8)
        mask[:, 8:] = 0
        file = self.asset("button.png", original)
        maskfile = self.asset("keep.png", mask)
        changed = original.copy()
        changed[:, 8:] = 250
        self.screen[34:58, 92:120] = cv2.resize(changed, (28, 24))
        self.screen[75:87, 12:26] = original
        options = dict(
            mask=maskfile,
            scale_range=(1, 2, 0.5),
            expected=(106 / 199, 46 / 119),
            tolerance=(0.015, 0.015),
            size=(27, 23, 29, 25),
            threshold=0.9,
        )
        t = Template(file, region=(0.35, 0.15, 0.85, 0.65), **options)
        self.assertEqual(self.session.wait(t), (106, 46))
        for changes in [
            dict(expected=(0.15, 0.7)),
            dict(size=(5, 5, 15, 15)),
            dict(region=(0, 0, 0.30, 0.60)),
        ]:
            settings = dict(options, threshold="relaxed", **{})
            settings.update(changes)
            self.assertFalse(self.session.exists(Template(file, **settings)))

    def test_dialog_restricts_child_and_reports_ambiguity(self):
        rng = np.random.default_rng(31)
        button = rng.integers(25, 240, (10, 8, 3), dtype=np.uint8)
        panel = rng.integers(25, 240, (32, 50, 3), dtype=np.uint8)
        panel[8:18, 28:36] = button
        self.screen[45:77, 100:150] = panel
        self.screen[12:22, 18:26] = button
        button = Template(self.asset("x.png", button))
        panel = Template(self.asset("dialog.png", panel))
        with self.assertRaises(AmbiguousTargetError):
            self.session.wait(button, timeout=0)
        count = self.session.stats["captures"]
        self.assertEqual(self.session.wait(Dialog(panel, button)), (132, 58))
        self.assertEqual(self.session.stats["captures"] - count, 1)
        self.assertEqual(self.session.wait(Dialog(panel, Position(1, 1))), (149, 76))

    def test_converter_preserves_assets_comments_and_does_not_execute(self):
        source = self.base / "Old.air"
        source.mkdir()
        (source / "Old.py").write_text(
            "# preserve\nfrom airtest.core.api import *\nraise RuntimeError('must not execute during migration')\n",
            encoding="utf8",
        )
        (source / "icon.png").write_bytes(b"test-image")
        dest = self.base / "Converted"
        report = convert(source, dest)
        self.assertFalse(report["issues"])
        self.assertIn(
            "# preserve\nfrom swiftest.compat import *",
            (dest / "Old.py").read_text(),
        )
        self.assertEqual((dest / "icon.png").read_bytes(), b"test-image")
        with self.assertRaises(ValueError):
            convert(source, dest)

    def test_converter_flags_unknown_and_unimplemented_keywords(self):
        file = self.base / "input.py"
        file.write_text(
            'from airtest.core.api import *\nunknown_action()\nwait(Template("a.png"), intervalfunc=callback)\n'
        )
        report = convert(file, self.base / "output")
        self.assertEqual({p["line"] for p in report["issues"]}, {2, 3})

    def test_airtest_positions_and_exceptions(self):
        rng = np.random.default_rng(12)
        img = rng.integers(30, 240, (12, 14, 3), dtype=np.uint8)
        self.screen[30:42, 55:69] = img
        template = compat.Template(self.asset("air.png", img), target_pos=5)
        with self.session:
            self.assertEqual(compat.exists(template), (62, 36))
            self.assertEqual(compat.touch(template), (62, 36))
            self.screen[:] = 0
            self.assertFalse(compat.exists(template))
            with self.assertRaises(TargetNotFoundError):
                compat.wait(template, timeout=0)
