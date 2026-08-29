import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDER_HTML = ROOT / "acquisition" / "static" / "recorder.html"


class _IdLocationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.section_stack = []
        self.locations = {}
        self.attributes = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "section":
            self.section_stack.append(values.get("id"))
        element_id = values.get("id")
        if element_id:
            self.locations[element_id] = next(
                (item for item in reversed(self.section_stack) if item), None
            )
            self.attributes[element_id] = values

    def handle_endtag(self, tag):
        if tag == "section":
            self.section_stack.pop()


class WorkbenchRouteTracerUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RECORDER_HTML.read_text(encoding="utf-8")
        cls.parser = _IdLocationParser()
        cls.parser.feed(cls.source)

    def test_route_compilation_and_tracking_share_a_dedicated_tab(self):
        self.assertEqual(
            self.parser.attributes["routeTab"].get("aria-controls"), "routeTask"
        )
        for element_id in (
            "routeSource",
            "routeAtlas",
            "runRouteCompile",
            "routePackage",
            "routeTrackingProfile",
            "startRouteTracker",
            "stopRouteTracker",
            "routeTrackerStatus",
            "routeTrackerMetrics",
            "routeTrackerOverlay",
        ):
            self.assertEqual(self.parser.locations[element_id], "routeTask")

    def test_live_tracker_remains_a_separate_free_roam_workflow(self):
        self.assertNotIn('id="trackingMode"', self.source)
        self.assertEqual(self.parser.locations["trackingProfile"], "trackerTask")
        self.assertEqual(self.parser.locations["mapAtlas"], "trackerTask")
        self.assertEqual(self.parser.locations["startTracker"], "trackerTask")
        self.assertIn(
            "ui.startTracker.onclick=()=>startLiveTracker('free-roam')", self.source
        )
        self.assertIn(
            "ui.startRouteTracker.onclick=()=>startLiveTracker('route-assisted')",
            self.source,
        )

    def test_route_package_selector_matches_the_selected_inputs(self):
        for constraint in (
            "item.source_session_key===ui.routeSource.value",
            "item.atlas_id===ui.routeAtlas.value",
            "item.minimap_calibration_id===(calibration&&calibration.calibration_id)",
        ):
            self.assertIn(constraint, self.source)

    def test_javascript_ui_references_have_unique_html_elements(self):
        ids = re.findall(r'id="([^"]+)"', self.source)
        self.assertEqual(len(ids), len(set(ids)))
        referenced = set(re.findall(r"\bui\.([A-Za-z][A-Za-z0-9_]*)", self.source))
        self.assertEqual(sorted(referenced - set(ids)), [])


if __name__ == "__main__":
    unittest.main()
