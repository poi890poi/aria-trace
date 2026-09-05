import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDER_HTML = ROOT / "aria_trace" / "apps" / "workbench" / "static" / "recorder.html"


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

    def test_teleport_analysis_requires_ready_map_localization(self):
        self.assertIn(
            "localizationReady=localization.status==='ready'", self.source
        )
        self.assertIn("stitch&&!localizationReady", self.source)
        self.assertIn(
            "Rebuild it with compatible mini-map evidence before teleport analysis.",
            self.source,
        )

    def test_teleport_review_exposes_arrival_localization_source(self):
        self.assertIn("arrival_localization_source_counts", self.source)
        self.assertIn("geometric fallback", self.source)

    def test_map_atlas_uses_one_master_stitch_and_optional_transition(self):
        self.assertEqual(self.parser.locations["worldLayer"], "mapTask")
        self.assertEqual(self.parser.locations["worldLayerSummary"], "mapTask")
        self.assertNotIn("townLayer", self.parser.locations)
        self.assertNotIn("map_stitch_id:ui.worldLayer.value", self.source)
        self.assertIn("recommended_for_atlas", self.source)
        self.assertIn("history_only", self.source)
        self.assertIn("None · build a single-scale atlas", self.source)
        self.assertIn("automatically uses the current stitch", self.source)

    def test_map_review_reports_rejected_torn_captures(self):
        self.assertIn("Rejected torn captures", self.source)
        self.assertIn("spatially_incoherent_frame_count", self.source)
        self.assertIn("spatially_incoherent_registrations", self.source)

    def test_capture_inventory_uses_compact_progressive_disclosure(self):
        self.assertEqual(self.parser.locations["sourceInventoryPanel"], None)
        self.assertIn("<details id=\"sourceInventoryPanel\">", self.source)
        self.assertNotIn("<details id=\"sourceInventoryPanel\" open", self.source)
        self.assertIn("ui.sourceInventorySummary.textContent='Sources · '", self.source)
        self.assertIn("rig calibration'+(rigs.length===1?'':'s')+' ready", self.source)

    def test_session_collection_is_filterable_and_paged(self):
        for element_id in (
            "sessionSearch",
            "sessionLabelFilter",
            "sessionPageSize",
            "sessionPrevious",
            "sessionNext",
            "sessionPageStatus",
        ):
            self.assertIn(element_id, self.parser.locations)
        self.assertIn("filtered.slice(start,end)", self.source)
        self.assertIn("Math.ceil(filtered.length/pageSize)", self.source)
        self.assertIn("sessionPage=0;renderSessions()", self.source)
        self.assertIn("No matching sessions", self.source)


if __name__ == "__main__":
    unittest.main()
