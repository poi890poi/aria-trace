import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imports_in(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_new_contract_package_has_no_legacy_or_platform_dependencies(self):
        forbidden = ("acquisition", "replay", "poc", "cv2", "numpy", "PySide6")
        violations = []
        for layer in ("domain", "ports"):
            for path in (ROOT / "aria_trace" / layer).rglob("*.py"):
                for imported in imports_in(path):
                    if imported.startswith(forbidden):
                        violations.append((str(path.relative_to(ROOT)), imported))
        self.assertEqual([], violations)

    def test_no_new_production_imports_from_poc_are_added(self):
        violations = set()
        for package in ("acquisition", "replay", "aria_trace"):
            for path in (ROOT / package).rglob("*.py"):
                for imported in imports_in(path):
                    if imported == "poc" or imported.startswith("poc."):
                        violations.add((path.relative_to(ROOT).as_posix(), imported))
        self.assertEqual(set(), violations)

    def test_poc_imports_are_compatibility_aliases_for_promoted_services(self):
        from aria_trace.services.tracking import PoseFusionGate as production_fusion
        from aria_trace.services.vision import KltAngularYawEstimator as production_yaw
        from poc.pose_fusion import PoseFusionGate as compatibility_fusion
        from poc.yaw_estimation import KltAngularYawEstimator as compatibility_yaw

        self.assertIs(production_fusion, compatibility_fusion)
        self.assertIs(production_yaw, compatibility_yaw)

    def test_workbench_server_legacy_exports_are_exact_compatibility_aliases(self):
        from acquisition import workbench as legacy
        from aria_trace.apps.workbench import server

        self.assertEqual(server.WORKBENCH_SERVICE, legacy.WORKBENCH_SERVICE)
        self.assertIs(server.WorkbenchHttpServer, legacy.WorkbenchHttpServer)
        self.assertIs(
            server.discover_workbench_instance,
            legacy.discover_workbench_instance,
        )
        self.assertIs(server.is_client_disconnect, legacy.is_client_disconnect)
        self.assertIs(server.occupied_port_message, legacy.occupied_port_message)

    def test_workbench_source_legacy_exports_are_exact_compatibility_aliases(self):
        from acquisition import workbench as legacy
        from aria_trace.apps.workbench import sources

        self.assertIs(sources.SourceFactory, legacy.SourceFactory)
        self.assertIs(sources.parse_adb_devices, legacy.parse_adb_devices)

    def test_workbench_application_legacy_export_is_exact_compatibility_alias(self):
        from acquisition import workbench as legacy
        from aria_trace.apps.workbench import application

        self.assertIs(application.AcquisitionWorkbench, legacy.AcquisitionWorkbench)
        self.assertIs(application.make_handler, legacy.make_handler)
        self.assertIs(application.main, legacy.main)

        from aria_trace.apps.workbench.api import make_handler as api_handler

        self.assertIs(api_handler, application.make_handler)

    def test_cursor_legacy_exports_are_exact_service_aliases(self):
        from acquisition.cursor_pose import CursorPoseEstimator as legacy_pose
        from acquisition.cursor_worker import CursorPoseProcessExecutor as legacy_worker
        from aria_trace.services.calibration.cursor import (
            CursorPoseEstimator,
            CursorPoseProcessExecutor,
        )

        self.assertIs(CursorPoseEstimator, legacy_pose)
        self.assertIs(CursorPoseProcessExecutor, legacy_worker)

    def test_minimap_legacy_exports_are_exact_service_aliases(self):
        from acquisition.minimap_calibration import (
            calibrate_minimap_boundary_frames as legacy_boundary,
        )
        from acquisition.minimap_transition import TransitionController as legacy_transition
        from aria_trace.services.calibration.minimap import (
            TransitionController,
            calibrate_minimap_boundary_frames,
        )

        self.assertIs(calibrate_minimap_boundary_frames, legacy_boundary)
        self.assertIs(TransitionController, legacy_transition)

    def test_mapping_legacy_exports_are_exact_service_aliases(self):
        from acquisition.map_layers import LayeredGlobalLocalizer as legacy_localizer
        from acquisition.map_stitching import stitch_map_session as legacy_stitch
        from aria_trace.services.mapping import LayeredGlobalLocalizer, stitch_map_session

        self.assertIs(LayeredGlobalLocalizer, legacy_localizer)
        self.assertIs(stitch_map_session, legacy_stitch)

    def test_tracking_legacy_exports_are_exact_layer_aliases(self):
        from acquisition.frame_pump import LatestFramePump as legacy_pump
        from acquisition.live_tracker import TwoRateRealtimeTracker as legacy_tracker
        from acquisition.route_tracker import RouteVisualTracker as legacy_route
        from aria_trace.services.capture import LatestFramePump
        from aria_trace.services.localization.route import RouteVisualTracker
        from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker

        self.assertIs(LatestFramePump, legacy_pump)
        self.assertIs(TwoRateRealtimeTracker, legacy_tracker)
        self.assertIs(RouteVisualTracker, legacy_route)

    def test_teleport_record_and_legacy_exports_have_domain_owners(self):
        from acquisition.models import TeleportBehaviorSample as legacy_record
        from acquisition.teleport_behavior import make_teleport_behavior_sample as legacy_make
        from aria_trace.domain import TeleportBehaviorSample
        from aria_trace.services.localization.teleport import make_teleport_behavior_sample

        self.assertIs(TeleportBehaviorSample, legacy_record)
        self.assertIs(make_teleport_behavior_sample, legacy_make)

    def test_scene_yaw_legacy_export_is_exact_service_alias(self):
        from acquisition.scene_yaw_calibration import calibrate_scene_yaw_session as legacy
        from aria_trace.services.calibration.scene_yaw import calibrate_scene_yaw_session

        self.assertIs(calibrate_scene_yaw_session, legacy)


if __name__ == "__main__":
    unittest.main()
