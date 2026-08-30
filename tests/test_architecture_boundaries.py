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
    def test_legacy_hardware_exports_are_exact_adapter_aliases(self):
        from acquisition.android_capture import ScrcpyCaptureHub as legacy_android
        from acquisition.hik_capture import CalibratedHikFrameSource as legacy_hik
        from acquisition.windows import WindowsWindowFrameSource as legacy_windows
        from aria_trace.adapters.android.capture import ScrcpyCaptureHub
        from aria_trace.adapters.hik.capture import CalibratedHikFrameSource
        from aria_trace.adapters.windows import WindowsWindowFrameSource

        self.assertIs(legacy_android, ScrcpyCaptureHub)
        self.assertIs(legacy_hik, CalibratedHikFrameSource)
        self.assertIs(legacy_windows, WindowsWindowFrameSource)

    def test_legacy_storage_and_recording_exports_are_exact_aliases(self):
        from acquisition.annotations import AnnotationStore as legacy_annotations
        from acquisition.recorder import AcquisitionRecorder as legacy_recorder
        from acquisition.session import SessionReader as legacy_session
        from acquisition.video import create_video_sink as legacy_video
        from aria_trace.adapters.filesystem.annotations import AnnotationStore
        from aria_trace.adapters.filesystem.session import SessionReader
        from aria_trace.adapters.filesystem.video import create_video_sink
        from aria_trace.workflows.recording import AcquisitionRecorder

        self.assertIs(legacy_annotations, AnnotationStore)
        self.assertIs(legacy_recorder, AcquisitionRecorder)
        self.assertIs(legacy_session, SessionReader)
        self.assertIs(legacy_video, create_video_sink)

    def test_legacy_profile_exports_are_exact_configuration_aliases(self):
        from acquisition.profile_manager import publish_rig_calibration as legacy_publish
        from acquisition.profile_registry import ProfileRegistry as legacy_registry
        from acquisition.profiles import ProfileCatalog as legacy_catalog
        from aria_trace.adapters.filesystem.profile_registry import ProfileRegistry
        from aria_trace.adapters.filesystem.profiles import ProfileCatalog
        from aria_trace.workflows.profile_management import publish_rig_calibration

        self.assertIs(legacy_registry, ProfileRegistry)
        self.assertIs(legacy_catalog, ProfileCatalog)
        self.assertIs(legacy_publish, publish_rig_calibration)

    def test_legacy_packet_exports_are_exact_domain_aliases(self):
        from acquisition.models import FramePacket as legacy_frame
        from acquisition.models import InputPacket as legacy_input
        from aria_trace.domain.packets import FramePacket, InputPacket

        self.assertIs(legacy_frame, FramePacket)
        self.assertIs(legacy_input, InputPacket)

    def test_legacy_evidence_and_app_exports_are_exact_aliases(self):
        from acquisition.features import OnlineSiftRecorder as legacy_features
        from acquisition.hud import _UrlStatusProvider as legacy_hud_provider
        from acquisition.media_trace import raster_record as legacy_media
        from acquisition.poc_evidence import build_poc_evidence_index as legacy_poc
        from acquisition.review import ReviewState as legacy_review
        from aria_trace.apps.review import ReviewState
        from aria_trace.apps.workbench.hud import _UrlStatusProvider
        from aria_trace.evidence.features import OnlineSiftRecorder
        from aria_trace.evidence.media_trace import raster_record
        from aria_trace.evidence.poc_catalog import build_poc_evidence_index

        self.assertIs(legacy_features, OnlineSiftRecorder)
        self.assertIs(legacy_hud_provider, _UrlStatusProvider)
        self.assertIs(legacy_media, raster_record)
        self.assertIs(legacy_poc, build_poc_evidence_index)
        self.assertIs(legacy_review, ReviewState)

        from acquisition.capture_game_minimap_zigzag import main as legacy_capture
        from acquisition.record import main as legacy_record
        from aria_trace.apps.record import main as record_main
        from aria_trace.workflows.minimap_capture import main as capture_main

        self.assertIs(legacy_capture, capture_main)
        self.assertIs(legacy_record, record_main)

    def test_legacy_rig_exports_are_exact_service_aliases(self):
        from acquisition.rig_calibration import FrameSample as legacy_result
        from acquisition.rig_calibration.hik.driver import HikMvsCameraAdapter as legacy_hik
        from aria_trace.services.calibration.rig import FrameSample
        from aria_trace.services.calibration.rig.hik.driver import HikMvsCameraAdapter

        self.assertIs(legacy_result, FrameSample)
        self.assertIs(legacy_hik, HikMvsCameraAdapter)

        from acquisition.dual_source_spaces import write_dual_source_space_yaml as legacy_spaces
        from acquisition.game_cross_source_check import GameCrossSourceEvidenceRecorder as legacy_cross
        from acquisition.hik_bayer_color_match import optimize_mvs_bayer_conversion as legacy_color
        from aria_trace.services.calibration.rig.cross_source import GameCrossSourceEvidenceRecorder
        from aria_trace.services.calibration.rig.dual_source_spaces import write_dual_source_space_yaml
        from aria_trace.services.calibration.rig.hik.color_match import optimize_mvs_bayer_conversion

        self.assertIs(legacy_spaces, write_dual_source_space_yaml)
        self.assertIs(legacy_cross, GameCrossSourceEvidenceRecorder)
        self.assertIs(legacy_color, optimize_mvs_bayer_conversion)

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
