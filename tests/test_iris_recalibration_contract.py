"""Recalibration regressions through the real registry, CLI and camera facade.

Only optical measurement and physical acquisition are substituted. All profile
selection, composition, activation, ROI planning and geometry conversion are real.
"""
import contextlib
import io
import importlib.util
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from rig_runtime.apps import hik_rig_calibration as cli, hik_stream as gui
from rig_runtime.adapters.filesystem.profile_registry import (
    AdapterRequest, ProfileContext, ProfileRegistry, ProfileResolutionError,
    context_from_rig_calibration,
)
from rig_runtime.adapters.hik.compat import HikCamera
from rig_runtime.adapters.hik.game_camera import ProfiledHikGameCamera
from rig_runtime.domain.spatial import bind_geometry, oriented_circle, raster_space
from rig_runtime.workflows.profile_management import publish_rig_calibration
from rig_runtime.workflows.adapter_export import export_resolved_adapter
from tests.test_profile_manager import write_rig
from tests.test_hik_game_camera import FakeAdapter


class RecalibrationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = ProfileRegistry(self.root / "profiles")
        self.initial_path = self.rig_file("initial", 0)
        initial = publish_rig_calibration(self.initial_path, registry=self.registry)
        rig_context = context_from_rig_calibration(json.loads(self.initial_path.read_text()))
        self.context = ProfileContext(
            game_id="game-1", camera_id="CAM-1", phone_id="PHONE-1",
            panel_display=rig_context.panel_display,
            game_display={"logical_frame_px": [100, 200], "rotation_quarter_turns": 0},
        )
        space = raster_space("android_phone_natural_display_pixels", [100, 200])
        point = bind_geometry({"x": 75, "y": 30}, "point", space)
        self.payload = {
            "canonical_phone_crop_xywh": [65, 20, 20, 20],
            "outer_boundary": oriented_circle(bind_geometry(
                {"center_x": 75, "center_y": 30, "radius": 8}, "circle", space)),
            "rotation_center": point,
            "cursor_geometry": {"rotation_center": point,
                                "rotating_cursor_envelope_diameter_px": 5},
            "game_orientation": "portrait",
        }
        self.phone = self.registry.publish("phone_game", self.context, self.payload,
                                           review_state="accepted", activate=True)
        self.initial = publish_rig_calibration(self.initial_path, registry=self.registry)

    def rig_file(self, name, offset):
        path = write_rig(self.root / name, camera_x_offset=offset)
        data = json.loads(path.read_text())
        # Independent synthetic ground truth: screen x maps to sensor x+offset.
        data["normalization"]["full_sensor_camera_to_output_3x3"] = [
            [1, 0, -offset], [0, 1, 0], [0, 0, 1]]
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def camera(self, mode="dual", **options):
        return HikCamera(config={"profile_root": str(self.registry.root),
                                 "game_id": "game-1", "phone_id": "PHONE-1",
                                 "camera_id": "CAM-1", "mode": mode,
                                 "color_order": "BGR", **options})

    def assert_streams_work(self, offset):
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter",
                   side_effect=lambda **kwargs: FakeAdapter()):
            for mode in ("full", "minimap", "dual"):
                with self.camera(mode) as camera:
                    frames = camera.get_frames()
                    for stream, frame in frames.items():
                        boundary = camera.get_minimap_geometry(stream)
                        cursor = camera.get_cursor_geometry(stream)
                        self.assertTrue(boundary["available_in_stream_space"])
                        self.assertTrue(cursor["available_in_stream_space"])
                        self.assertEqual(list(frame.shape[:2][::-1]), boundary["image_space"]["stored_size_px"])
                        self.assertEqual([0.0, -1.0], boundary["orientation_frame"]["up_unit_xy"])
                        x, y = map(round, boundary["center_xy_px"])
                        np.testing.assert_array_equal(frame[y, x], [75 + offset, 30, 7])

    def headless(self, path, *, reuse=False, adapter=None):
        session = Mock()
        session.run.return_value = path.parent
        session.stage_timings = []
        output = self.root / "reuse" if reuse else path.parent
        arguments = ["--camera-id", "CAM-1", "--phone-serial", "PHONE-1",
                     "--profile-root", str(self.registry.root), "--output", str(output),
                     "--headless", "--save", "--target-presenter", "owned_http"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(cli, "HikMvsCameraAdapter", return_value=adapter or FakeAdapter()))
            stack.enter_context(patch.object(cli, "AdbPhoneSession"))
            stack.enter_context(patch.object(cli, "resolve_adb_executable", return_value=Path("adb")))
            stack.enter_context(patch.object(cli, "HikRigCalibrationSession", return_value=session))
            if reuse:
                arguments.append("--reuse-if-unchanged")
                stack.enter_context(patch(
                    "rig_runtime.workflows.rig_reuse_precheck.run_active_reuse_precheck",
                    return_value={"reusable": True, "camera_adapter_is_calibrated": True,
                                  "calibration": str(path)}))
            return cli.main(arguments)

    def test_headless_success_retains_source_and_correct_pixels_in_every_mode(self):
        self.assert_streams_work(0)
        path = self.rig_file("moved", 1)
        self.assertEqual(0, self.headless(path))
        self.assertEqual(self.phone["revision_id"], self.registry.resolve(
            "rig_game", self.context)["dependencies"]["phone_game"])
        self.assert_streams_work(1)
        report = json.loads((path.parent / "game_readiness.json").read_text())
        self.assertEqual("camera_frames", report["validation"])
        self.assertEqual(3, len(report["games"]))

    def test_newer_other_phone_cannot_replace_established_geometry(self):
        other = ProfileContext(game_id="game-1", phone_id="PHONE-2",
                               panel_display=self.context.panel_display,
                               game_display=self.context.game_display)
        self.registry.publish("phone_game", other,
                              {"canonical_phone_crop_xywh": [65, 20, 20, 20]},
                              review_state="accepted", activate=True)
        publish_rig_calibration(self.rig_file("new-rig", 1), registry=self.registry)
        self.assertEqual(self.phone["revision_id"], self.registry.resolve(
            "rig_game", self.context)["dependencies"]["phone_game"])
        self.assert_streams_work(1)

    def test_incomplete_coverage_cannot_publish_success_or_replace_active_graph(self):
        before = self.registry.active_revision_ids()
        path = self.rig_file("outside", 30)
        with self.assertRaisesRegex(ProfileResolutionError, "not ready.*game-1"):
            self.headless(path)
        self.assertEqual(before, self.registry.active_revision_ids())
        self.assertFalse((path.parent / "game_readiness.json").exists())
        self.assertFalse((path.parent / "hikcam_adapter.py").exists())

    def test_invalid_optional_orientation_hints_preserve_working_axes_and_pixels(self):
        manifest = Path(self.phone["revision_directory"]) / "profile.json"
        original = json.loads(manifest.read_text())
        for field in ("game_orientation", "game_surface_quarter_turns_clockwise_from_phone_natural"):
            with self.subTest(field=field):
                document = json.loads(json.dumps(original))
                document["payload"][field] = "unknown"
                manifest.write_text(json.dumps(document))
                self.assertEqual(0, self.headless(self.rig_file(field, 1)))
                self.assert_streams_work(1)
                orientation = self.registry.resolve("rig_game_orientation", self.context)
                self.assertTrue(orientation["payload"]["orientation_consistency"]["fallback_reasons"])
                report = json.loads((self.root / field / "game_readiness.json").read_text())
                self.assertTrue(report["notices"])

    def test_invalid_optional_game_model_does_not_block_rebuilding(self):
        for field in ("game_orientation", "cursor_follows"):
            with self.subTest(field=field):
                self.registry.publish("game_model", ProfileContext(game_id="game-1"),
                                      {field: "unknown"}, activate=True)
                self.assertEqual(0, self.headless(self.rig_file("invalid-model-" + field, 1)))
                self.assert_streams_work(1)

    def test_unreadable_optional_game_model_does_not_block_rebuilding_or_reopen(self):
        model = self.registry.publish("game_model", ProfileContext(game_id="game-1"), {}, activate=True)
        (Path(model["revision_directory"]) / "profile.json").unlink()
        self.assertEqual(0, self.headless(self.rig_file("missing-game-model", 1)))
        self.assert_streams_work(1)

    def test_unreadable_unrelated_profiles_do_not_block_working_game(self):
        other = ProfileContext(game_id="other-game", camera_id="CAM-2", phone_id="PHONE-2",
                               panel_display=self.context.panel_display, game_display=self.context.game_display)
        for kind in ("rig_game", "phone_game"):
            with self.subTest(kind=kind):
                profile = self.registry.publish(kind, other, self.payload, activate=True)
                manifest = Path(profile["revision_directory"]) / "profile.json"
                content = manifest.read_text()
                manifest.unlink()
                try:
                    self.assertEqual(0, self.headless(self.rig_file("unrelated-" + kind, 1)))
                    self.assert_streams_work(1)
                finally:
                    manifest.write_text(content)

    def test_missing_established_geometry_source_preserves_active_graph(self):
        before = self.registry.active_revision_ids()
        (Path(self.phone["revision_directory"]) / "profile.json").unlink()
        with self.assertRaises((ProfileResolutionError, OSError)):
            self.headless(self.rig_file("missing-required-source", 1))
        self.assertEqual(before, self.registry.active_revision_ids())

    def test_changed_phone_dimensions_still_block_incompatible_geometry(self):
        before = self.registry.active_revision_ids()
        path = self.rig_file("wrong-dimensions", 1)
        document = json.loads(path.read_text())
        document["phone"]["natural_screen_size_px"] = [100, 240]
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ProfileResolutionError, "incompatible.*raster"):
            self.headless(path)
        self.assertEqual(before, self.registry.active_revision_ids())

    def test_frame_delivery_failure_preserves_active_graph(self):
        class FailedCamera(FakeAdapter):
            def read(self):
                raise RuntimeError("camera disconnected")
        before = self.registry.active_revision_ids()
        with self.assertRaisesRegex(ProfileResolutionError, "camera disconnected"):
            self.headless(self.rig_file("failed-read", 1), adapter=FailedCamera())
        self.assertEqual(before, self.registry.active_revision_ids())

    def test_composition_exception_does_not_activate_the_candidate_rig(self):
        before = self.registry.active_revision_ids()
        with patch("rig_runtime.workflows.profile_management.recompose_active_rig_game_orientation_profiles",
                   side_effect=RuntimeError("composition interrupted")):
            with self.assertRaisesRegex(RuntimeError, "composition interrupted"):
                publish_rig_calibration(self.rig_file("interrupted", 1), registry=self.registry)
        self.assertEqual(before, self.registry.active_revision_ids())
        self.assert_streams_work(0)

    def test_reuse_repairs_legacy_stale_dependents_before_success(self):
        staged = publish_rig_calibration(self.rig_file("legacy-partial", 1),
                                         registry=self.registry, activate=False)
        # Reconstruct the old version's partially activated state.
        self.registry.activate(staged["revision_id"])
        with self.assertRaisesRegex(ProfileResolutionError, "stale"):
            self.camera()
        path = self.registry.runtime_file(staged, "hik_camera_calibration")
        self.assertEqual(0, self.headless(path, reuse=True))
        self.assert_streams_work(1)
        receipt = json.loads((self.root / "reuse/reused_calibration.json").read_text())
        self.assertEqual("ready", receipt["game_readiness"]["status"])

    def test_displacement_reuses_color_and_never_blocks_headless_success(self):
        color = self.registry.publish("rig_game_color", self.context, {
            "hik_bayer_conversion": {"status": "selected", "gamma": 1,
                                    "ccm_rgb_3x3": np.eye(3).tolist()}},
            dependencies={"rig": self.initial["revision_id"]}, review_state="accepted", activate=True)
        path = self.rig_file("changed-color-rig", 1)
        self.assertEqual(0, self.headless(path))
        self.assert_streams_work(1)
        resolved = self.registry.resolve_adapter(self.context, AdapterRequest(
            mode="full", color_policy="game_matched"))
        self.assertEqual(color["revision_id"], resolved["profiles"]["rig_game_color"])
        self.assertEqual("game_matched", resolved["adapter_plan"]["color_policy"])
        self.assertNotEqual(self.initial["revision_id"], resolved["profiles"]["rig"])
        # The source fit and its measurement provenance remain immutable.
        self.assertEqual(self.initial["revision_id"], self.registry.revision(color["revision_id"])["dependencies"]["rig"])
        adapter = FakeAdapter()
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", return_value=adapter):
            with self.camera("dual", color_policy="game_matched") as camera:
                self.assertTrue(camera.get_frames())
        self.assertEqual([(1.0, np.eye(3).tolist())], adapter.bayer_conversion_calls)

    def test_missing_color_falls_back_even_when_game_matched_was_requested(self):
        self.assertEqual(0, self.headless(self.rig_file("no-color", 1)))
        adapter = FakeAdapter()
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", return_value=adapter):
            with self.camera("dual", color_policy="game_matched") as camera:
                self.assertTrue(camera.get_frames())
                self.assertEqual("rig_locked", camera.resolved_config["adapter_plan"]["color_policy"])
        self.assertEqual([], adapter.bayer_conversion_calls)

    def test_broken_optional_color_does_not_block_publication_or_output(self):
        color = self.registry.publish("rig_game_color", self.context,
                                      {"hik_bayer_conversion": {"status": "selected", "gamma": "invalid"}},
                                      dependencies={"rig": self.initial["revision_id"]}, activate=True)
        self.assertEqual(0, self.headless(self.rig_file("bad-color", 1)))
        resolved = self.registry.resolve_adapter(self.context, AdapterRequest(mode="full", color_policy="game_matched"))
        self.assertEqual("rig_locked", resolved["adapter_plan"]["color_policy"])
        (Path(color["revision_directory"]) / "profile.json").unlink()
        self.assertEqual(0, self.headless(self.rig_file("missing-color-file", 2)))
        resolved = self.registry.resolve_adapter(self.context, AdapterRequest(mode="full", color_policy="game_matched"))
        self.assertEqual("rig_locked", resolved["adapter_plan"]["color_policy"])

    def test_missing_manual_color_revision_is_non_blocking(self):
        resolved = self.registry.resolve_adapter(self.context, AdapterRequest(mode="full", color_policy="game_matched"),
                                                 profile_revisions={"rig_game_color": "missing-color"})
        self.assertEqual("rig_locked", resolved["adapter_plan"]["color_policy"])

    def test_embedded_adapter_game_matched_request_without_color_keeps_geometry(self):
        output = self.root / "without_color.py"
        export_resolved_adapter(output, registry=self.registry, context=self.context,
                                request=AdapterRequest(mode="dual", color_policy="game_matched"))
        specification = importlib.util.spec_from_file_location("without_color", output)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", return_value=FakeAdapter()):
            with module.HikCamera(config={"color_policy": "game_matched", "color_order": "BGR"}) as camera:
                frames = camera.get_frames()
                self.assertEqual("rig_locked", camera.resolved_config["adapter_plan"]["color_policy"])
                self.assertIn("color", camera.resolved_config["game_color_fallback"])
                for stream, frame in frames.items():
                    geometry = camera.get_minimap_geometry(stream)
                    self.assertTrue(geometry["available_in_stream_space"])
                    x, y = map(round, geometry["center_xy_px"])
                    np.testing.assert_array_equal(frame[y, x], [75, 30, 7])

    def test_color_file_lost_during_export_does_not_block_working_adapter(self):
        color = self.registry.publish("rig_game_color", self.context, {
            "hik_bayer_conversion": {"status": "selected", "gamma": 1,
                                    "ccm_rgb_3x3": np.eye(3).tolist()}},
            dependencies={"rig": self.initial["revision_id"]}, activate=True)
        color_path = Path(color["revision_directory"]) / "profile.json"
        resolve = self.registry.resolve_adapter

        def resolve_then_lose_color(*args, **kwargs):
            result = resolve(*args, **kwargs)
            color_path.unlink()
            return result

        output = self.root / "lost_color.py"
        with patch.object(self.registry, "resolve_adapter", side_effect=resolve_then_lose_color):
            result = export_resolved_adapter(output, registry=self.registry, context=self.context,
                                            request=AdapterRequest(mode="dual", color_policy="game_matched"))
        self.assertIsNone(result["profile_revisions"]["rig_game_color"])
        specification = importlib.util.spec_from_file_location("lost_color", output)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", return_value=FakeAdapter()):
            with module.HikCamera() as camera:
                self.assertTrue(camera.get_frames())
                self.assertEqual("rig_locked", camera.resolved_config["adapter_plan"]["color_policy"])

    def test_missing_optional_orientation_does_not_block_working_geometry(self):
        with patch("rig_runtime.workflows.profile_management.recompose_active_rig_game_orientation_profiles", return_value=[]):
            published = publish_rig_calibration(self.rig_file("no-orientation", 1), registry=self.registry)
        self.assertEqual("ready", published["readiness"]["status"])
        self.assertTrue(published["readiness"]["notices"])
        self.assert_streams_work(1)

    def test_unreconstructable_legacy_orientation_does_not_block_rebuilding(self):
        context = ProfileContext(game_id="legacy-game", camera_id="CAM-1", phone_id="PHONE-1",
                                 panel_display=self.context.panel_display, game_display=self.context.game_display)
        self.registry.publish("rig_game_orientation", context,
                              {"camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 1}, activate=True)
        published = publish_rig_calibration(self.rig_file("legacy-orientation", 1), registry=self.registry)
        self.assertEqual("ready", published["readiness"]["status"])
        self.assert_streams_work(1)

    def test_bad_optional_axes_do_not_block_boundary_rebuilding(self):
        from rig_runtime.workflows.profile_management import recompose_active_rig_game_profiles
        import copy
        payload = copy.deepcopy(self.payload)
        payload["outer_boundary"]["orientation_frame"]["up_unit_xy"] = [0, 0]
        self.registry.publish("phone_game", self.context, payload, activate=True)
        # Establish this source as the selected game before rebuilding the rig.
        previous = self.registry.resolve("rig_game", self.context)
        rig = self.registry.revision(previous["dependencies"]["rig"])
        with patch("rig_runtime.workflows.profile_management._portable_sources_for_rig", return_value=[
                self.registry.resolve("phone_game", self.context)]):
            recompose_active_rig_game_profiles(rig, registry=self.registry)
        published = publish_rig_calibration(self.rig_file("bad-axes", 1), registry=self.registry)
        self.assertEqual("ready", published["readiness"]["status"])
        self.assertTrue(any("axes" in notice for notice in published["readiness"]["notices"]))
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", return_value=FakeAdapter()):
            with self.camera("dual") as camera:
                camera.get_frames()
                self.assertTrue(camera.get_minimap_geometry("minimap")["available_in_stream_space"])

    def test_missing_game_identity_does_not_make_color_a_requirement(self):
        context = ProfileContext(camera_id="CAM-1", phone_id="PHONE-1", panel_display=self.context.panel_display)
        resolved = self.registry.resolve_adapter(context, AdapterRequest(mode="full", color_policy="game_matched"))
        self.assertEqual("rig_locked", resolved["adapter_plan"]["color_policy"])

    def test_sdk_color_failure_reopens_without_correction(self):
        self.registry.publish("rig_game_color", self.context, {
            "hik_bayer_conversion": {"status": "selected", "gamma": 1.2, "ccm_rgb_3x3": np.eye(3).tolist()}},
            dependencies={"rig": self.initial["revision_id"]}, activate=True)

        class MissingColorSdk(FakeAdapter):
            def set_bayer_conversion(self, gamma, ccm):
                raise RuntimeError("fused color conversion unsupported")

        failed, fallback = MissingColorSdk(), FakeAdapter()
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", side_effect=[failed, fallback]):
            with self.camera("dual", color_policy="game_matched") as camera:
                self.assertTrue(camera.get_frames())
                self.assertEqual("rig_locked", camera.resolved_config["adapter_plan"]["color_policy"])
                self.assertIn("unsupported", camera.resolved_config["game_color_fallback"])
        self.assertTrue(failed.closed)
        self.assertEqual([], fallback.bayer_conversion_calls)

    def test_color_file_lost_between_resolution_and_open_is_non_blocking(self):
        color = self.registry.publish("rig_game_color", self.context, {
            "hik_bayer_conversion": {"status": "selected", "gamma": 1.2, "ccm_rgb_3x3": np.eye(3).tolist()}},
            dependencies={"rig": self.initial["revision_id"]}, activate=True)
        camera = self.camera("dual", color_policy="game_matched")
        (Path(color["revision_directory"]) / "profile.json").unlink()
        with patch("rig_runtime.adapters.hik.game_camera.HikMvsCameraAdapter", return_value=FakeAdapter()):
            with camera:
                self.assertTrue(camera.get_frames())
                self.assertEqual("rig_locked", camera.resolved_config["adapter_plan"]["color_policy"])

    def test_active_graph_recovers_as_a_unit_without_database(self):
        publish_rig_calibration(self.rig_file("portable", 1), registry=self.registry)
        expected = self.registry.active_revision_ids()
        moved = self.root / "moved-profiles"
        shutil.copytree(self.registry.root, moved, ignore=shutil.ignore_patterns(".registry"))
        # Simulate an interrupted update of the old per-profile mirror.
        pointer = next(moved.rglob("active.json"))
        pointer.unlink()
        recovered = ProfileRegistry(moved)
        self.assertEqual(expected, recovered.active_revision_ids())

    def test_database_error_rolls_back_every_activation_and_preserves_snapshot(self):
        from rig_runtime.workflows.profile_management import reconcile_active_rig_dependents
        staged = publish_rig_calibration(self.rig_file("sql-failure", 1),
                                         registry=self.registry, activate=False)
        dependents = reconcile_active_rig_dependents(staged, registry=self.registry, activate=False)
        revisions = [staged["revision_id"]] + [
            item["revision_id"] for group in dependents["recomposed"].values() for item in group]
        before = self.registry.active_revision_ids()
        snapshot = (self.registry.root / "active-profiles.json").read_bytes()
        with contextlib.closing(sqlite3.connect(self.registry.database)) as connection, connection:
            connection.execute("CREATE TRIGGER reject_game BEFORE UPDATE ON active_profiles "
                               "WHEN NEW.kind='rig_game' BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected failure"):
            self.registry.activate_many(revisions, expected_active=before)
        self.assertEqual(before, self.registry.active_revision_ids())
        self.assertEqual(snapshot, (self.registry.root / "active-profiles.json").read_bytes())
        self.assert_streams_work(0)

    def test_concurrent_profile_change_prevents_outdated_publication(self):
        before = self.registry.active_revision_ids()
        staged = publish_rig_calibration(self.rig_file("concurrent", 1),
                                         registry=self.registry, activate=False)
        self.registry.publish("phone_game", self.context, {**self.payload, "generation": 2},
                              review_state="accepted", activate=True)
        current = self.registry.active_revision_ids()
        with self.assertRaisesRegex(RuntimeError, "Profiles changed during validation"):
            self.registry.activate_many([staged["revision_id"]], expected_active=before)
        self.assertEqual(current, self.registry.active_revision_ids())

    def test_damaged_portable_mirror_does_not_override_healthy_live_database(self):
        before = self.registry.active_revision_ids()
        (self.registry.root / "active-profiles.json").write_text("interrupted copy")
        reopened = ProfileRegistry(self.registry.root)
        self.assertEqual(before, reopened.active_revision_ids())


class AnnotationExplanationTests(unittest.TestCase):
    def test_equal_sized_geometry_from_a_different_space_is_explained(self):
        camera = Mock()
        camera.get_minimap_geometry.return_value = {
            "available_in_stream_space": True, "center_xy_px": [20, 20],
            "image_space": {"stored_size_px": [100, 80], "space_id": "previous-crop"},
        }
        camera.get_iris_frame_metadata.return_value = {
            "image_space": {"stored_size_px": [100, 80], "space_id": "current-crop"},
        }
        frame = np.zeros((80, 100, 3), np.uint8)
        state = gui.GeometryOverlayState(cursor=False)
        rendered = gui.overlay_stream_geometry(frame, camera, "full", state)
        self.assertIn("coordinate space", state.status_by_stream["full"]["Boundary"])
        np.testing.assert_array_equal(frame, rendered[:80, :100])
        camera.get_iris_frame_metadata.side_effect = RuntimeError("metadata unavailable")
        gui.overlay_stream_geometry(frame, camera, "full", state)
        self.assertIn("metadata unavailable", state.status_by_stream["full"]["Boundary"])

    def test_provider_errors_and_disable_state_are_visible_without_log_spam(self):
        camera = Mock()
        camera.get_minimap_geometry.side_effect = ValueError("wrong coordinate space")
        camera.get_cursor_geometry.return_value = {"available_in_stream_space": False,
                                                   "reason": "Unrectified camera ROI is projective"}
        frame = np.zeros((80, 100, 3), np.uint8)
        state = gui.GeometryOverlayState()
        with patch.object(gui, "print") as log:
            first = gui.overlay_stream_geometry(frame, camera, "full", state)
            gui.overlay_stream_geometry(frame, camera, "full", state)
        self.assertEqual(1, log.call_count)
        self.assertIn("wrong coordinate space", state.status_by_stream["full"]["Boundary"])
        self.assertIn("projective", state.status_by_stream["full"]["Cursor"])
        self.assertGreater(first.shape[0], frame.shape[0])
        self.assertTrue(np.array_equal(frame, first[:80, :100]))
        state.handle_key(ord("g"))
        gui.overlay_stream_geometry(frame, camera, "full", state)
        self.assertEqual({"disabled"}, set(state.status_by_stream["full"].values()))

    def test_bad_optional_axes_do_not_hide_valid_boundary(self):
        from tests.test_hik_game_camera import rig_document
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rig = root / "rig.json"
            rig.write_text(json.dumps(rig_document()))
            space = raster_space("android_phone_natural_display_pixels", [100, 80])
            boundary = oriented_circle(bind_geometry(
                {"center_x": 25, "center_y": 30, "radius": 6}, "circle", space))
            boundary["orientation_frame"]["up_unit_xy"] = [0, 0]
            game = root / "game.json"
            game.write_text(json.dumps({"canonical_phone_crop_xywh": [10, 20, 30, 20],
                                        "outer_boundary": boundary}))
            camera = ProfiledHikGameCamera(rig, game, mode="dual", adapter=FakeAdapter()).open()
            try:
                camera.read_streams()
                geometry = camera.get_minimap_geometry("full")
                self.assertTrue(geometry["available_in_stream_space"])
                self.assertIn("non-zero", geometry["orientation_reason"])
                self.assertNotIn("orientation_frame", geometry)
            finally:
                camera.release()
