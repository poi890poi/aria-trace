import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from rig_runtime.adapters.filesystem.profile_registry import (
    AdapterRequest,
    ProfileContext,
    ProfileRegistry,
)
from rig_runtime.services.calibration.minimap.discovery import (
    discover_android_minimap_crop,
)
from rig_runtime.workflows.game_calibration import (
    _available_cursor_acquisition_series,
    _touch_intervals,
    calibrate_game_session,
    format_game_calibration_report,
)
from rig_runtime.workflows.game_orientation_calibration import (
    calibrate_portable_game_orientation_session,
)


class GameCalibrationTests(unittest.TestCase):
    def test_final_report_separates_status_reason_and_evidence(self):
        report = format_game_calibration_report(
            {
                "status": "partial",
                "game_id": "game",
                "capabilities": {
                    "minimap_boundary": {
                        "status": "review_required",
                        "calibration": "evidence/minimap.json",
                    },
                    "game_color": {
                        "status": "optional_failed_non_gating",
                        "error": "fit rejected",
                    },
                },
            },
            Path("output"),
        )
        text = "\n".join(report)
        self.assertIn("IRIS game calibration summary", text)
        self.assertIn("[REVIEW", text)
        self.assertIn("Evidence: evidence/minimap.json", text)
        self.assertIn("[OPTIONAL-WARN]", text)
        self.assertIn("Reason: fit rejected", text)

    def test_touch_intervals_accept_one_recorded_high_level_swipe(self):
        class Reader:
            inputs = [
                {
                    "kind": "zigzag_touch",
                    "session_time_ns": 2_000_000_000,
                    "payload": {
                        "action": "SWIPE",
                        "duration_ms": 350,
                        "command_start_host_time_ns": 10,
                        "command_end_host_time_ns": 350_000_010,
                    },
                }
            ]

        self.assertEqual(
            [(1_650_000_000, 2_000_000_000)],
            _touch_intervals(Reader()),
        )

    def test_cursor_uses_boundary_profile_from_same_candidate_run(self):
        class Reader:
            manifest = {
                "status": "complete",
                "session_id": "session",
                "context": {
                    "game_id": "game",
                    "phone_surface_orientation": {
                        "natural_size_px": [100, 200],
                        "logical_size_px": [200, 100],
                    },
                },
            }
            frames_by_stream = {"android_phone": [{"frame_index": 0}]}
            inputs = [
                {
                    "kind": "zigzag_touch",
                    "session_time_ns": 10,
                    "payload": {"action": "DOWN"},
                },
                {
                    "kind": "zigzag_touch",
                    "session_time_ns": 20,
                    "payload": {"action": "UP"},
                },
            ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            session.mkdir()
            output = root / "output"
            cursor_value = {
                "status": "partial",
                "result_level": "rotation_center_only",
                "capabilities": {
                    "rotation_center": {"status": "available"},
                    "cursor_shape": {"status": "unavailable"},
                },
                "failure_reasons": [
                    {"stage": "cursor_shape", "reason": "shape unavailable"}
                ],
                "profiles": {"phone_game": "cursor-phone-revision"},
            }
            with mock.patch(
                "rig_runtime.workflows.game_calibration.SessionReader",
                return_value=Reader(),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.load_system_configuration",
                return_value={
                    "game": {"game_id": "game"},
                    "devices": {"phone_id": None, "camera_id": None},
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.resolve_game_model",
                return_value={
                    "cursor_follows": "camera",
                    "cursor_behavior_by_acquisition": {
                        "zigzag": "rotating",
                        "micro_movement": "static",
                    },
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.calibrate_portable_game_orientation_session",
                side_effect=ValueError("not relevant"),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration._calibrate_available_minimap_boundary",
                return_value={
                    "status": "review_required",
                    "profiles": {"phone_game": "boundary-phone-revision"},
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration._calibrate_available_cursor_series",
                return_value=cursor_value,
            ) as cursor_calibration:
                result = calibrate_game_session(
                    session,
                    output,
                    profile_root=root / "profiles",
                    game_id="game",
                    activate=False,
                )

            self.assertEqual(
                "boundary-phone-revision",
                cursor_calibration.call_args[1]["phone_game_revision"],
            )
            cursor = result["capabilities"]["cursor_pose"]
            self.assertEqual("partial", cursor["status"])
            self.assertEqual(
                "rotation_center_only", cursor["series"]["zigzag"]["result_level"]
            )
            self.assertIn("shape unavailable", cursor["series"]["zigzag"]["reason"])

    def test_synchronized_hik_session_runs_and_activates_game_color_calibration(self):
        class Reader:
            manifest = {"status": "complete", "context": {"game_id": "game"}}
            frames_by_stream = {
                "android_phone": [{"frame_index": 0}],
                "hik_phone": [{"frame_index": 0}],
            }
            inputs = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            session.mkdir()
            (session / "coordinate_spaces.yaml").write_text(
                "schema_version: '1.0'\n", encoding="utf-8"
            )
            output = root / "output"
            color_result = {
                "status": "accepted",
                "profile_revision": "rig-game-color-revision",
            }
            with mock.patch(
                "rig_runtime.workflows.game_calibration.SessionReader",
                return_value=Reader(),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.load_system_configuration",
                return_value={
                    "game": {"game_id": "game"},
                    "devices": {"phone_id": None, "camera_id": None},
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.resolve_game_model",
                return_value={
                    "cursor_follows": "character",
                    "cursor_behavior_by_acquisition": {
                        "zigzag": "static",
                        "micro_movement": "rotating",
                    },
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.calibrate_portable_game_orientation_session",
                side_effect=ValueError("not relevant"),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration._calibrate_available_minimap_boundary",
                side_effect=ValueError("not relevant"),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.calibrate_game_color_session",
                return_value=color_result,
            ) as color_calibration:
                result = calibrate_game_session(
                    session,
                    output,
                    profile_root=root / "profiles",
                    game_id="game",
                    activate=True,
                    include_color=True,
                    activate_color=True,
                )

            color_calibration.assert_called_once_with(
                session.resolve(),
                output.resolve() / "color",
                profile_root=(root / "profiles").resolve(),
                game_id="game",
                maximum_pairs=12,
                activate=True,
                phone_game_revision=None,
            )
            self.assertEqual(
                "accepted", result["capabilities"]["game_color"]["status"]
            )
            self.assertEqual(
                "rig-game-color-revision",
                result["capabilities"]["game_color"]["profile_revision"],
            )

    def test_color_is_not_run_or_gating_by_default(self):
        class Reader:
            manifest = {"status": "complete", "context": {"game_id": "game"}}
            frames_by_stream = {
                "android_phone": [{"frame_index": 0}],
                "hik_phone": [{"frame_index": 0}],
            }
            inputs = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            session.mkdir()
            output = root / "output"
            with mock.patch(
                "rig_runtime.workflows.game_calibration.SessionReader",
                return_value=Reader(),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.load_system_configuration",
                return_value={
                    "game": {"game_id": "game"},
                    "devices": {"phone_id": None, "camera_id": None},
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.resolve_game_model",
                return_value={
                    "cursor_follows": "character",
                    "cursor_behavior_by_acquisition": {
                        "zigzag": "static",
                        "micro_movement": "rotating",
                    },
                },
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.calibrate_portable_game_orientation_session",
                side_effect=ValueError("not relevant"),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration._calibrate_available_minimap_boundary",
                side_effect=ValueError("not relevant"),
            ), mock.patch(
                "rig_runtime.workflows.game_calibration.calibrate_game_color_session"
            ) as color_calibration:
                result = calibrate_game_session(
                    session,
                    output,
                    profile_root=root / "profiles",
                    game_id="game",
                )
            color_calibration.assert_not_called()
            self.assertEqual(
                "optional_not_requested",
                result["capabilities"]["game_color"]["status"],
            )
            self.assertNotIn("game_color", result["successful_capabilities"])

    def test_portable_orientation_uses_android_space_metadata_without_rig(self):
        class Reader:
            manifest = {
                "status": "complete",
                "session_id": "session",
                "context": {
                    "game_id": "game",
                    "game_launch": {"package": "example.game"},
                },
                "devices": {"phone": {"serial": "PHONE"}},
            }
            frames_by_stream = {
                "android_phone": [
                    {
                        "frame_index": index,
                        "metadata": {
                            "image_space": {
                                "surface_quarter_turns_clockwise_from_canonical": 1,
                                "canonical_size_px": [100, 200],
                                "source_logical_size_px": [200, 100],
                                "orientation_source": "adb_surface",
                            }
                        },
                    }
                    for index in range(4)
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            session.mkdir()
            output = root / "orientation"
            with mock.patch(
                "rig_runtime.workflows.game_orientation_calibration.SessionReader",
                return_value=Reader(),
            ), mock.patch(
                "rig_runtime.workflows.game_orientation_calibration.decode_session_records",
                return_value=np.zeros((1, 100, 200, 3), np.uint8),
            ):
                result = calibrate_portable_game_orientation_session(
                    session,
                    output,
                    profile_root=root / "profiles",
                    game_id="game",
                    activate=True,
                )
            registry = ProfileRegistry(root / "profiles")
            profile = registry.revision(result["profile_revision"])
            self.assertEqual({}, profile["dependencies"])
            self.assertEqual("phone_game", profile["identity"]["kind"])
            self.assertEqual(
                1,
                profile["payload"][
                    "game_surface_quarter_turns_clockwise_from_phone_natural"
                ],
            )
            self.assertIsNone(result["rig_dependency"])

    def test_cursor_series_classification_describes_input_not_cursor_behavior(self):
        class Reader:
            manifest = {
                "context": {
                    "capture_kind": "micro_movement_game_calibration_source_data"
                }
            }
            frames_by_stream = {"android_phone": [{"frame_index": 0}]}
            inputs = [
                {
                    "kind": "micro_movement_touch",
                    "session_time_ns": 10,
                    "payload": {"action": "DOWN"},
                },
                {
                    "kind": "micro_movement_touch",
                    "session_time_ns": 20,
                    "payload": {"action": "UP"},
                },
            ]

        self.assertEqual(
            [
                {
                    "acquisition_pattern": "micro_movement",
                    "touch_kind": "micro_movement_touch",
                }
            ],
            _available_cursor_acquisition_series(Reader()),
        )

    def test_bounded_discovery_finds_dynamic_upper_left_circle(self):
        height, width = 360, 640
        frames = []
        rng = np.random.default_rng(4)
        for index in range(16):
            image = np.full((height, width, 3), 30, np.uint8)
            texture = rng.integers(20, 220, (112, 112, 3), dtype=np.uint8)
            mask = np.zeros((112, 112), np.uint8)
            cv2.circle(mask, (56, 56), 52, 255, -1)
            area = image[24:136, 28:140]
            area[mask > 0] = texture[mask > 0]
            cv2.circle(image, (84, 80), 52, (220, 220, 220), 2)
            cv2.putText(image, str(index), (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 90), 2)
            frames.append(image)
        result = discover_android_minimap_crop(np.stack(frames))
        center = result["selected_candidate"]["center_xy"]
        self.assertLess(abs(center[0] - 84), 12)
        self.assertLess(abs(center[1] - 80), 12)
        self.assertLess(abs(result["selected_candidate"]["radius_px"] - 52), 14)

    def test_orientation_profile_is_optional_and_resolved_for_full_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "rig.json"
            calibration.write_text(json.dumps({"camera": {}}), encoding="utf-8")
            context = ProfileContext(
                game_id="game",
                camera_id="CAM",
                panel_display={"natural_panel_px": [100, 200]},
                game_display={
                    "logical_frame_px": [200, 100],
                    "rotation_quarter_turns": 1,
                },
            )
            registry = ProfileRegistry(root / "profiles")
            rig = registry.publish(
                "rig", context, {},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted", activate=True,
            )
            orientation = registry.publish(
                "rig_game_orientation",
                context,
                {"camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3},
                dependencies={"rig": rig["revision_id"]},
                review_state="accepted", activate=True,
            )
            resolved = registry.resolve_adapter(context, AdapterRequest(mode="full"))
            self.assertEqual(
                orientation["revision_id"],
                resolved["profiles"]["rig_game_orientation"],
            )
            self.assertEqual(
                3,
                resolved["adapter_plan"][
                    "game_upright_quarter_turns_clockwise"
                ],
            )

    def test_stale_orientation_profile_warns_and_does_not_gate_full_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "rig.json"
            calibration.write_text(json.dumps({"camera": {}}), encoding="utf-8")
            context = ProfileContext(
                game_id="game",
                camera_id="CAM",
                panel_display={"natural_panel_px": [100, 200]},
                game_display={"logical_frame_px": [200, 100]},
            )
            registry = ProfileRegistry(root / "profiles")
            first_rig = registry.publish(
                "rig",
                context,
                {"generation": 1},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted",
                activate=True,
            )
            registry.publish(
                "rig_game_orientation",
                context,
                {
                    "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3
                },
                dependencies={"rig": first_rig["revision_id"]},
                review_state="accepted",
                activate=True,
            )
            registry.publish(
                "rig",
                context,
                {"generation": 2},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted",
                activate=True,
            )

            with self.assertWarnsRegex(
                RuntimeWarning, "ignored stale active revisions"
            ):
                resolved = registry.resolve_adapter(
                    context, AdapterRequest(mode="full")
                )
            self.assertIsNone(resolved["profiles"]["rig_game_orientation"])
            self.assertEqual(0, resolved["adapter_plan"][
                "game_upright_quarter_turns_clockwise"
            ])


if __name__ == "__main__":
    unittest.main()
