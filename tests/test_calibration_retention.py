"""Automatic maintenance uses real registry retention at completion boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rig_runtime.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from rig_runtime.workflows import calibration_retention as cleanup
from rig_runtime.workflows.game_calibration import calibrate_game_sessions
from rig_runtime.workflows.profile_management import publish_rig_calibration
from tests.test_profile_manager import write_rig


class CalibrationRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = ProfileRegistry(self.root / "profiles")

    def test_accepted_rig_publication_applies_defaults_after_activation(self):
        for index in range(12):
            self.registry.publish("game_model", ProfileContext(game_id="game"), {"version": index})
        for index in range(5):
            publish_rig_calibration(write_rig(self.root / str(index), camera_x_offset=index),
                                    registry=self.registry, activate=False)
        self.assertEqual(5, len(self.registry.list_revisions(kind="rig")))
        purge = cleanup.purge_profiles

        def apply(registry, **kwargs):
            active = registry.active_revision_ids()
            self.assertEqual(1, len(active))
            self.assertEqual(6, len(registry.list_revisions(kind="rig")))
            return purge(registry, **kwargs)

        with mock.patch.object(cleanup, "purge_profiles", side_effect=apply) as called:
            profile = publish_rig_calibration(write_rig(self.root / "accepted"), registry=self.registry)
        called.assert_called_once()
        self.assertEqual(3, len(self.registry.list_revisions(kind="rig")))
        self.assertEqual(10, len(self.registry.list_revisions(kind="game_model")))
        self.assertIn(profile["revision_id"], self.registry.active_revision_ids())
        self.assertTrue(Path(profile["revision_directory"]).is_dir())

    def test_candidate_and_failed_readiness_do_not_trigger_cleanup(self):
        path = write_rig(self.root / "rig")
        with mock.patch.object(cleanup, "purge_profiles") as purge:
            publish_rig_calibration(path, registry=self.registry, activate=False)
            with mock.patch("rig_runtime.workflows.rig_readiness.validate_rig_configuration",
                            side_effect=RuntimeError("bad dimensions")):
                with self.assertRaisesRegex(RuntimeError, "bad dimensions"):
                    publish_rig_calibration(path, registry=self.registry)
        purge.assert_not_called()
        self.assertEqual([], self.registry.active_revision_ids())

    def test_cleanup_error_does_not_undo_or_fail_accepted_calibration(self):
        with mock.patch.object(cleanup, "purge_profiles", side_effect=PermissionError("WinError 5")), \
                self.assertLogs(cleanup.__name__, level="WARNING") as messages:
            profile = publish_rig_calibration(write_rig(self.root / "rig"), registry=self.registry)
        self.assertIn(profile["revision_id"], self.registry.active_revision_ids())
        self.assertIn("cleanup deferred", " ".join(messages.output))
        self.assertIn("WinError 5", " ".join(messages.output))
        # Failure resets batching state, allowing the next run to retry.
        with mock.patch.object(cleanup, "purge_profiles", wraps=cleanup.purge_profiles) as purge:
            publish_rig_calibration(write_rig(self.root / "retry"), registry=self.registry)
        purge.assert_called_once()

    def test_multi_session_cleanup_waits_for_final_summary_and_runs_once(self):
        output = self.root / "output"
        summaries_at_cleanup = []
        sessions = [self.root / "zigzag", self.root / "micro"]
        descriptors = {path.resolve(): dict(path=path.resolve(), order=index, game_id="game",
                                            acquisition_patterns=["zigzag" if index == 0 else "micro_movement"])
                       for index, path in enumerate(sessions)}

        def component(session, child_output, **kwargs):
            cleanup.request_profile_cleanup(self.registry)
            purge.assert_not_called()
            return {"status": "partial", "successful_capabilities": ["cursor_pose"], "capabilities": {}}

        def apply(registry, **kwargs):
            summaries_at_cleanup.append((output / "game_calibration_summary.json").is_file())
            return {"removed_revisions": [], "removed_evidence": [], "warnings": []}

        with mock.patch("rig_runtime.workflows.game_calibration._session_calibration_descriptor",
                        side_effect=lambda path: descriptors[path]), \
                mock.patch("rig_runtime.workflows.game_calibration.calibrate_game_session", side_effect=component), \
                mock.patch.object(cleanup, "purge_profiles", side_effect=apply) as purge:
            result = calibrate_game_sessions(sessions, output, profile_root=self.registry.root)
        purge.assert_called_once_with(self.registry, dry_run=False)
        self.assertEqual([True], summaries_at_cleanup)
        self.assertEqual("partial", result["status"])

    def test_outer_failure_only_cleans_already_accepted_components(self):
        @cleanup.calibration_cleanup_boundary
        def run(accepted):
            if accepted:
                cleanup.request_profile_cleanup(self.registry)
            raise RuntimeError("later component failed")

        with mock.patch.object(cleanup, "purge_profiles", wraps=cleanup.purge_profiles) as purge:
            with self.assertRaisesRegex(RuntimeError, "later component failed"):
                run(False)
            purge.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "later component failed"):
                run(True)
            purge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
