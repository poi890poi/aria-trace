import tempfile
import unittest
from pathlib import Path

from acquisition.calibration_profiles import (
    CalibrationProfileKey,
    CalibrationProfileStore,
)


class CalibrationProfileStoreTests(unittest.TestCase):
    def test_profile_scope_includes_rig_game_and_image_source(self):
        android = CalibrationProfileKey("rig-1", "game-1", "android_scrcpy")
        hik = CalibrationProfileKey("rig-1", "game-1", "hik_mvs")
        self.assertNotEqual(android.profile_id, hik.profile_id)
        self.assertEqual("rig-1--game-1--android_scrcpy", android.profile_id)

    def test_publish_writes_immutable_revision_manifest_and_current_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = CalibrationProfileStore(root)
            key = CalibrationProfileKey("rig-1", "game-1", "hik_mvs")
            revision = store.create_revision_directory(key, "revision-1")
            calibration = revision / "minimap_calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            manifest = store.publish(
                key,
                revision,
                {"minimap_calibration": str(calibration)},
                session_path=root / "session",
            )
            current = store.current(key)
            self.assertEqual("review_required", manifest["status"])
            self.assertEqual("revision-1", current["current_revision_id"])
            self.assertEqual(str(revision.resolve()), current["current_revision"])
            self.assertTrue((revision / "profile_revision.json").is_file())
            self.assertTrue((revision / "profile_revision.yaml").is_file())
            self.assertTrue(
                (store.profile_directory(key) / "current.yaml").is_file()
            )
            yaml_text = (revision / "profile_revision.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("# Stable scope and identity", yaml_text)

    def test_revision_ids_cannot_overwrite_an_existing_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = CalibrationProfileStore(Path(temporary))
            key = CalibrationProfileKey("rig-1", "game-1", "hik_mvs")
            store.create_revision_directory(key, "revision-1")
            with self.assertRaises(FileExistsError):
                store.create_revision_directory(key, "revision-1")


if __name__ == "__main__":
    unittest.main()
