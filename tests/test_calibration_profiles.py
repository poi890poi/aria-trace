import tempfile
import unittest
from pathlib import Path

from acquisition.calibration_profiles import (
    CalibrationProfileKey,
    CalibrationProfileStore,
    ScopedCalibrationProfileStore,
    ScopedProfileKey,
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

    def test_phone_game_and_rig_game_are_independent_profile_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ScopedCalibrationProfileStore(Path(temporary))
            phone = ScopedProfileKey("phone_game", "phone-1", "game-1")
            rig = ScopedProfileKey("rig_game", "camera-1--phone-1", "game-1")
            phone_revision = store.create_revision_directory(phone, "r1")
            rig_revision = store.create_revision_directory(rig, "r1")
            store.publish(phone, phone_revision, {"coordinate_space": "phone"})
            store.publish(rig, rig_revision, {"base_rig_calibration": "rig.json"})
            self.assertNotEqual(
                store.profile_directory(phone), store.profile_directory(rig)
            )
            self.assertTrue((phone_revision / "profile.yaml").is_file())
            self.assertIn(
                "# AriaTrace calibration profile",
                (rig_revision / "profile.yaml").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
