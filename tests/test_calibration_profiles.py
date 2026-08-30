import tempfile
import unittest
from pathlib import Path

from acquisition.calibration_profiles import (
    CalibrationProfileKey,
    CalibrationProfileStore,
    ScopedCalibrationProfileStore,
    ScopedProfileKey,
)


class ObsoleteCalibrationProfileStoreTests(unittest.TestCase):
    def test_legacy_identity_values_remain_readable_for_migration(self):
        key = CalibrationProfileKey("rig-1", "game-1", "hik_mvs")
        scoped = ScopedProfileKey("phone_game", "phone-1", "game-1")
        self.assertEqual("rig-1--game-1--hik_mvs", key.profile_id)
        self.assertEqual("phone_game--phone-1--game-1", scoped.profile_id)

    def test_legacy_stores_fail_before_writing_mutable_current_pointers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "obsolete"):
                CalibrationProfileStore(root)
            with self.assertRaisesRegex(RuntimeError, "obsolete"):
                ScopedCalibrationProfileStore(root)
            self.assertEqual([], list(root.iterdir()))


if __name__ == "__main__":
    unittest.main()
