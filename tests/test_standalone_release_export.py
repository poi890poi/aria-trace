import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StandaloneReleaseExportTests(unittest.TestCase):
    def test_build_rejects_git_metadata_and_records_exported_tree_hash(self):
        script = (ROOT / "build-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Get-PortableTreeSha256", script)
        self.assertIn("exported_python_source_tree_sha256", script)
        self.assertIn("git_metadata_included: false", script)
        self.assertIn("Release export contains forbidden Git metadata", script)

    def test_publish_recomputes_exported_tree_hash_and_rejects_git_metadata(self):
        script = (ROOT / "publish-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Get-PortableTreeSha256", script)
        self.assertIn("Exported Python source-tree hash mismatch", script)
        self.assertIn("Release package contains forbidden Git metadata", script)


if __name__ == "__main__":
    unittest.main()
