import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "restore-workbench-backup.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class RestoreWorkbenchBackupTests(unittest.TestCase):
    def repository(self, parent):
        repository = Path(parent) / "fresh-clone"
        (repository / ".git").mkdir(parents=True)
        (repository / "acquisition").mkdir()
        (repository / "acquisition" / "workbench.py").write_text("# marker\n")
        return repository

    def run_restore(self, archive, repository, source_root, expected_hash=None):
        command = [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Archive",
            str(archive),
            "-RepoRoot",
            str(repository),
            "-SourceRoot",
            source_root,
        ]
        if expected_hash:
            command += ["-ExpectedSha256", expected_hash]
        return subprocess.run(command, text=True, capture_output=True)

    def test_restores_allowed_roots_and_rebases_absolute_provenance_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            archive = Path(directory) / "backup.zip"
            source_root = r"E:\workspace\aria-trace"
            source_path = source_root + r"\sessions\workbench\run_01"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(
                    "sessions/workbench/run_01/manifest.json",
                    json.dumps({"session_path": source_path}),
                )
                output.writestr(
                    "artifacts/workbench/workbench_state.json",
                    json.dumps({"active": None}),
                )
                output.writestr(
                    "profiles/rig/device/active.json",
                    json.dumps({"profile": source_root + r"\profiles\rig\device"}),
                )
            expected_hash = hashlib.sha256(archive.read_bytes()).hexdigest()

            result = self.run_restore(
                archive, repository, source_root, expected_hash
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            manifest = json.loads(
                (repository / "sessions/workbench/run_01/manifest.json").read_text()
            )
            self.assertEqual(
                str(repository) + r"\sessions\workbench\run_01",
                manifest["session_path"],
            )
            self.assertIn("rebased_text_files", result.stdout)

    def test_rejects_archive_parent_traversal_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            archive = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "unsafe")

            result = self.run_restore(
                archive, repository, r"E:\workspace\aria-trace"
            )
            self.assertNotEqual(0, result.returncode)
            self.assertFalse((Path(directory) / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
