"""Release packages exclude unrelated checkout files and unsafe archive paths."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.build_swiftest import stage_source, unpack_source


class BuildTests(unittest.TestCase):
    def test_staging_ignores_stale_builds_and_generated_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkout"
            selected = {"pyproject.toml": "[project]\nname='swiftest'\n",
                        "docs/swiftest.md": "# SwifTest\n",
                        "swiftest/__init__.py": "# current source\n",
                        "swiftest/assets/IntentScreenshot.java": "// helper\n"}
            junk = {"build/lib/androidtestkit/__init__.py": "# stale\n",
                    "swiftest/__pycache__/cached.pyc": "cache",
                    "swiftest/assets/trace.json": "raw telemetry",
                    "other_project/private.txt": "unrelated"}
            for name, content in dict(selected, **junk).items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            staged = Path(temp) / "staged"
            stage_source(root, staged)
            self.assertEqual({p.relative_to(staged).as_posix()
                              for p in staged.rglob('*') if p.is_file()},
                             set(selected) | {"README.md"})
            self.assertEqual((staged / "swiftest/__init__.py").read_text(),
                             selected["swiftest/__init__.py"])

    def test_archive_paths_cannot_escape_extraction_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                entry = tarfile.TarInfo("../escaped.txt")
                entry.size = 4
                bundle.addfile(entry, io.BytesIO(b"data"))
            with self.assertRaises(ValueError):
                unpack_source(archive, Path(temp) / "unpacked")
            self.assertFalse((Path(temp) / "escaped.txt").exists())
