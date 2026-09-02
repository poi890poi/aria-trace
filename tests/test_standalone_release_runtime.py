import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aria_trace.adapters.android.capture import find_scrcpy_server
from aria_trace.adapters.filesystem.video import find_ffmpeg
from aria_trace.adapters.runtime_tools import find_release_tool


class StandaloneReleaseRuntimeTests(unittest.TestCase):
    def test_release_tree_is_found_from_apps_and_python_subdirectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release-manifest.yaml").write_text("product: iris\n")
            tool = root / "third_party" / "ffmpeg" / "bin" / "ffmpeg.exe"
            tool.parent.mkdir(parents=True)
            tool.write_bytes(b"ffmpeg")
            self.assertEqual(
                tool,
                find_release_tool(
                    "third_party/ffmpeg/bin/ffmpeg.exe",
                    anchors=(root / "apps" / "iris-tool", root / "python"),
                ),
            )

    def test_scrcpy_server_uses_release_environment_path(self):
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory) / "scrcpy-server"
            server.write_bytes(b"server")
            with mock.patch.dict(
                os.environ, {"IRIS_SCRCPY_SERVER": str(server)}, clear=False
            ), mock.patch("shutil.which", return_value=None):
                self.assertEqual(server, find_scrcpy_server())

    def test_ffmpeg_uses_release_environment_path(self):
        with tempfile.TemporaryDirectory() as directory:
            ffmpeg = Path(directory) / "ffmpeg.exe"
            ffmpeg.write_bytes(b"ffmpeg")
            with mock.patch.dict(
                os.environ, {"IRIS_FFMPEG": str(ffmpeg)}, clear=False
            ), mock.patch("shutil.which", return_value=None):
                self.assertEqual(ffmpeg, find_ffmpeg())


if __name__ == "__main__":
    unittest.main()
