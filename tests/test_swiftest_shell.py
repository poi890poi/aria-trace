"""Command framing and uncertain-input failures must not cause duplicate input."""

import io
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swiftest.shell import Shell, ShellError


class ShellTests(unittest.TestCase):
    def channel(self, output):
        process = Mock(stdin=io.BytesIO(), stdout=io.BytesIO(output))
        with patch("swiftest.shell.subprocess.Popen", return_value=process):
            channel = Shell("adb", "phone")
        self.addCleanup(channel.close)
        return channel, process

    @patch("swiftest.shell.uuid.uuid4", return_value=SimpleNamespace(hex="test"))
    def test_windows_adb_translation_preserves_original_crlf_and_empty_output(self, _):
        channel, _ = self.channel(b"line\r\r\nlast\r\ntestkit_test:0\r\n\r\ntestkit_test:0\r\n")
        self.assertEqual(channel.run("first"), b"line\r\nlast")
        self.assertEqual(channel.run("second"), b"")

    @patch("swiftest.shell.uuid.uuid4", return_value=SimpleNamespace(hex="test"))
    def test_framing_preserves_unicode_blank_lines_and_no_final_newline(self, _):
        channel, _ = self.channel("中文\n\nlast\ntestkit_test:0\n\ntestkit_test:0\n".encode())
        self.assertEqual(channel.run("first"), "中文\n\nlast".encode())
        self.assertEqual(channel.run("second"), b"")

    @patch("swiftest.shell.uuid.uuid4", return_value=SimpleNamespace(hex="test"))
    def test_nonzero_status_is_an_error_and_closes_channel(self, _):
        channel, process = self.channel(b"permission denied\n\ntestkit_test:1\n")
        with self.assertRaisesRegex(ShellError, "permission denied"):
            channel.run("input tap 1 2")
        self.assertTrue(channel.closed)
        process.terminate.assert_called_once()
        with self.assertRaisesRegex(ShellError, "not sent"):
            channel.run("input tap 1 2")

    def test_disconnect_is_not_retried(self):
        channel, process = self.channel(b"")
        channel.process.stdin = Mock()
        with self.assertRaisesRegex(ShellError, "outcome is unknown"):
            channel.run("input tap 1 2")
        channel.process.stdin.write.assert_called_once()
        process.terminate.assert_called_once()

    def test_timeout_closes_channel_and_does_not_retry(self):
        channel, process = self.channel(b"")
        channel.responses = Mock(get=Mock(side_effect=queue.Empty))
        channel.process.stdin = Mock()
        with self.assertRaisesRegex(ShellError, "not retried"):
            channel.run("input tap 1 2", timeout=.01)
        channel.process.stdin.write.assert_called_once()
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
