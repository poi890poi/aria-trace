"""Direct acquisition reports capability and never reuses an image on failure."""
import base64
import io
import queue
import unittest
from unittest.mock import Mock

from swiftest.capture import Capture


class CaptureTests(unittest.TestCase):
    def capture(self, wire):
        capture = Capture.__new__(Capture)
        capture.phone = Mock()
        capture.process = Mock(stdin=Mock(), stdout=io.BytesIO(wire))
        capture.responses = queue.Queue()
        capture.mode = "jpeg"
        capture.prefer_direct = True
        capture.prefer_bitmap = False
        capture.acquisition = capture.reason = capture.previous = None
        capture.repeat = False
        capture.bytes = 0
        capture._receive()
        return capture

    def test_direct_frame_and_exact_repeat(self):
        jpeg = b"\xff\xd8image\xff\xd9"
        capture = self.capture(b"D:"+base64.b64encode(jpeg)+b"\r\nD:\r\n")
        self.assertEqual(capture.read(), jpeg)
        self.assertEqual(capture.acquisition, "direct")
        self.assertFalse(capture.repeat)
        self.assertEqual(capture.read(), jpeg)
        self.assertTrue(capture.repeat)

    def test_unavailable_direct_uses_current_legacy_capture_and_reports_it(self):
        jpeg = b"\xff\xd8new\xff\xd9"
        capture = self.capture(b"S:"+base64.b64encode(jpeg)+b"\n")
        capture.previous = b"old image"
        self.assertEqual(capture.read(), jpeg)
        self.assertFalse(capture.repeat)
        self.assertEqual(capture.acquisition, "screencap")
        self.assertIn("unavailable", capture.reason)

    def test_bitmap_request_reports_exact_repeat_without_a_second_decode(self):
        jpeg = b"\xff\xd8image\xff\xd9"
        capture = self.capture(b"D:"+base64.b64encode(jpeg)+b"\nD:\n")
        capture.prefer_bitmap = True
        self.assertEqual(capture.read(), jpeg)
        self.assertEqual(capture.read(), jpeg)
        self.assertTrue(capture.repeat)
        self.assertEqual(capture.acquisition, "direct-bitmap")
        self.assertEqual(capture.process.stdin.write.call_args.args, (b"R",))

    def test_empty_first_response_requires_a_fresh_fallback_image(self):
        capture = self.capture(b"D:\n")
        capture.phone.run.return_value = b"fresh png"
        self.assertEqual(capture.read(), b"fresh png")
        self.assertFalse(capture.repeat)
        self.assertEqual(capture.acquisition, "screencap-png")

    def test_stream_failure_acquires_png_instead_of_holding_previous_image(self):
        capture = self.capture(b"invalid\n")
        capture.previous = b"old image"
        capture.phone.run.return_value = b"fresh png"
        self.assertEqual(capture.read(), b"fresh png")
        self.assertFalse(capture.repeat)
        self.assertEqual(capture.acquisition, "screencap-png")
        capture.phone.run.assert_called_once_with("exec-out", "screencap", "-p")


if __name__ == "__main__":
    unittest.main()
