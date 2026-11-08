"""The experimental transport cannot turn truncated or invalid frames into reuse."""
import struct
import unittest
from unittest.mock import Mock

from benchmarks.android.binary_capture import BinaryCapture


class BinaryCaptureProbeTests(unittest.TestCase):
    def backend(self, packet):
        capture = BinaryCapture.__new__(BinaryCapture)
        capture.mode = "jpeg"
        capture.bytes = 0
        capture.repeat = False
        capture.previous = b"\xff\xd8previous\xff\xd9"
        capture.reason = None
        capture.phone = Mock()
        capture.phone.run.return_value = b"fresh png"
        capture.connection = Mock()
        # Deliberately fragmented socket reads.
        remaining = bytearray(packet)
        def recv(size):
            result = bytes(remaining[:min(size,3)])
            del remaining[:len(result)]
            return result
        capture.connection.recv.side_effect = recv
        return capture

    def test_fragmented_complete_frame_and_repeat(self):
        data = b"\xff\xd8new-image\xff\xd9"
        capture = self.backend(struct.pack("!4scI",b"ATK1",b"D",len(data))+data+
                               struct.pack("!4scI",b"ATK1",b"D",0))
        self.assertEqual(capture.read(),data)
        self.assertFalse(capture.repeat)
        self.assertEqual(capture.read(),data)
        self.assertTrue(capture.repeat)
        self.assertEqual(capture.acquisition,"direct-bitmap")
        capture.phone.run.assert_not_called()

    def test_truncated_payload_acquires_fresh_png(self):
        capture = self.backend(struct.pack("!4scI",b"ATK1",b"D",100)+b"short")
        self.assertEqual(capture.read(),b"fresh png")
        self.assertFalse(capture.repeat)
        self.assertEqual(capture.acquisition,"screencap-png")

    def test_invalid_or_oversize_header_never_reuses_old_image(self):
        for magic, acquisition, length in [(b"BAD!",b"D",0),(b"ATK1",b"X",0),
                                           (b"ATK1",b"D",33*1024*1024)]:
            with self.subTest(magic=magic,acquisition=acquisition,length=length):
                capture = self.backend(struct.pack("!4scI",magic,acquisition,length))
                self.assertEqual(capture.read(),b"fresh png")
                self.assertFalse(capture.repeat)

    def test_first_frame_cannot_be_a_repeat(self):
        capture = self.backend(struct.pack("!4scI",b"ATK1",b"D",0))
        capture.previous = None
        self.assertEqual(capture.read(),b"fresh png")
        self.assertFalse(capture.repeat)


if __name__ == "__main__":
    unittest.main()
