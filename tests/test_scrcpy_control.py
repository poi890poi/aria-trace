import struct
import threading
import unittest

from acquisition.android_zigzag import AndroidZigzagInputSource, ZigzagTouchPlan
from acquisition.scrcpy_control import (
    GENERIC_FINGER_POINTER_ID,
    ScrcpyTouchController,
    serialize_touch_event,
)


class FakeSocket:
    def __init__(self):
        self.messages = []

    def sendall(self, value):
        self.messages.append(value)


class FakeController:
    def __init__(self):
        self.actions = []
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return self

    def inject_touch(self, action, point_xy):
        self.actions.append((action, list(point_xy)))
        return 1

    def close(self):
        self.closed = True

    def describe(self):
        return {"type": type(self).__name__}


class ScrcpyControlTests(unittest.TestCase):
    def test_touch_message_matches_scrcpy_4_1_wire_layout(self):
        message = serialize_touch_event("MOVE", [123, 456], [2400, 1080])
        self.assertEqual(len(message), 32)
        values = struct.unpack(">BBQiiHHHII", message)
        self.assertEqual(
            values,
            (
                2,
                2,
                GENERIC_FINGER_POINTER_ID,
                123,
                456,
                2400,
                1080,
                0xFFFF,
                0,
                0,
            ),
        )
        up = struct.unpack(">BBQiiHHHII", serialize_touch_event("UP", [1, 2], [3, 4]))
        self.assertEqual(up[1], 1)
        self.assertEqual(up[7], 0)

    def test_touch_message_rejects_points_outside_display(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            serialize_touch_event("DOWN", [2400, 20], [2400, 1080])

    def test_controller_sends_serialized_messages_over_existing_socket(self):
        controller = ScrcpyTouchController(
            "adb.exe", "scrcpy-server", "serial", [2400, 1080]
        )
        fake_socket = FakeSocket()
        controller._socket = fake_socket
        controller.inject_touch("DOWN", [100, 200])
        controller.inject_touch("UP", [100, 200])
        self.assertEqual(len(fake_socket.messages), 2)
        self.assertEqual(controller.describe()["events_sent"], 2)

    def test_zigzag_uses_reusable_controller_and_reports_completion(self):
        controller = FakeController()
        plan = ZigzagTouchPlan(
            start_xy=[90, 50],
            end_x=10,
            vertical_amplitude_px=10,
            move_count=4,
            step_seconds=0.03,
            settle_seconds=0.0,
        )
        source = AndroidZigzagInputSource(
            "adb.exe", "serial", plan, controller=controller
        )
        packets = []
        source.start(packets.append)
        self.assertTrue(source.wait_completed(2.0))
        source.stop()

        self.assertTrue(source.completed)
        self.assertEqual(source.events_issued, source.expected_event_count)
        self.assertEqual([item[0] for item in controller.actions], [
            "DOWN", "MOVE", "MOVE", "MOVE", "MOVE", "UP"
        ])
        self.assertEqual(len(packets), 6)
        self.assertTrue(controller.opened)
        self.assertTrue(controller.closed)


if __name__ == "__main__":
    unittest.main()
