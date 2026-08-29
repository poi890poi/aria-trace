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

    def test_zigzag_interpolates_sustained_motion_along_every_leg(self):
        plan = ZigzagTouchPlan(
            start_xy=[90, 50],
            end_x=10,
            vertical_amplitude_px=20,
            move_count=4,
            step_seconds=0.12,
            sample_hz=30.0,
            settle_seconds=0.0,
        )
        moves = plan.sampled_moves()
        self.assertEqual(16, len(moves))
        self.assertEqual(4, plan.as_dict()["samples_per_leg"])
        waypoints = plan.points()
        self.assertEqual(waypoints[0], moves[3]["point_xy"])
        self.assertEqual(waypoints[1], moves[7]["point_xy"])
        self.assertEqual(waypoints[2], moves[11]["point_xy"])
        self.assertEqual(waypoints[3], moves[15]["point_xy"])
        first_leg_y = [item["point_xy"][1] for item in moves[:4]]
        second_leg_y = [item["point_xy"][1] for item in moves[4:8]]
        self.assertEqual(first_leg_y, sorted(first_leg_y, reverse=True))
        self.assertEqual(second_leg_y, sorted(second_leg_y))
        self.assertTrue(all(item["leg_index"] == 0 for item in moves[:4]))


if __name__ == "__main__":
    unittest.main()
