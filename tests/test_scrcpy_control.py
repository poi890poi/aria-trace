import struct
import threading
import time
import unittest
from unittest import mock

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
        self.action_times = []
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return self

    def inject_touch(self, action, point_xy):
        self.actions.append((action, list(point_xy)))
        self.action_times.append((action, time.perf_counter()))
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

    def test_zigzag_uses_one_android_swipe_command_per_stroke(self):
        controller = FakeController()
        plan = ZigzagTouchPlan(
            start_xy=[90, 50],
            end_x=80,
            vertical_amplitude_px=10,
            move_count=4,
            step_seconds=0.03,
            endpoint_hold_seconds=0.0,
            settle_seconds=0.0,
            reset_seconds=0.0,
        )
        source = AndroidZigzagInputSource(
            "adb.exe", "serial", plan, controller=controller
        )
        packets = []
        with mock.patch(
            "rig_runtime.adapters.android.zigzag.subprocess.check_call"
        ) as command:
            source.start(packets.append)
            self.assertTrue(source.wait_completed(2.0))
            source.stop()

        self.assertTrue(source.completed)
        self.assertEqual(source.events_issued, source.expected_event_count)
        self.assertEqual(4, command.call_count)
        for call in command.call_args_list:
            arguments = call[0][0]
            self.assertEqual("swipe", arguments[6])
            self.assertEqual("30", arguments[-1])
        self.assertEqual([packet.payload["action"] for packet in packets], [
            "SWIPE", "SWIPE", "SWIPE", "SWIPE",
        ])
        self.assertEqual(
            "adb_shell_input_touchscreen_swipe",
            source.describe()["transport"],
        )
        self.assertFalse(controller.opened)
        self.assertFalse(controller.closed)

    def test_zigzag_holds_finger_down_at_endpoint_before_one_logical_swipe(self):
        controller = FakeController()
        plan = ZigzagTouchPlan(
            start_xy=[90, 50],
            end_x=80,
            vertical_amplitude_px=10,
            move_count=4,
            step_seconds=0.03,
            endpoint_hold_seconds=0.04,
            settle_seconds=0.0,
            reset_seconds=0.0,
            move_sample_hz=10.0,
        )
        source = AndroidZigzagInputSource(
            "adb.exe", "serial", plan, controller=controller
        )
        packets = []
        source.start(packets.append)
        self.assertTrue(source.wait_completed(2.0))
        source.stop()

        self.assertTrue(source.completed)
        self.assertTrue(controller.opened)
        self.assertTrue(controller.closed)
        self.assertEqual(4, len(packets))
        self.assertEqual(["SWIPE"] * 4, [packet.payload["action"] for packet in packets])
        self.assertEqual(4, source.events_issued)
        self.assertEqual(4, source.expected_event_count)
        for stroke_index in range(4):
            actions = controller.actions[stroke_index * 4 : (stroke_index + 1) * 4]
            self.assertEqual(["DOWN", "MOVE", "MOVE", "UP"], [row[0] for row in actions])
            times = controller.action_times[stroke_index * 4 : (stroke_index + 1) * 4]
            self.assertGreaterEqual(times[-1][1] - times[-2][1], 0.03)
            payload = packets[stroke_index].payload
            self.assertEqual(70, payload["duration_ms"])
            self.assertEqual(30, payload["travel_duration_ms"])
            self.assertEqual(40, payload["endpoint_hold_duration_ms"])
            self.assertGreaterEqual(payload["actual_endpoint_hold_duration_ms"], 30.0)

    def test_adb_only_held_swipe_uses_motion_events_and_one_summary_packet(self):
        plan = ZigzagTouchPlan(
            start_xy=[90, 50],
            end_x=80,
            vertical_amplitude_px=10,
            move_count=4,
            step_seconds=0.03,
            endpoint_hold_seconds=0.01,
            settle_seconds=0.0,
            reset_seconds=0.0,
            move_sample_hz=10.0,
        )
        source = AndroidZigzagInputSource("adb.exe", "serial", plan)
        packets = []
        with mock.patch(
            "rig_runtime.adapters.android.zigzag.subprocess.check_call"
        ) as command:
            source.start(packets.append)
            self.assertTrue(source.wait_completed(2.0))
            source.stop()

        self.assertTrue(source.completed)
        self.assertEqual(4, len(packets))
        self.assertEqual(16, command.call_count)
        actions = [call[0][0][7] for call in command.call_args_list]
        self.assertEqual(
            ["DOWN", "MOVE", "MOVE", "UP"] * 4,
            actions,
        )
        self.assertNotIn("swipe", [item for call in command.call_args_list for item in call[0][0]])

    def test_default_zigzag_splits_motion_into_twelve_balanced_strokes(self):
        plan = ZigzagTouchPlan(
            start_xy=[1320, 540],
            end_x=1077,
            vertical_amplitude_px=486,
        )
        strokes = plan.strokes()
        self.assertEqual(len(strokes), 12)
        self.assertEqual(
            [stroke["direction"] for stroke in strokes],
            [
                "up", "up", "down", "down", "down", "down",
                "up", "up", "up", "up", "down", "down",
            ],
        )
        self.assertEqual(
            sum(stroke["direction"] == "up" for stroke in strokes), 6
        )
        self.assertEqual(
            sum(stroke["direction"] == "down" for stroke in strokes), 6
        )
        for stroke in strokes:
            delta_x = stroke["end_xy"][0] - stroke["start_xy"][0]
            delta_y = stroke["end_xy"][1] - stroke["start_xy"][1]
            self.assertEqual(abs(delta_x), 243)
            self.assertEqual(abs(delta_y), 486)

    def test_legacy_progressive_points_remain_available_for_micro_movements(self):
        plan = ZigzagTouchPlan(
            start_xy=[1320, 540],
            end_x=1077,
            vertical_amplitude_px=486,
            move_count=12,
            step_seconds=0.35,
            endpoint_hold_seconds=0.0,
            move_sample_hz=22.0,
        )
        stroke = plan.sampled_strokes()[0]
        moves = stroke["move_points_xy"]

        self.assertEqual(plan.move_samples_per_stroke, 8)
        self.assertEqual(len(moves), 8)
        self.assertNotEqual(moves[0], stroke["end_xy"])
        self.assertEqual(moves[-1], stroke["end_xy"])
        self.assertEqual(
            [point[0] for point in moves],
            sorted((point[0] for point in moves), reverse=True),
        )
        self.assertEqual(
            [point[1] for point in moves],
            sorted((point[1] for point in moves), reverse=True),
        )
        for point in moves:
            delta_x = point[0] - stroke["start_xy"][0]
            delta_y = point[1] - stroke["start_xy"][1]
            self.assertLessEqual(abs(abs(delta_y) - 2 * abs(delta_x)), 2)

        self.assertEqual(plan.move_samples_per_stroke, 8)
        self.assertEqual(len(plan.strokes()), 12)
        self.assertEqual(12, len(plan.strokes()))

if __name__ == "__main__":
    unittest.main()
