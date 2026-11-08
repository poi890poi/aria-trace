"""Batched input keeps intervention priority and never retries uncertain input."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from swiftest import DeviceError
from swiftest.guarded import checked_tap
from examples.genshin.profile import PACKAGE, Phone


class GuardedTests(unittest.TestCase):
    def phone(self, response=b"SENT\n"):
        device=Mock(package=PACKAGE)
        device.foreground.return_value=PACKAGE
        device.run.return_value=response
        session=SimpleNamespace(device=device,popups={},busy_states=[],
                                stats={"input_events":0},events=[],touch=Mock())
        return Phone(session),session

    def test_outside_capture_never_contacts_device(self):
        phone,session=self.phone()
        with self.assertRaises(RuntimeError):
            phone.tap(2400,5,2400,1080)
        session.device.run.assert_not_called()

    def test_focus_or_geometry_rejection_is_not_a_touch(self):
        for status in (b"FOCUS_CHANGED\n",b"DISPLAY_CHANGED\n",b"STATE_ERROR\n"):
            phone,session=self.phone(status)
            with self.assertRaises(DeviceError):
                phone.tap(100,100,2400,1080)
            self.assertEqual(session.stats["input_events"],0)
            session.device.shell.assert_not_called()

    def test_lost_acknowledgement_does_not_fall_back_or_repeat(self):
        phone,session=self.phone()
        session.device.run.side_effect=DeviceError("unknown command outcome")
        with self.assertRaises(DeviceError):
            phone.tap(100,100,2400,1080)
        session.device.run.assert_called_once()
        session.device.shell.assert_not_called()

    def test_overlay_recovery_never_sends_the_underlying_target(self):
        phone,session=self.phone(b"OVERLAY\n")
        self.assertFalse(phone.tap(100,100,2400,1080))
        self.assertEqual(session.stats["input_events"],0)
        self.assertEqual(session.device.shell.call_args_list[0].args,("input","tap","1200","540"))
        self.assertEqual(session.device.shell.call_args_list[1].args[:2],("input","swipe"))
        with self.assertRaises(RuntimeError):
            phone.tap(100,100,2400,1080)
        self.assertEqual(session.device.shell.call_count,2)

    def test_missing_tool_falls_back_only_before_input(self):
        phone,session=self.phone(b"UNSUPPORTED\n")
        session.device.run.side_effect=[b"UNSUPPORTED\n",b""]
        self.assertTrue(phone.tap(100,100,2400,1080))
        session.touch.assert_called_once()

    def test_invalid_numbers_are_rejected_before_contact(self):
        phone,session=self.phone()
        with self.assertRaises(ValueError):
            checked_tap(session.device,float("nan"),1,2400,1080,PACKAGE,"blocked")
        session.device.run.assert_not_called()

    def test_registered_session_rules_keep_the_shared_lookup_path(self):
        phone,session=self.phone(b"")
        session.popups={"notice":object()}
        self.assertTrue(phone.tap(100,100,2400,1080))
        session.touch.assert_called_once()

    def test_confirmed_input_updates_the_session_once(self):
        phone,session=self.phone()
        self.assertTrue(phone.tap(100,100,2400,1080))
        self.assertEqual(session.stats["input_events"],1)
        self.assertEqual(session.events,[{"touch":[100,100]}])


if __name__=="__main__":
    unittest.main()
