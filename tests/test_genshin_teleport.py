"""Behavior checks for the standalone no-VLM Android prototype."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from examples.genshin.profile import (
    classify,
    matching_line,
    Phone,
    Runner,
    has_world_hud,
    is_loading_screen,
    ocr_image,
)
from swiftest.workflow import TouchSequence, Rule, download_delay, point_for


def observation(*entries):
    return {
        "width": 2400,
        "height": 1080,
        "lines": [
            {
                "text": text,
                "words": [{"text": text, "x": x, "y": y, "width": 100, "height": 30}],
            }
            for text, x, y in entries
        ],
    }


class GenshinObservationTests(unittest.TestCase):
    def test_slow_teleport_sends_one_tap_and_requires_consecutive_arrival(self):
        seq = TouchSequence(
            [{"element": "teleport"}],
            {"teleport": Rule("selected", "world", require_loading=True)},
            120,
        )
        tap = Mock()

        def frame(state, second, loading=False):
            return seq.advance(state, loading, lambda step: (10, 20), tap, now=second)

        for second in (0, 5, 20):
            self.assertFalse(frame("selected", second))
        self.assertFalse(frame("world", 22))
        self.assertFalse(frame("unknown", 27, True))
        self.assertFalse(frame("world", 30))
        self.assertFalse(frame("unknown", 31))
        self.assertFalse(frame("world", 32))
        self.assertTrue(frame("world", 33))
        tap.assert_called_once()

    def test_download_poll_uses_reported_remaining_size(self):
        self.assertEqual(download_delay(["10000 KB/s（200 MB / 11200 MB）"], 2), 60)
        self.assertEqual(download_delay(["no progress value"], 6), 6)
        self.assertEqual(download_delay(["10000 KB/s (11199 MB / 11200 MB)"], 2), 1)

    def test_duplicate_template_blocks_touch_until_unique(self):
        import cv2, numpy as np

        sample = np.random.RandomState(13).randint(0, 255, (14, 16, 3)).astype(np.uint8)
        screen = np.zeros((90, 160, 3), dtype=np.uint8)
        screen[30:44, 40:56] = sample
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "target.png")
            cv2.imwrite(path, sample)
            self.assertEqual(point_for({"template": path}, screen, None), (48, 37))
            screen[10:24, 80:96] = sample
            self.assertIsNone(point_for({"template": path}, screen, None))

    def test_network_error_overrides_other_screen_evidence(self):
        obs = observation(("網 路 錯 誤", 900, 200), ("錯誤碼：4201", 900, 500))
        self.assertEqual(classify(obs, world_hud=True), "network_error")

    def test_chinese_ocr_character_spacing_and_bounds(self):
        obs = observation(("傳 送", 2100, 990))
        result = matching_line(obs, ("傳送",), (0.65, 0.65, 1, 1), exact=True)
        self.assertEqual((result["x"], result["y"]), (2150, 1005))

    def test_teleport_word_in_description_is_not_a_button(self):
        obs = observation(
            ("傳送錨點", 1900, 200), ("使用傳送錨點前往目的地", 1900, 800)
        )
        self.assertNotEqual(classify(obs), "waypoint_selected")

    def test_teleport_requires_waypoint_context_and_button_region(self):
        for obs in (
            observation(("傳送", 2100, 990)),
            observation(("傳送錨點", 1900, 200), ("傳送", 200, 500)),
        ):
            self.assertNotEqual(classify(obs), "waypoint_selected")
        good = observation(("傳送錨點", 1900, 200), ("傳送", 2100, 990))
        self.assertEqual(classify(good), "waypoint_selected")

    def test_download_does_not_count_as_ready(self):
        self.assertEqual(
            classify(observation(("正在下載資源 42%", 900, 800))), "downloading"
        )

    def test_start_requires_observed_text_in_lower_screen(self):
        self.assertEqual(classify(observation(("點 擊 進 入", 1100, 900))), "start")
        self.assertEqual(
            classify(observation(("輕 按 螢 幕 進 入", 1100, 990))), "start"
        )
        self.assertEqual(classify(observation(("點 擊 進 入", 1100, 100))), "unknown")

    def test_current_start_screen_requires_server_context(self):
        for text in ("開始遊戲", "開始游獻", "开始游戏"):
            self.assertEqual(classify(observation((text, 1100, 490),
                                                  ("Asia", 1150, 880))), "start")
            self.assertEqual(classify(observation((text, 1100, 490))), "unknown")
            self.assertEqual(classify(observation((text, 100, 490),
                                                  ("Asia", 1150, 880))), "unknown")

    def test_ocr_omitted_entry_glyph_requires_release_footer(self):
        prompt = ("輕按幕進入", 1120, 990)
        self.assertEqual(classify(observation(prompt, ("OSRELAndroid7.2.0", 180, 1038))), "start")
        self.assertEqual(classify(observation(prompt)), "unknown")
        self.assertEqual(classify(observation(("輕按幕進入", 1120, 400),
                                              ("OSRELAndroid7.2.0", 180, 1038))), "unknown")

    def test_region_map_uses_exploration_label_in_footer(self):
        self.assertEqual(classify(observation(("探 索 度 1%", 2000, 1030))), "map")
        self.assertEqual(classify(observation(("探 索 度 1%", 400, 300))), "unknown")

    def test_split_hp_bar_on_green_terrain(self):
        import numpy as np

        frame = np.full((1080, 2400, 3), (61, 115, 42), dtype=np.uint8)
        obs = observation(("UID: 123456789", 2150, 960))
        for left in (1010, 1234):
            frame[985:999, left : left + 155] = (50, 210, 80)
        self.assertTrue(has_world_hud(frame, obs))
        frame[985:999, 1234:1389] = (61, 115, 42)
        self.assertFalse(has_world_hud(frame, obs))

    def test_game_booster_dismissal_requires_fresh_target(self):
        phone = Phone(Mock())
        phone.session.device.foreground.return_value = "com.miHoYo.GenshinImpact"
        phone.session.device.run.return_value = (
            b"name='abc GameBooster Lock Screen', inputConfig=NOT_FOCUSABLE"
        )
        phone.shell = Mock(side_effect=["", ""])
        self.assertFalse(phone.tap(200, 155, 2400, 1080))
        self.assertNotIn(
            ("input", "tap", "200", "155"),
            [call[0] for call in phone.shell.call_args_list],
        )
        self.assertEqual(phone.shell.call_args_list[-1][0][:2], ("input", "swipe"))

    def test_focus_loss_prevents_touch(self):
        phone = Phone(Mock())
        phone.session.device.foreground.return_value = "com.other.app"
        with self.assertRaisesRegex(RuntimeError, "not focused"):
            phone.tap(100, 100, 2400, 1080)
        phone.shell.assert_not_called()

    def test_out_of_bounds_never_contacts_device(self):
        phone = Phone(Mock())
        phone.shell = Mock()
        with self.assertRaisesRegex(RuntimeError, "out-of-bounds"):
            phone.tap(2400, 100, 2400, 1080)
        phone.shell.assert_not_called()

    def test_uid_alone_on_loading_screen_is_not_world(self):
        import numpy as np

        frame = np.full((1080, 2400, 3), 245, dtype=np.uint8)
        obs = observation(("UID: 123456789", 2150, 1040))
        self.assertFalse(has_world_hud(frame, obs))
        self.assertTrue(is_loading_screen(frame))
        frame[960:966, 1000:1400] = (50, 210, 80)
        self.assertTrue(has_world_hud(frame, obs))

    def test_offline_phone_stops_without_dismissing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=directory, timeout=20, inspect=False, poll=0.2
            )
            phone = Mock()
            phone.shell.return_value = "Active default network: none"
            runner = Runner(phone, args)
            runner.observe = Mock(
                return_value=("network_error", observation(("確認", 1100, 800)), None)
            )
            runner.tap = Mock()
            with self.assertRaisesRegex(RuntimeError, "working internet"):
                runner.run()
            runner.tap.assert_not_called()

    def test_dimmed_ocr_preprocessing_preserves_original(self):
        import numpy as np

        frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        corrected = ocr_image(frame)
        self.assertGreater(float(corrected.mean()), 150)
        self.assertEqual(int(frame.max()), 20)
        self.assertEqual(corrected.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
