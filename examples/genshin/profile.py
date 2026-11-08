"""Genshin screen recognition and action policies for the example intent.

Run from the repository root with Python and OpenCV available. Captures and logs
are kept under ignored artifacts/. Coordinates are always captured-screen pixels.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor


from swiftest import Pixel
from swiftest.guarded import checked_tap
from swiftest.workflow import (
    Rule,
    TouchSequence,
    download_delay,
    interval,
    load_intent,
    point_for,
)

DEFAULT_INTENT = Path(__file__).with_name("teleport.json")
PACKAGE = "com.miHoYo.GenshinImpact"
ENTRY_LABELS = (
    "輕按螢幕進入",
    "轻触屏幕进入",
    "點擊進入",
    "点击进入",
    "點擊開始",
    "点击开始",
    "startgame",
    "taptobegin",
    "taptoenter",
)
RULES = {
    "minimap": Rule("world", "map", retry_safe=True),
    "unlocked_waypoint": Rule("map", "waypoint_selected"),
    "teleport": Rule("waypoint_selected", "world", require_loading=True),
}


def compact(text):
    return (
        re.sub(r"\s+", "", text)
        .casefold()
        .translate(str.maketrans("傳送錨點", "传送锚点"))
    )


def matching_line(observation, labels, region=None, exact=False):
    """Match a complete OCR line and return its observed bounds, never guessed text."""
    width, height = observation["width"], observation["height"]
    for line in observation.get("lines", []):
        value = compact(line["text"])
        if not any(
            value == compact(label) if exact else compact(label) in value
            for label in labels
        ):
            continue
        words = line.get("words", [])
        if not words:
            continue
        left, top = min(w["x"] for w in words), min(w["y"] for w in words)
        right = max(w["x"] + w["width"] for w in words)
        bottom = max(w["y"] + w["height"] for w in words)
        x, y = (left + right) / 2, (top + bottom) / 2
        if region and not (
            region[0] <= x / width <= region[2] and region[1] <= y / height <= region[3]
        ):
            continue
        return {"text": line["text"], "x": x, "y": y}
    return None


def classify(observation, world_hud=False):
    text = compact(" ".join(line["text"] for line in observation.get("lines", [])))
    if any(
        s in text
        for s in ("4201", "網路錯誤", "网络错误", "networkerror", "無法連接網路")
    ):
        return "network_error"
    if any(
        s in text for s in ("輸入密碼", "输入密码", "enterpassword", "驗證碼", "验证码")
    ):
        return "login_required"
    if any(
        s in text
        for s in ("下載", "下载", "downloading", "檢查更新", "检查更新", "verifying")
    ):
        return "downloading"
    shader_label = "色器" in text and ("編" in text or "编" in text)
    if shader_label or any(
        s in text for s in ("著色器", "着色器", "compilingshaders", "解壓縮", "解压缩")
    ):
        return "preparing"
    if entry_button(observation):
        return "start"
    teleport = matching_line(
        observation, ("傳送", "传送", "teleport"), region=(0.65, 0.65, 1, 1), exact=True
    )
    waypoint = matching_line(
        observation,
        (
            "傳送錨點",
            "传送锚点",
            "传送點",
            "傳送點",
            "传送点",
            "teleportwaypoint",
            "七天神像",
            "statueofthe",
        ),
        region=(0.6, 0, 1, 0.40),
    )
    if teleport and waypoint:
        return "waypoint_selected"
    if world_hud:
        return "world"
    if matching_line(
        observation,
        ("提瓦特", "teyvat", "探索度", "explorationprogress"),
        region=(0.65, 0.65, 1, 1),
    ):
        return "map"
    return "unknown"


def entry_button(observation):
    button = matching_line(observation, ENTRY_LABELS, region=(0.15, 0.45, 0.85, 1))
    if button:
        # Logical identity, not raw OCR spelling, owns retry timing.
        launch = any(compact(label) in compact(button["text"])
                     for label in ("點擊開始", "点击开始", "startgame"))
        return dict(button, stage="launch" if launch else "enter")
    # Current Chinese launch UI uses 開始遊戲. OCR can misread its final two
    # characters against the animated background; require the server selector
    # as independent context before accepting the stable 開始 prefix.
    server = matching_line(observation, ("Asia", "Europe", "America", "TW, HK, MO"),
                           region=(0.30, 0.70, 0.70, 0.91), exact=True)
    if server:
        button = matching_line(observation, ("開始", "开始"),
                               region=(0.40, 0.40, 0.60, 0.58))
        return dict(button, stage="launch") if button else None
    version = matching_line(observation, ("OSREL",), region=(0, 0.94, 0.5, 1))
    if version:
        # RapidOCR can omit 螢 in the small footer prompt. Keep this observed
        # alias exact and require the game's release footer as context.
        button = matching_line(observation, ("輕按幕進入", "轻按幕进入"),
                               region=(0.35, 0.85, 0.65, 0.98), exact=True)
        return dict(button, stage="enter") if button else None
    return None


def has_world_hud(image, observation):
    """Require both UID text and the distinctive bottom-center green HP bar."""
    import cv2
    import numpy as np

    if not matching_line(observation, ("uid",), region=(0.70, 0.85, 1, 1)):
        return False
    # Portable OCR can also read the numerical health label on a partially filled bar.
    has_level = matching_line(observation, ("lv.",), region=(0.3, 0.82, 0.60, 0.98))
    for line in observation.get("lines", []):
        health = re.fullmatch(r"(\d+)\s*/\s*(\d+)", line["text"])
        if (
            has_level
            and health
            and 0 < int(health[1]) <= int(health[2])
            and matching_line(
                observation,
                (line["text"],),
                region=(0.37, 0.82, 0.65, 0.98),
                exact=True,
            )
        ):
            return True
    h, w = image.shape[:2]
    crop = image[int(h * 0.82) : int(h * 0.97), int(w * 0.32) : int(w * 0.68)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 70, 80]), np.array([85, 255, 255]))
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if any(
        bw > w * 0.07 and bw > bh * 12 and bh >= 2
        for _, _, bw, bh in (cv2.boundingRect(c) for c in contours)
    ):
        return True
    # On the phone the health readout divides the bright bar into two halves.
    bright = cv2.inRange(hsv, np.array([28, 50, 180]), np.array([85, 255, 255]))
    contours = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    segments = sorted(cv2.boundingRect(c) for c in contours)
    segments = [
        (x, y, bw, bh)
        for x, y, bw, bh in segments
        if 0.04 * w < bw < 0.10 * w and bw > 7 * bh and bh >= 2
    ]
    center = crop.shape[1] / 2
    for left, right in zip(segments, segments[1:]):
        x, y, bw, bh = left
        rx, ry, rw, rh = right
        if (
            x < center < rx + rw
            and abs(y - ry) <= h * 0.006
            and 0 < rx - x - bw < w * 0.045
            and abs(bw - rw) < w * 0.025
            and abs(bh - rh) <= h * 0.007
        ):
            return True
    return False


def waypoint_candidates(image):
    """Propose blue map markers; OCR of the selected panel must authorize teleport."""
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 90, 110]), np.array([115, 255, 255]))
    mask[: int(h * 0.12)] = 0
    mask[int(h * 0.88) :] = 0
    mask[:, : int(w * 0.12)] = 0
    mask[:, int(w * 0.70) :] = 0
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    candidates = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (
            w * 0.004 <= bw <= w * 0.025
            and h * 0.008 <= bh <= h * 0.065
            and 0.3 <= bw / bh <= 1.6
        ):
            continue
        # An unlocked map marker has a light outline around its blue center.
        pad = max(3, int(h * 0.007))
        area = hsv[max(0, y - pad) : y + bh + pad, max(0, x - pad) : x + bw + pad]
        light = (area[:, :, 1] < 80) & (area[:, :, 2] > 180)
        if float(light.mean()) < 0.10:
            continue
        point = (x + bw / 2, y + bh / 2)
        # The observed unlocked marker has a mirrored white outline and white lower tip.
        # Blue streets and coastlines also pass the color test but lack this shape.
        cx, cy = round(point[0]), round(point[1])
        px, py = round(h * 0.0204), round(h * 0.026)
        patch = hsv[cy - py : cy + py + 1, cx - px : cx + px + 1]
        white = (patch[:, :, 1] < 65) & (patch[:, :, 2] > 190)
        reflected = white[:, ::-1]
        symmetry = np.count_nonzero(white & reflected) / max(
            1, np.count_nonzero(white | reflected)
        )
        tip = white[
            py + round(h * 0.008) : py + round(h * 0.021),
            px - round(h * 0.005) : px + round(h * 0.005) + 1,
        ]
        if symmetry < 0.45 or not tip.size or tip.mean() < 0.55:
            continue
        if all(
            (point[0] - a) ** 2 + (point[1] - b) ** 2 > (h * 0.03) ** 2
            for a, b in candidates
        ):
            candidates.append(point)
    return sorted(
        candidates, key=lambda p: (p[0] - w * 0.4) ** 2 + (p[1] - h * 0.5) ** 2
    )[:12]


def is_loading_screen(image):
    """Conservative visual loading cue; an OCR miss alone is not a transition."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    neutral = hsv[:, :, 1] < 35
    bright = neutral & (hsv[:, :, 2] > 220)
    dark = hsv[:, :, 2] < 35
    return float(np.mean(bright)) > 0.80 or float(np.mean(dark)) > 0.90


def brightness_percentile(image):
    """Exact 99.5th percentile for 8-bit frames, without partitioning the pixels."""
    import cv2
    import numpy as np

    # Float histograms count integers exactly only up to 2**24. Retain the
    # original operation for other image types and unusually large frames.
    if image.dtype != np.uint8 or not 0 < image.size <= 2**24:
        return float(np.percentile(image, 99.5))
    counts = cv2.calcHist([image.reshape(-1, 1)], [0], None, [256], [0, 256]).ravel()
    cumulative = counts.cumsum(dtype=np.float64)
    rank = .995 * (image.size - 1)
    low, high = np.searchsorted(cumulative, [int(rank), int(np.ceil(rank))], side="right")
    fraction = rank - int(rank)
    return float(high - (high - low) * (1 - fraction) if fraction >= .5
                 else low + (high - low) * fraction)


def ocr_image(image):
    """Recover OCR contrast when Samsung's idle overlay dims the captured game."""
    import cv2

    level = brightness_percentile(image)
    if 3 < level < 65:
        return cv2.convertScaleAbs(image, alpha=min(16.0, 230.0 / level))
    return image


class Phone:
    def __init__(self, session):
        self.session = session
        self.shell = session.device.shell
        self.protection_released_at = None

    def launch(self):
        self.session.open(PACKAGE)

    def in_focus(self):
        return self.session.device.foreground() == PACKAGE

    def tap(self, x, y, width, height):
        if not (0 <= x < width and 0 <= y < height):
            raise RuntimeError("Refusing an out-of-bounds tap.")
        if not self.session.popups and not self.session.busy_states:
            sent = checked_tap(self.session.device, x, y, width, height, PACKAGE,
                               "GameBooster Lock Screen")
            if sent is True:
                self.session.stats["input_events"] += 1
                self.session.events.append({"touch": [round(x), round(y)]})
                return True
            if sent is False:
                return self.release_protection(width, height)
            # An unavailable tool is reported before input. Registered Session
            # rules and unsupported devices retain the original shared path.
        return self._legacy_tap(x, y, width, height)

    def release_protection(self, width, height):
        if (self.protection_released_at is not None
                and time.monotonic() - self.protection_released_at < 10):
            raise RuntimeError("Samsung Game Booster touch protection still covers the game.")
        if not self.in_focus():
            raise RuntimeError("Focus changed before overlay recovery")
        cx, cy = str(width // 2), str(height // 2)
        self.shell("input", "tap", cx, cy)
        self.shell("input", "swipe", cx, cy, str(int(width * .78)), cy, "450")
        self.protection_released_at = time.monotonic()
        return False

    def _legacy_tap(self, x, y, width, height):
        if not self.in_focus():
            raise RuntimeError("Genshin is not focused; no tap sent.")
        overlay = self.session.device.run(
            "shell", "dumpsys input | grep 'GameBooster Lock Screen' | head -1"
        ).decode()
        if (
            "name='" in overlay
            and "GameBooster Lock Screen" in overlay
            and "NOT_VISIBLE" not in overlay
            and "NOT_TOUCHABLE" not in overlay
        ):
            return self.release_protection(width, height)
        self.session.touch(Pixel(round(x), round(y)))
        return True


class Runner:
    def __init__(self, phone, args, session=None):
        self.phone, self.args = phone, args
        self.session = session if session is not None else phone.session
        self.directory = Path(args.output).resolve() / (
            time.strftime("%Y%m%d-%H%M%S") + "-{}".format(time.time_ns() % 1000000000)
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.actions = 0
        self.last_observation = 0
        self.dimensions = None
        self.cached = None
        self.frame = None
        self.ocr_calls = 0
        self.capture_seconds = 0.0
        self.ocr_seconds = 0.0
        self._writer = None
        self._writes = []

    def log(self, event, **details):
        record = dict(event=event, time=time.time(), **details)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # ASCII console output also works with legacy Windows code pages.
        print(json.dumps(record, ensure_ascii=True), flush=True)

    def observe(self, waiting_for_loading=False):
        self.index += 1
        requested = time.monotonic()
        self.frame = image = self.session.frame()
        self.capture_seconds += time.monotonic() - requested
        self.last_observation = requested
        # The loading screen can disappear during a full OCR pass. Check its pixels first.
        if waiting_for_loading:
            state = (
                "loading"
                if is_loading_screen(ocr_image(image))
                else "transition_pending"
            )
            self.cached = None
            self.log(
                "observe",
                state=state,
                frame=self.index,
                bytes=self.session.report()["capture_bytes"],
            )
            return state, None, image
        if self.session.capture_repeated and self.cached:
            return self.cached
        prepared = ocr_image(image)
        started = time.perf_counter()
        lines = []
        for value, box in self.session.recognize(prepared):
            lines.append(
                {
                    "text": value,
                    "words": [
                        {
                            "text": value,
                            "x": box.x,
                            "y": box.y,
                            "width": box.w,
                            "height": box.h,
                        }
                    ],
                }
            )
        observation = {
            "width": image.shape[1],
            "height": image.shape[0],
            "lines": lines,
        }
        self.ocr_calls += 1
        self.ocr_seconds += time.perf_counter() - started
        self.dimensions = (observation["width"], observation["height"])
        state = classify(observation, has_world_hud(prepared, observation))
        (self.directory / "latest.ocr.json").write_text(
            json.dumps(observation, ensure_ascii=False), encoding="utf-8"
        )
        self.log(
            "observe",
            state=state,
            frame=self.index,
            bytes=self.session.report()["capture_bytes"],
        )
        self.cached = state, observation, image
        return self.cached

    def tap(self, point, reason):
        if time.monotonic() - self.last_observation > 8:
            raise RuntimeError("Observation is stale; no tap sent.")
        if self.phone.tap(point[0], point[1], *self.dimensions) is False:
            self.log("release_game_touch_protection")
            return False
        self.actions += 1
        self.log("tap", reason=reason, x=point[0], y=point[1])
        return True

    def resolve_target(self, step, obs, image):
        element = step["element"]

        def profile_selector():
            if element == "minimap":
                return obs["width"] * 0.085, obs["height"] * 0.145
            if element == "unlocked_waypoint":
                candidates = waypoint_candidates(image)
                return candidates[0] if candidates else None
            button = matching_line(
                obs, ("傳送", "传送", "teleport"), (0.65, 0.65, 1, 1), exact=True
            )
            return (button["x"], button["y"]) if button else None

        point = point_for(step, image, profile_selector)
        if element == "teleport" and point is not None:
            button = profile_selector()
            if (
                not button
                or abs(point[0] - button[0]) > obs["width"] * 0.055
                or abs(point[1] - button[1]) > obs["height"] * 0.023
            ):
                return None
        return point

    def touch_target(self, point, element):
        import cv2

        name = {
            "minimap": "before",
            "unlocked_waypoint": "map",
            "teleport": "selected",
        }[element]
        path = str(self.directory / (name + ".png"))
        # Captured arrays are replaced, not modified. Retain the exact observed
        # frame while its PNG is written during the following game transition.
        if self._writer is not None:
            self._writes.append(self._writer.submit(cv2.imwrite, path, self.frame))
        elif not cv2.imwrite(path, self.frame):
            raise OSError("Evidence image write failed: " + path)
        return self.tap(point, element)

    def run(self):
        with ThreadPoolExecutor(max_workers=1) as writer:
            self._writer = writer if getattr(self.args, "async_evidence", True) else None
            self._writes = []
            try:
                return self._run()
            finally:
                try:
                    for future in self._writes:
                        if not future.result():
                            raise OSError("Evidence image write failed")
                finally:
                    self._writer = None
                    self._writes = []

    def _run(self):
        intent = load_intent(getattr(self.args, "intent", DEFAULT_INTENT), RULES)
        if intent["open"] != PACKAGE:
            raise ValueError("This profile supports Genshin Impact.")
        deadline = time.monotonic() + self.args.timeout
        self.phone.launch()
        for step in intent["touch"]:
            if step["element"] == "unlocked_waypoint" and getattr(
                self.args, "waypoint", None
            ):
                step.pop("template", None)
                step["position"] = self.args.waypoint
        sequence = TouchSequence(
            intent["touch"], RULES, getattr(self.args, "arrival_timeout", 120)
        )
        start_taps = 0
        last_start = {}
        network_retry = False
        previous_state, repeats = None, 0
        while time.monotonic() < deadline:
            awaiting = (
                sequence.sent is not None
                and not sequence.loading
                and RULES[sequence.steps[sequence.index]["element"]].require_loading
            )
            state, obs, image = self.observe(awaiting)
            if self.args.inspect:
                return dict(status="inspected", state=state, actions=0)
            if state == "network_error":
                connected = re.search(
                    r"Active default network:\s*\d+",
                    self.phone.shell("dumpsys", "connectivity"),
                )
                confirm = matching_line(
                    obs,
                    ("確認", "确认", "confirm", "ok"),
                    (0.25, 0.5, 0.75, 0.9),
                    exact=True,
                )
                if connected and confirm and not network_retry:
                    network_retry = self.tap(
                        (confirm["x"], confirm["y"]),
                        "dismiss network error once after Android reports a connection",
                    )
                    time.sleep(self.args.poll)
                    continue
                raise RuntimeError(
                    "Genshin reports a network error. Connect the phone to working internet, then rerun."
                )
            if state == "login_required":
                raise RuntimeError(
                    "Genshin requires login or verification on the phone."
                )
            if state == "start":
                if start_taps >= 3:
                    raise RuntimeError(
                        "Start screen did not advance after three bounded attempts."
                    )
                button = entry_button(obs)
                if time.monotonic() - last_start.get(button["stage"], -float("inf")) >= 15:
                    if self.tap(
                        (button["x"], button["y"]), "recognized game entry text"
                    ):
                        start_taps += 1
                        last_start[button["stage"]] = time.monotonic()
            else:
                loading = state == "loading"
                if sequence.advance(
                    state,
                    loading,
                    lambda step: self.resolve_target(step, obs, image),
                    self.touch_target,
                ):
                    import cv2

                    if not cv2.imwrite(str(self.directory / "arrival.png"), image):
                        raise OSError("Arrival evidence image write failed")
                    status = (
                        "teleport_completed"
                        if intent["touch"][-1]["element"] == "teleport"
                        else "intent_completed"
                    )
                    return dict(status=status, actions=self.actions)
            repeats = repeats + 1 if state == previous_state else 0
            previous_state = state
            delay = interval(state, repeats, sequence.sent is not None, self.args.poll,
                             max_interactive=getattr(self.args, "poll_cap", .1))
            if state == "downloading":
                delay = download_delay(
                    [line["text"] for line in obs.get("lines", [])], delay
                )
            time.sleep(min(delay, max(0, deadline - time.monotonic())))
        raise RuntimeError(
            "Overall timeout reached; no unrecognized dialog was accepted."
        )


def run_intent(session, path):
    """The CLI supplies the same session used for ordinary Python and JSON tests."""
    args = SimpleNamespace(
        intent=path,
        timeout=session.timeout,
        arrival_timeout=120,
        inspect=False,
        poll=2,
        poll_cap=.1,
        output=session.output / "genshin",
    )
    runner = Runner(Phone(session), args, session)
    result = runner.run()
    return dict(result, evidence_directory=str(runner.directory))
