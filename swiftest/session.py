"""The polling, input and confirmation policies shared by every Android test."""

from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
import time

import cv2
import numpy as np

from .capture import Capture
from .device import Android, DeviceError
from .ocr import LocalOCR
from .targets import Absent, Box, Pixel, Position
from .popups import PopupRule, handle_popup
from .evaluation import VisualCache
from .outcome import Outcomes, tracked


class TargetNotFoundError(AssertionError):
    pass


class AmbiguousTargetError(TargetNotFoundError):
    pass


_current = ContextVar("swiftest_session", default=None)


def current():
    session = _current.get()
    if session is None:
        raise RuntimeError("Use a Session context or the swiftest run command.")
    return session


def selector(value):
    if hasattr(value, "matches"):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if any(isinstance(n, float) for n in value) and all(0 <= n <= 1 for n in value):
            return Position(*value)
        if all(isinstance(n, int) and not isinstance(n, bool) for n in value):
            return Pixel(*value)
    raise TypeError("Expected a UI selector, pixel pair or normalized position.")


class Observation:
    """All selectors in one decision share the same lazy pixel and UI-tree queries."""

    def __init__(self, session):
        self.session = session
        self.base = session.base
        self.created = time.monotonic()

    @cached_property
    def image(self):
        return self.session.frame()

    @cached_property
    def screen(self):
        h, w = self.image.shape[:2]
        return Box(0, 0, w, h)

    @cached_property
    def viewport(self):
        if "image" in self.__dict__:
            return self.screen
        size = self.session.device.display_size()
        return Box(0, 0, *size) if size else self.screen

    @cached_property
    def tree(self):
        self.session.stats["tree_reads"] += 1
        return self.session.device.hierarchy()

    @cached_property
    def words(self):
        return self.words_in()

    def words_in(self, scope=None, padding=8):
        image = self.image
        height, width = image.shape[:2]
        if scope is None:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            x0, y0 = max(0, scope.x - padding), max(0, scope.y - padding)
            x1, y1 = min(width, scope.x + scope.w + padding), min(height, scope.y + scope.h + padding)
        if x1 <= x0 or y1 <= y0:
            return []
        crop = image[y0:y1, x0:x1]
        engine = self.session.ocr
        config = tuple(getattr(engine, k, None) for k in ("threads", "max_side", "requested_backend"))
        key = ("ocr", id(engine), config, image.shape, x0, y0, x1, y1)

        def recognize():
            return [(value, Box(box.x + x0, box.y + y0, box.w, box.h, box.score))
                    for value, box in self.session.recognize(crop)]
        return self.session._visual_cache.evaluate(key, crop, recognize)


class Session(Outcomes):
    def __init__(
        self,
        serial=None,
        *,
        device=None,
        capture="auto",
        scrcpy_server=None,
        adb="adb",
        timeout=12,
        base=".",
        ocr=None,
        poll_interval=0.1,
    ):
        self.device = device if device is not None else Android(serial, adb)
        self.base = Path(base).resolve()
        self.timeout = self._timeout(timeout)
        self.poll_interval = self._timeout(poll_interval)
        if self.poll_interval <= 0:
            raise ValueError("Polling interval must be positive.")
        self.ocr = ocr or LocalOCR()
        self._backend = None
        self._mode, self._server = capture, scrcpy_server
        self._image = self._words = self._token = None
        self.stats = {"captures": 0, "tree_reads": 0, "ocr_calls": 0, "input_events": 0}
        self.events = []
        self.closed = False
        self.targets = {}
        self.busy_states = []
        self.popups = {}
        self._visual_cache = VisualCache()
        self.reset_outcome()

    @staticmethod
    def _timeout(value):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Timeout must be finite and nonnegative.")
        return float(value)

    def __enter__(self):
        if self.closed or self._token is not None:
            raise RuntimeError("Session already entered or closed.")
        self._token = _current.set(self)
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._visual_cache.clear()
        try:
            if self._backend:
                self._backend.close()
        finally:
            try:
                self.device.close()
            finally:
                if self._token is not None:
                    _current.reset(self._token)
                    self._token = None

    def _deadline(self, timeout):
        return time.monotonic() + (
            self.timeout if timeout is None else self._timeout(timeout)
        )

    def frame(self):
        if self.closed:
            raise RuntimeError("Session is closed.")
        if self._backend is None:
            if self._mode == "scrcpy":
                from .scrcpy import Scrcpy

                self._backend = Scrcpy(self.device, self._server)
            elif self._mode in ("auto", "adb", "png", "jpeg", "direct", "bitmap"):
                self._backend = Capture(
                    self.device, "jpeg" if self._mode == "adb" else self._mode
                )
            elif hasattr(self._mode, "read"):
                self._backend = self._mode
            else:
                raise ValueError("Unknown capture backend: " + str(self._mode))
        data = self._backend.read()
        self.stats["captures"] += 1
        if not self._backend.repeat or self._image is None:
            self._image = (
                data
                if isinstance(data, np.ndarray)
                else cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            )
            if self._image is None:
                raise DeviceError("Capture returned an invalid image.")
            self._words = None
        return self._image

    def report(self):
        backend = self._backend
        return dict(
            self.stats,
            backend=getattr(backend, "mode", None),
            capture_bytes=getattr(backend, "bytes", 0),
            capture_acquisition=getattr(backend, "acquisition", None),
            setup_bytes=getattr(backend, "setup_bytes", 0),
            fallback=getattr(backend, "reason", None),
            ocr=self.ocr.info if isinstance(self.ocr, LocalOCR) else None,
            actions=self.events.copy(),
            visual_cache_hits=self._visual_cache.hits,
            visual_cache_misses=self._visual_cache.misses,
            visual_cache_bytes=self._visual_cache.bytes,
            outcome=self.outcome(),
        )

    @property
    def capture_repeated(self):
        return bool(self._backend and self._backend.repeat)

    def recognize(self, image):
        self.stats["ocr_calls"] += 1
        self.stats["ocr_pixels"] = self.stats.get("ocr_pixels", 0) + image.shape[0] * image.shape[1]
        return list(self.ocr(image))

    def resolve(self, target):
        if isinstance(target, str):
            if target not in self.targets:
                raise ValueError("Unknown UI target: " + target)
            return self.targets[target]
        return selector(target)

    def register(self, name, target):
        self.targets[name] = self.resolve(target)

    def busy_while(self, target, allowance):
        """Withhold input during this state and give positive-timeout lookups a bounded extra wait."""
        self.busy_states.append((self.resolve(target), self._timeout(allowance)))

    def on_popup(self, name, when, dismiss, *, max_occurrences=1, timeout=5):
        """Dismiss a known interruption, within its bounds, during positive waits.

        Its time is charged to the original operation; the underlying target is
        reacquired afterward. Zero-time queries never send popup input.
        """
        if not isinstance(name, str) or not name or name in self.popups:
            raise ValueError("Pop-up names must be nonempty and unique.")
        if type(max_occurrences) is not int or max_occurrences < 1:
            raise ValueError("Pop-up occurrence limit must be a positive integer.")
        when, dismiss = self.resolve(when), self.resolve(dismiss)
        if any(getattr(t, "condition_only", False) for t in (when, dismiss)):
            raise ValueError("Pop-ups require real bounds and an input target.")
        timeout = self._timeout(timeout)
        if timeout == 0:
            raise ValueError("Pop-up timeout must be positive.")
        self.popups[name] = PopupRule(name, when, dismiss, max_occurrences, timeout)

    def off_popup(self, name):
        del self.popups[name]

    @tracked
    def open(self, package, activity=None):
        self.device.launch(package, activity)
        self.events.append({"open": package})

    def stop(self, package):
        self.device.shell("am", "force-stop", package)
        if self.device.package == package:
            self.device.package = None

    def snapshot(self, filename=None):
        image = self.frame()
        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(path.suffix or ".png", image)
            if not ok:
                raise ValueError("Could not encode screenshot.")
            path.write_bytes(encoded.tobytes())
        return image

    def find_all(self, target):
        return self.resolve(target).matches(Observation(self))

    def _pause(self, deadline, observation):
        # A streaming read already waits for the next frame notification. Native-only
        # observations and on-demand capture use a bounded acquisition interval.
        if "image" in observation.__dict__ and getattr(self._backend, "frame_driven", False):
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(self.poll_interval, remaining))

    def _points(self, targets, deadline, required=True, completed=None, first=False):
        targets = [self.resolve(target) for target in targets]
        completed = self.resolve(completed) if completed is not None else None
        can_extend = deadline > time.monotonic()
        spent = 0.0
        busy_since = None
        allowance = 0.0
        while True:
            obs = Observation(self)
            now = time.monotonic()
            if self.popups and handle_popup(self, obs, deadline, can_extend):
                if time.monotonic() >= deadline:
                    if not required:
                        return False, obs
                    raise TargetNotFoundError("Pop-up prevented target lookup before timeout.")
                continue
            if busy_since is not None:
                actual = min(now - busy_since, max(0.0, allowance - spent))
                spent += actual
                deadline += actual
                busy_since = None
            busy = next(
                (
                    (guard, limit)
                    for guard, limit in self.busy_states
                    if guard.matches(obs)
                ),
                None,
            )
            if busy is not None:
                allowance = busy[1] if can_extend else 0.0
                remaining = deadline - time.monotonic() + max(0.0, allowance - spent)
                if remaining <= 0:
                    if not required:
                        return False, obs
                    raise TargetNotFoundError(
                        "Timed out waiting for the busy screen to clear."
                    )
                busy_since = now
                self._pause(time.monotonic() + remaining, obs)
                continue
            if completed is not None and len(completed.matches(obs)) == 1:
                return None, obs
            found = [target.matches(obs) for target in targets]
            if first:
                for index, matches in enumerate(found):
                    if len(matches) == 1:
                        box = matches[0]
                        point = targets[index].point(box) if hasattr(targets[index], "point") else box.center
                        return (index, point), obs
            if all(len(matches) == 1 for matches in found):
                obs.boxes = [matches[0] for matches in found]
                return [
                    target.point(box) if hasattr(target, "point") else box.center
                    for target, box in zip(targets, obs.boxes)
                ], obs
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not required:
                    return False, obs
                if any(len(matches) > 1 for matches in found):
                    raise AmbiguousTargetError(
                        "Multiple matches; narrow the selector or its region."
                    )
                raise TargetNotFoundError(f"Timed out waiting for {targets!r}.")
            self._pause(deadline, obs)

    def _find(self, target, deadline, required=True, completed=None):
        points, obs = self._points([target], deadline, required, completed)
        return (points[0] if isinstance(points, list) else points), obs

    def _input_area(self, obs):
        area = obs.viewport
        if "image" in obs.__dict__:
            size = self.device.display_size()
            captured = (obs.image.shape[1], obs.image.shape[0])
            if captured != (area.w, area.h) or (size and size != captured):
                raise DeviceError("Display dimensions changed; input was withheld.")
        if time.monotonic() - obs.created > 8:
            raise DeviceError("Observation too old to use for input.")
        return area

    @tracked
    def wait(self, target, timeout=None):
        return self._find(target, self._deadline(timeout))[0]

    @tracked
    def exists(self, target, timeout=0):
        return self._find(target, self._deadline(timeout), False)[0]

    @tracked
    def assert_exists(self, target, timeout=None):
        return self.wait(target, timeout)

    @tracked
    def assert_not_exists(self, target, timeout=None):
        self.wait(Absent(self.resolve(target)), timeout)
        return True

    @tracked
    def wait_any(self, conditions, timeout=None):
        """Evaluate named alternatives on one observation; return the first unique match."""
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError("wait_any requires a nonempty mapping of names to selectors.")
        names = list(conditions)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Condition names must be nonempty strings.")
        (index, point), _ = self._points(list(conditions.values()), self._deadline(timeout), first=True)
        return {"condition": names[index], "point": point}

    @tracked
    def touch(
        self,
        target,
        *,
        expect=None,
        timeout=None,
        required=True,
        offset=(0, 0),
        offset_unit="pixels",
        retries=0,
        retry_safe=False,
        duration=0,
    ):
        if retries and (not retry_safe or expect is None):
            raise ValueError(
                "Repeating a dispatched touch requires retry_safe and a postcondition."
            )
        if (
            not isinstance(retries, int)
            or retries < 0
            or not np.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("Invalid touch policy.")
        if getattr(self.resolve(target), "condition_only", False):
            raise ValueError("Absence is a condition, not an input target.")
        deadline = self._deadline(timeout)
        previous = None
        for attempt in range(retries + 1):
            point, obs = self._find(
                target, deadline, required, expect if attempt else None
            )
            if point is None:
                return previous
            if point is False:
                return False
            x, y = point
            dx, dy = offset
            if offset_unit not in ("pixels", "screen", "target"):
                raise ValueError("Unknown offset unit.")
            # Node-only tests avoid transferring a screenshot to validate a native UI bound.
            area = self._input_area(obs)
            w, h = area.w, area.h
            if offset_unit == "screen":
                dx, dy = dx * (w - 1), dy * (h - 1)
            if offset_unit == "target":
                dx, dy = dx * obs.boxes[0].w, dy * obs.boxes[0].h
            x, y = round(x + dx), round(y + dy)
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError("Touch offset outside screen.")
            self.device.tap(x, y, duration)
            previous = (x, y)
            self.stats["input_events"] += 1
            self.events.append({"touch": [x, y]})
            if expect is None:
                return previous
            remaining = max(0, deadline - time.monotonic())
            if self.exists(expect, remaining / (retries - attempt + 1)):
                return previous
        if required:
            raise TargetNotFoundError("Touch sent; its postcondition was not observed.")
        return False

    @tracked
    def swipe(self, start, end, duration=0.35, timeout=None):
        if any(getattr(self.resolve(t), "condition_only", False) for t in (start, end)):
            raise ValueError("Swipe endpoints must identify screen positions.")
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError("Invalid swipe duration.")
        deadline = self._deadline(timeout)
        (a, b), obs = self._points([start, end], deadline)
        area = self._input_area(obs)
        if any(not (0 <= x < area.w and 0 <= y < area.h) for x, y in (a, b)):
            raise ValueError("Swipe outside the screen.")
        self.device.swipe(a, b, duration)
        self.stats["input_events"] += 1
        self.events.append({"swipe": [a, b]})
        return a, b

    @tracked
    def key(self, key):
        self.device.key(key)
        self.stats["input_events"] += 1

    @tracked
    def text(self, value):
        self.device.text(value)
        self.stats["input_events"] += 1
