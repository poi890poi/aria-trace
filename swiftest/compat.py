"""Independent implementation of the documented Android scripting subset."""

import atexit
import time
from pathlib import Path
from urllib.parse import urlsplit

from .session import Session, current, TargetNotFoundError
from .targets import Template as NativeTemplate

__all__ = [
    "Template",
    "TargetNotFoundError",
    "connect_device",
    "device",
    "auto_setup",
    "touch",
    "click",
    "swipe",
    "wait",
    "exists",
    "find_all",
    "assert_exists",
    "assert_not_exists",
    "keyevent",
    "text",
    "snapshot",
    "sleep",
    "start_app",
    "stop_app",
    "assert_equal",
    "assert_not_equal",
]
__all__ += ["home", "wake", "install", "uninstall", "clear_app", "push", "pull"]


def connect_device(uri="Android:///"):
    parts = urlsplit(uri)
    if (
        parts.scheme.lower() != "android"
        or parts.netloc
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "Compatibility URI must be Android:/// or Android:///SERIAL. Set backend on the runner."
        )
    serial = parts.path.strip("/") or None
    try:
        session = current()
    except RuntimeError:
        session = Session(serial)
        session.__enter__()
        atexit.register(session.close)
    if serial and session.device.serial != serial:
        raise ValueError("URI identifies a different device from this run.")
    return session.device


def device():
    return current().device


def auto_setup(basedir=None, devices=None, logdir=None, project_root=None):
    if devices and len(devices) != 1:
        raise ValueError("Use one device per test session.")
    connect_device(devices[0] if devices else "Android:///")
    session = current()
    if basedir:
        path = Path(basedir).resolve()
        session.base = path if path.is_dir() else path.parent
    if project_root:
        session.base = Path(project_root).resolve()
    if logdir:
        session.logdir = session.base / ("log" if logdir is True else logdir)


class Template:
    def __init__(
        self,
        filename,
        threshold=None,
        target_pos=5,
        record_pos=None,
        resolution=(),
        rgb=False,
    ):
        if target_pos not in range(1, 10):
            raise ValueError("target_pos must be 1 to 9.")
        if record_pos is not None and len(record_pos) != 2:
            raise ValueError("Invalid record_pos.")
        if resolution and (len(resolution) != 2 or min(resolution) <= 0):
            raise ValueError("Invalid recorded resolution.")
        self.filename, self.threshold, self.record_pos, self.resolution = (
            filename,
            threshold,
            record_pos,
            resolution,
        )
        self.rgb = rgb
        self.anchor = (((target_pos - 1) % 3) / 2, ((target_pos - 1) // 3) / 2)

    def matches(self, obs, scope=None):
        scale = (
            min(obs.screen.w / self.resolution[0], obs.screen.h / self.resolution[1])
            if self.resolution
            else 1.0
        )
        template = NativeTemplate(
            self.filename,
            scales=tuple(dict.fromkeys((scale, 1.0))),
            threshold=0.8 if self.threshold is None else self.threshold,
            anchor=self.anchor,
        )
        template.color = self.rgb
        boxes = template.matches(obs, scope)
        if len(boxes) > 1 and self.record_pos is not None:
            p = (
                obs.screen.w * (0.5 + self.record_pos[0]),
                obs.screen.h * 0.5 + obs.screen.w * self.record_pos[1],
            )
            boxes.sort(
                key=lambda b: (b.center[0] - p[0]) ** 2 + (b.center[1] - p[1]) ** 2
            )
        return boxes

    def point(self, b):
        return b.x + int((b.w - 1) * self.anchor[0] + 0.5), b.y + int(
            (b.h - 1) * self.anchor[1] + 0.5
        )


class _First:
    def __init__(self, target):
        self.target = target

    def matches(self, obs, scope=None):
        return self.target.matches(obs, scope)[:1]

    def point(self, b):
        return self.target.point(b)


def _target(t):
    return _First(t) if isinstance(t, Template) else t


def touch(v, times=1, duration=0):
    if not isinstance(times, int) or times < 1:
        raise ValueError("times must be positive.")
    for _ in range(times):
        point = current().touch(_target(v), duration=duration)
    return point


click = touch


def swipe(v1, v2=None, vector=None, duration=0.35):
    if (v2 is None) == (vector is None):
        raise ValueError("Supply one of v2 or vector.")
    if vector is not None:
        a = current().wait(_target(v1))
        height, width = current().frame().shape[:2]
        dx, dy = vector
        if all(abs(n) <= 1 for n in vector):
            dx, dy = dx * width, dy * height
        return current().swipe(a, (round(a[0] + dx), round(a[1] + dy)), duration)
    return current().swipe(_target(v1), _target(v2), duration)


def wait(v, timeout=None, interval=0.5, intervalfunc=None):
    session = current()
    if interval == 0.5 and intervalfunc is None:
        return session.wait(_target(v), timeout)
    if interval <= 0:
        raise ValueError("interval must be positive.")
    deadline = session._deadline(timeout)
    while True:
        point = session.exists(_target(v))
        if point is not False:
            return point
        if time.monotonic() >= deadline:
            raise TargetNotFoundError("Template wait timed out.")
        if intervalfunc is not None:
            intervalfunc()
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def exists(v):
    return current().exists(_target(v), current().timeout)


def assert_exists(v, msg=""):
    return wait(v)


def assert_not_exists(v, msg=""):
    return current().assert_not_exists(_target(v))


def keyevent(keyname):
    current().key(keyname)


def text(text, enter=True):
    current().text(text)
    if enter:
        keyevent("ENTER")


def sleep(secs=1):
    time.sleep(secs)


def start_app(package, activity=None):
    current().open(package, activity)


def stop_app(package):
    current().stop(package)


def home():
    device().shell("input", "keyevent", "HOME")
    device().package = None


def wake():
    device().shell("input", "keyevent", "KEYCODE_WAKEUP")


def install(filepath):
    return device().install(filepath)


def uninstall(package):
    return device().uninstall(package)


def clear_app(package):
    return device().clear(package)


def push(local, remote):
    return device().push(local, remote)


def pull(remote, local):
    return device().pull(remote, local)


def assert_equal(first, second, msg=""):
    if first != second:
        raise AssertionError(msg or f"{first!r} != {second!r}")


def assert_not_equal(first, second, msg=""):
    if first == second:
        raise AssertionError(msg or f"{first!r} == {second!r}")


def find_all(v):
    boxes = current().find_all(v)
    return [
        dict(
            result=v.point(b),
            rectangle=(
                (b.x, b.y),
                (b.x, b.y + b.h),
                (b.x + b.w, b.y + b.h),
                (b.x + b.w, b.y),
            ),
            confidence=b.score,
        )
        for b in boxes
    ]


def snapshot(filename=None, msg=""):
    session = current()
    folder = getattr(session, "logdir", session.base)
    path = folder / (filename or f"screen-{time.time_ns()}.png")
    image = session.snapshot(path)
    return {"screen": str(path), "resolution": (image.shape[1], image.shape[0])}
