"""Python API for Android test scripts."""

from .device import Android, DeviceError, devices
from .session import Session, TargetNotFoundError, AmbiguousTargetError, current
from .targets import Box, Position, Pixel, Template, Node, Text, Dialog, Absent, AnyOf, Variants
from .textmatch import TextPattern
from .popups import PopupHandlingError
from .ocr import LocalOCR

__all__ = [
    "Android",
    "DeviceError",
    "devices",
    "Session",
    "TargetNotFoundError",
    "AmbiguousTargetError",
    "current",
    "Box",
    "Position",
    "Pixel",
    "Template",
    "Node",
    "Text",
    "Dialog",
    "Absent",
    "AnyOf",
    "Variants",
    "TextPattern",
    "PopupHandlingError",
    "LocalOCR",
    "open_app",
    "touch",
    "swipe",
    "exists",
    "wait",
    "wait_any",
    "intent",
    "checkpoint",
    "assert_exists",
    "assert_not_exists",
    "keyevent",
    "text",
    "snapshot",
]


def open_app(package, activity=None):
    return current().open(package, activity)


def touch(target, **options):
    return current().touch(target, **options)


def swipe(start, end, **options):
    return current().swipe(start, end, **options)


def exists(target, **options):
    return current().exists(target, **options)


def wait(target, **options):
    return current().wait(target, **options)


def wait_any(conditions, **options):
    return current().wait_any(conditions, **options)


def intent(name, **details):
    return current().intent(name, **details)


def checkpoint(value, name="checkpoint"):
    return current().checkpoint(value, name)


def assert_exists(target, **options):
    return current().assert_exists(target, **options)


def assert_not_exists(target, **options):
    return current().assert_not_exists(target, **options)


def keyevent(key):
    return current().key(key)


def text(value):
    return current().text(value)


def snapshot(filename=None):
    return current().snapshot(filename)
