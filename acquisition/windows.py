"""Dependency-free Windows game-window capture and passive input observation."""

import ctypes
import os
import threading
import time
from ctypes import wintypes
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import FramePacket, InputPacket
from .sources import FrameSource, InputSource


_MOUSE_KEYS = {
    0x01: "left",
    0x02: "right",
    0x04: "middle",
    0x05: "x1",
    0x06: "x2",
}
_KEY_NAMES = {
    0x08: "backspace",
    0x09: "tab",
    0x0D: "enter",
    0x10: "shift",
    0x11: "ctrl",
    0x12: "alt",
    0x1B: "escape",
    0x20: "space",
    0x25: "left",
    0x26: "up",
    0x27: "right",
    0x28: "down",
}
_KEY_NAMES.update({code: chr(code) for code in range(ord("0"), ord("9") + 1)})
_KEY_NAMES.update({code: chr(code) for code in range(ord("A"), ord("Z") + 1)})


def select_window(
    windows: Sequence[Tuple[int, str]], title: str, exact: bool = False
) -> Tuple[int, str]:
    """Select one visible window, rejecting ambiguous substring matches."""
    query = title.strip()
    if not query:
        raise ValueError("Window title must not be empty")
    if exact:
        matches = [item for item in windows if item[1].casefold() == query.casefold()]
    else:
        matches = [item for item in windows if query.casefold() in item[1].casefold()]
        exact_matches = [item for item in matches if item[1].casefold() == query.casefold()]
        if len(exact_matches) == 1:
            matches = exact_matches
    if not matches:
        raise RuntimeError("No visible window matches {!r}".format(title))
    if len(matches) > 1:
        names = ", ".join(repr(item[1]) for item in matches[:8])
        raise RuntimeError("Window title {!r} is ambiguous: {}".format(title, names))
    return matches[0]


def key_name(virtual_key: int) -> str:
    return _KEY_NAMES.get(virtual_key, "vk_{:02x}".format(virtual_key))


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("bmiHeader", _BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]


class WindowsDesktopApi:
    """Small ctypes wrapper kept injectable for deterministic tests."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows desktop capture is available only on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._configure_signatures()
        try:
            self.user32.SetProcessDPIAware()
        except AttributeError:
            pass

    def _configure_signatures(self) -> None:
        handle = wintypes.HANDLE
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.IsWindowVisible.restype = wintypes.BOOL
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.user32.IsIconic.restype = wintypes.BOOL
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.GetClientRect.restype = wintypes.BOOL
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.ClientToScreen.restype = wintypes.BOOL
        self.user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.ScreenToClient.restype = wintypes.BOOL
        self.user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self.user32.GetCursorPos.restype = wintypes.BOOL
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.user32.GetDC.argtypes = [wintypes.HWND]
        self.user32.GetDC.restype = handle
        self.user32.ReleaseDC.argtypes = [wintypes.HWND, handle]
        self.user32.ReleaseDC.restype = ctypes.c_int
        self.gdi32.CreateCompatibleDC.argtypes = [handle]
        self.gdi32.CreateCompatibleDC.restype = handle
        self.gdi32.CreateCompatibleBitmap.argtypes = [handle, ctypes.c_int, ctypes.c_int]
        self.gdi32.CreateCompatibleBitmap.restype = handle
        self.gdi32.SelectObject.argtypes = [handle, handle]
        self.gdi32.SelectObject.restype = handle
        self.gdi32.BitBlt.argtypes = [
            handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            handle, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
        ]
        self.gdi32.BitBlt.restype = wintypes.BOOL
        self.gdi32.GetDIBits.argtypes = [
            handle, handle, wintypes.UINT, wintypes.UINT, wintypes.LPVOID,
            ctypes.POINTER(_BitmapInfo), wintypes.UINT,
        ]
        self.gdi32.GetDIBits.restype = ctypes.c_int
        self.gdi32.DeleteObject.argtypes = [handle]
        self.gdi32.DeleteObject.restype = wintypes.BOOL
        self.gdi32.DeleteDC.argtypes = [handle]
        self.gdi32.DeleteDC.restype = wintypes.BOOL

    def list_windows(self) -> List[Tuple[int, str]]:
        values = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def visit(hwnd, _parameter):
            if not self.user32.IsWindowVisible(hwnd):
                return True
            length = self.user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title:
                values.append((int(hwnd), title))
            return True

        callback = callback_type(visit)
        if not self.user32.EnumWindows(callback, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return values

    def _client_geometry(self, hwnd: int) -> Tuple[int, int, int, int]:
        if self.user32.IsIconic(hwnd):
            raise RuntimeError("Selected window is minimized")
        rect = wintypes.RECT()
        if not self.user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        origin = wintypes.POINT(0, 0)
        if not self.user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise ctypes.WinError(ctypes.get_last_error())
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            raise RuntimeError("Selected window has an empty client area")
        return int(origin.x), int(origin.y), width, height

    def capture_client(self, hwnd: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        left, top, width, height = self._client_geometry(hwnd)
        screen_dc = self.user32.GetDC(None)
        memory_dc = self.gdi32.CreateCompatibleDC(screen_dc)
        bitmap = self.gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        previous = self.gdi32.SelectObject(memory_dc, bitmap)
        try:
            if not self.gdi32.BitBlt(
                memory_dc, 0, 0, width, height, screen_dc, left, top, 0x00CC0020
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            info = _BitmapInfo()
            info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            buffer = ctypes.create_string_buffer(width * height * 4)
            copied = self.gdi32.GetDIBits(
                memory_dc, bitmap, 0, height, buffer, ctypes.byref(info), 0
            )
            if copied != height:
                raise RuntimeError("Windows copied {} of {} capture rows".format(copied, height))
            bgra = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
            return bgra[:, :, :3].copy(), (left, top, width, height)
        finally:
            if previous:
                self.gdi32.SelectObject(memory_dc, previous)
            if bitmap:
                self.gdi32.DeleteObject(bitmap)
            if memory_dc:
                self.gdi32.DeleteDC(memory_dc)
            if screen_dc:
                self.user32.ReleaseDC(None, screen_dc)

    def input_snapshot(self, hwnd: int) -> dict:
        foreground = int(self.user32.GetForegroundWindow() or 0) == int(hwnd)
        pressed = []
        buttons = []
        if foreground:
            for virtual_key in range(1, 256):
                if not self.user32.GetAsyncKeyState(virtual_key) & 0x8000:
                    continue
                if virtual_key in _MOUSE_KEYS:
                    buttons.append(_MOUSE_KEYS[virtual_key])
                else:
                    pressed.append((virtual_key, key_name(virtual_key)))
        point = wintypes.POINT()
        if not self.user32.GetCursorPos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        client_point = wintypes.POINT(point.x, point.y)
        if not self.user32.ScreenToClient(hwnd, ctypes.byref(client_point)):
            raise ctypes.WinError(ctypes.get_last_error())
        _left, _top, width, height = self._client_geometry(hwnd)
        return {
            "foreground": foreground,
            "keys": pressed,
            "buttons": sorted(buttons),
            "cursor_client": (int(client_point.x), int(client_point.y)),
            "cursor_normalized": (
                float(client_point.x) / float(max(1, width)),
                float(client_point.y) / float(max(1, height)),
            ),
        }


class WindowsWindowFrameSource(FrameSource):
    def __init__(
        self,
        window_title: str,
        stream_id: str = "main",
        fps: float = 30.0,
        exact_title: bool = False,
        api=None,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("Window capture FPS must be positive")
        self.window_title = window_title
        self.stream_id = stream_id
        self.fps = float(fps)
        self.exact_title = bool(exact_title)
        self.api = api
        self.hwnd = None
        self.matched_title = None
        self._running = False
        self._next_frame_ns = 0
        self._shape = None

    def start(self) -> None:
        self.api = self.api or WindowsDesktopApi()
        self.hwnd, self.matched_title = select_window(
            self.api.list_windows(), self.window_title, self.exact_title
        )
        self._running = True
        self._next_frame_ns = time.perf_counter_ns()
        self._shape = None

    def read(self) -> Optional[FramePacket]:
        if not self._running or self.hwnd is None:
            return None
        remaining = self._next_frame_ns - time.perf_counter_ns()
        if remaining > 0:
            time.sleep(remaining / 1.0e9)
        self._next_frame_ns += int(1.0e9 / self.fps)
        capture_time = time.perf_counter_ns()
        image, geometry = self.api.capture_client(self.hwnd)
        receive_time = time.perf_counter_ns()
        height, width = image.shape[:2]
        image = image[: height - height % 2, : width - width % 2]
        shape = image.shape[:2]
        if min(shape) <= 0:
            raise RuntimeError("Window client area is too small to capture")
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise RuntimeError(
                "Window client size changed from {} to {}; keep the game window fixed".format(
                    self._shape, shape
                )
            )
        return FramePacket(
            self.stream_id,
            image,
            capture_time,
            receive_time,
            metadata={
                "window_title": self.matched_title,
                "client_screen_rect": list(geometry),
                "capture_backend": "win32_gdi_visible_client_v1",
            },
        )

    def stop(self) -> None:
        self._running = False

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update(
            {
                "window_title_query": self.window_title,
                "matched_window_title": self.matched_title,
                "exact_title": self.exact_title,
                "requested_fps": self.fps,
                "capture_backend": "win32_gdi_visible_client_v1",
                "requires_visible_unobstructed_window": True,
            }
        )
        return result


class WindowsKeyboardMouseSource(InputSource):
    """Poll keyboard/button/cursor state and emit changes on the PC timeline."""

    def __init__(
        self,
        window_title: str,
        source_id: str = "pc-input",
        poll_hz: float = 125.0,
        exact_title: bool = False,
        api=None,
        ignored_virtual_keys=(),
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("Input polling rate must be positive")
        self.window_title = window_title
        self.source_id = source_id
        self.poll_hz = float(poll_hz)
        self.exact_title = bool(exact_title)
        self.api = api
        self.ignored_virtual_keys = {int(value) for value in ignored_virtual_keys}
        self.hwnd = None
        self.matched_title = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        self.api = self.api or WindowsDesktopApi()
        self.hwnd, self.matched_title = select_window(
            self.api.list_windows(), self.window_title, self.exact_title
        )
        self._stop_event.clear()

        def run() -> None:
            previous = None
            last_emit_ns = 0
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                try:
                    snapshot = self.api.input_snapshot(self.hwnd)
                    snapshot["keys"] = [
                        item for item in snapshot["keys"]
                        if item[0] not in self.ignored_virtual_keys
                    ]
                except Exception as exc:
                    emit(
                        InputPacket(
                            self.source_id,
                            "pc_input_error",
                            now_ns,
                            {"error": "{}: {}".format(type(exc).__name__, exc)},
                        )
                    )
                    return
                signature = (
                    snapshot["foreground"],
                    tuple(snapshot["keys"]),
                    tuple(snapshot["buttons"]),
                    tuple(snapshot["cursor_client"]),
                )
                if signature != previous or now_ns - last_emit_ns >= 1_000_000_000:
                    emit(
                        InputPacket(
                            self.source_id,
                            "pc_input_state",
                            now_ns,
                            {
                                "window_title": self.matched_title,
                                "foreground": snapshot["foreground"],
                                "keys": [
                                    {"virtual_key": code, "name": name}
                                    for code, name in snapshot["keys"]
                                ],
                                "mouse_buttons": list(snapshot["buttons"]),
                                "cursor_client": list(snapshot["cursor_client"]),
                                "cursor_normalized": list(snapshot["cursor_normalized"]),
                            },
                        )
                    )
                    previous = signature
                    last_emit_ns = now_ns
                self._stop_event.wait(1.0 / self.poll_hz)

        self._thread = threading.Thread(target=run, name="pc-input-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update(
            {
                "window_title_query": self.window_title,
                "matched_window_title": self.matched_title,
                "exact_title": self.exact_title,
                "poll_hz": self.poll_hz,
                "foreground_only": True,
                "mouse_motion": "absolute_client_cursor_not_raw_relative_input",
                "ignored_virtual_keys": sorted(self.ignored_virtual_keys),
            }
        )
        return result

_XINPUT_BUTTONS = {
    0x0001: "dpad_up",
    0x0002: "dpad_down",
    0x0004: "dpad_left",
    0x0008: "dpad_right",
    0x0010: "start",
    0x0020: "back",
    0x0040: "left_thumb",
    0x0080: "right_thumb",
    0x0100: "left_shoulder",
    0x0200: "right_shoulder",
    0x1000: "a",
    0x2000: "b",
    0x4000: "x",
    0x8000: "y",
}


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", wintypes.WORD),
        ("bLeftTrigger", wintypes.BYTE),
        ("bRightTrigger", wintypes.BYTE),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD), ("Gamepad", _XInputGamepad)]


def _normalize_stick(value: int) -> float:
    return float(value) / (32767.0 if value >= 0 else 32768.0)


class WindowsXInputApi:
    """Read a standard Windows gamepad without adding project dependencies."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("XInput capture is available only on Windows")
        self.dll = None
        self.dll_name = None
        for name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
            try:
                self.dll = ctypes.WinDLL(name, use_last_error=True)
                self.dll_name = name
                break
            except OSError:
                continue
        if self.dll is None:
            raise RuntimeError("No Windows XInput library is available")
        self.dll.XInputGetState.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(_XInputState),
        ]
        self.dll.XInputGetState.restype = wintypes.DWORD

    def read_state(self, user_index: int) -> Optional[dict]:
        state = _XInputState()
        result = int(self.dll.XInputGetState(int(user_index), ctypes.byref(state)))
        if result == 1167:
            return None
        if result != 0:
            raise OSError(result, "XInputGetState failed")
        gamepad = state.Gamepad
        return {
            "packet_number": int(state.dwPacketNumber),
            "buttons_mask": int(gamepad.wButtons),
            "buttons": [
                name
                for mask, name in _XINPUT_BUTTONS.items()
                if int(gamepad.wButtons) & mask
            ],
            "left_trigger_raw": int(gamepad.bLeftTrigger),
            "right_trigger_raw": int(gamepad.bRightTrigger),
            "left_trigger": float(gamepad.bLeftTrigger) / 255.0,
            "right_trigger": float(gamepad.bRightTrigger) / 255.0,
            "left_stick_raw": [int(gamepad.sThumbLX), int(gamepad.sThumbLY)],
            "right_stick_raw": [int(gamepad.sThumbRX), int(gamepad.sThumbRY)],
            "left_stick": [
                _normalize_stick(int(gamepad.sThumbLX)),
                _normalize_stick(int(gamepad.sThumbLY)),
            ],
            "right_stick": [
                _normalize_stick(int(gamepad.sThumbRX)),
                _normalize_stick(int(gamepad.sThumbRY)),
            ],
        }


class WindowsXInputSource(InputSource):
    """Capture complete standard-controller state with source timestamps."""

    def __init__(
        self,
        window_title: str,
        source_id: str = "pc-xinput",
        poll_hz: float = 250.0,
        user_index: int = 0,
        exact_title: bool = False,
        desktop_api=None,
        xinput_api=None,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("XInput polling rate must be positive")
        if user_index < 0 or user_index > 3:
            raise ValueError("XInput user index must be between 0 and 3")
        self.window_title = window_title
        self.source_id = source_id
        self.poll_hz = float(poll_hz)
        self.user_index = int(user_index)
        self.exact_title = bool(exact_title)
        self.desktop_api = desktop_api
        self.xinput_api = xinput_api
        self.hwnd = None
        self.matched_title = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        self.desktop_api = self.desktop_api or WindowsDesktopApi()
        self.xinput_api = self.xinput_api or WindowsXInputApi()
        self.hwnd, self.matched_title = select_window(
            self.desktop_api.list_windows(), self.window_title, self.exact_title
        )
        self._stop_event.clear()

        def run() -> None:
            previous = None
            last_emit_ns = 0
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                try:
                    foreground = bool(
                        self.desktop_api.input_snapshot(self.hwnd)["foreground"]
                    )
                    state = self.xinput_api.read_state(self.user_index)
                except Exception as exc:
                    emit(
                        InputPacket(
                            self.source_id,
                            "pc_xinput_error",
                            now_ns,
                            {"error": "{}: {}".format(type(exc).__name__, exc)},
                        )
                    )
                    return
                payload = {
                    "window_title": self.matched_title,
                    "foreground": foreground,
                    "connected": state is not None,
                    "user_index": self.user_index,
                    "state": state,
                }
                signature = (
                    foreground,
                    state["packet_number"] if state is not None else None,
                )
                if signature != previous or now_ns - last_emit_ns >= 1_000_000_000:
                    emit(InputPacket(self.source_id, "pc_xinput_state", now_ns, payload))
                    previous = signature
                    last_emit_ns = now_ns
                self._stop_event.wait(1.0 / self.poll_hz)

        self._thread = threading.Thread(
            target=run, name="pc-xinput-poll", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update(
            {
                "window_title_query": self.window_title,
                "matched_window_title": self.matched_title,
                "exact_title": self.exact_title,
                "poll_hz": self.poll_hz,
                "user_index": self.user_index,
                "foreground_tagged": True,
                "motion_axes": "raw_and_normalized_without_deadzone",
                "behavior_fidelity": "buttons_triggers_locomotion_and_view_axes",
            }
        )
        return result