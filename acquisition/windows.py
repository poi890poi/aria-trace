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

    def is_foreground(self, hwnd: int) -> bool:
        return int(self.user32.GetForegroundWindow() or 0) == int(hwnd)

    def input_snapshot(self, hwnd: int) -> dict:
        foreground = self.is_foreground(hwnd)
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

# Windows Raw Input structures are kept private; WindowsRawKeyboardMouseSource
# exposes only game-neutral input packets.
_WM_INPUT = 0x00FF
_WM_QUIT = 0x0012
_RID_INPUT = 0x10000003
_RIM_TYPEMOUSE = 0
_RIM_TYPEKEYBOARD = 1
_RIDEV_REMOVE = 0x00000001
_RIDEV_INPUTSINK = 0x00000100
_MOUSE_MOVE_ABSOLUTE = 0x0001
_RI_KEY_BREAK = 0x0001
_RI_KEY_E0 = 0x0002
_RI_KEY_E1 = 0x0004

_RAW_MOUSE_BUTTONS = {
    0x0001: "left_down",
    0x0002: "left_up",
    0x0004: "right_down",
    0x0008: "right_up",
    0x0010: "middle_down",
    0x0020: "middle_up",
    0x0040: "x1_down",
    0x0080: "x1_up",
    0x0100: "x2_down",
    0x0200: "x2_up",
}


class _RawInputDevice(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class _RawInputHeader(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RawMouseButtonData(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class _RawMouseButtons(ctypes.Union):
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("data", _RawMouseButtonData),
    ]


class _RawMouse(ctypes.Structure):
    _anonymous_ = ("buttons",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("buttons", _RawMouseButtons),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class _RawKeyboard(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class _RawHid(ctypes.Structure):
    _fields_ = [
        ("dwSizeHid", wintypes.DWORD),
        ("dwCount", wintypes.DWORD),
        ("bRawData", wintypes.BYTE * 1),
    ]


class _RawInputData(ctypes.Union):
    _fields_ = [
        ("mouse", _RawMouse),
        ("keyboard", _RawKeyboard),
        ("hid", _RawHid),
    ]


class _RawInput(ctypes.Structure):
    _fields_ = [("header", _RawInputHeader), ("data", _RawInputData)]


_RAW_WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)


class _RawWindowClass(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", _RAW_WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


def decode_raw_input(raw: _RawInput, host_time_ns: int) -> Optional[dict]:
    """Convert one Windows RAWINPUT record into durable neutral evidence."""
    device_handle = int(raw.header.hDevice or 0)
    if int(raw.header.dwType) == _RIM_TYPEMOUSE:
        mouse = raw.data.mouse
        button_flags = int(mouse.data.usButtonFlags)
        button_data = int(mouse.data.usButtonData)
        transitions = [
            name
            for flag, name in _RAW_MOUSE_BUTTONS.items()
            if button_flags & flag
        ]
        wheel_delta = (
            int(ctypes.c_short(button_data).value)
            if button_flags & 0x0400
            else 0
        )
        horizontal_wheel_delta = (
            int(ctypes.c_short(button_data).value)
            if button_flags & 0x0800
            else 0
        )
        return {
            "kind": "pc_raw_mouse",
            "host_time_ns": int(host_time_ns),
            "payload": {
                "device_handle": device_handle,
                "movement_mode": (
                    "absolute"
                    if int(mouse.usFlags) & _MOUSE_MOVE_ABSOLUTE
                    else "relative"
                ),
                "delta_x": int(mouse.lLastX),
                "delta_y": int(mouse.lLastY),
                "button_transitions": transitions,
                "wheel_delta": wheel_delta,
                "horizontal_wheel_delta": horizontal_wheel_delta,
                "raw_button_flags": button_flags,
                "raw_button_data": button_data,
                "raw_buttons": int(mouse.ulRawButtons),
                "mouse_flags": int(mouse.usFlags),
                "extra_information": int(mouse.ulExtraInformation),
            },
        }
    if int(raw.header.dwType) == _RIM_TYPEKEYBOARD:
        keyboard = raw.data.keyboard
        flags = int(keyboard.Flags)
        virtual_key = int(keyboard.VKey)
        return {
            "kind": "pc_raw_keyboard",
            "host_time_ns": int(host_time_ns),
            "payload": {
                "device_handle": device_handle,
                "virtual_key": virtual_key,
                "key_name": key_name(virtual_key),
                "scan_code": int(keyboard.MakeCode),
                "pressed": not bool(flags & _RI_KEY_BREAK),
                "extended_e0": bool(flags & _RI_KEY_E0),
                "extended_e1": bool(flags & _RI_KEY_E1),
                "raw_flags": flags,
                "windows_message": int(keyboard.Message),
                "extra_information": int(keyboard.ExtraInformation),
            },
        }
    return None

class WindowsRawInputApi:
    """Own a message-only window and decode WM_INPUT on its thread."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows Raw Input is available only on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._thread_id = None
        self._wndproc = None
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
        self.kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        self.user32.RegisterClassW.argtypes = [
            ctypes.POINTER(_RawWindowClass)
        ]
        self.user32.RegisterClassW.restype = wintypes.ATOM
        self.user32.UnregisterClassW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.HINSTANCE,
        ]
        self.user32.UnregisterClassW.restype = wintypes.BOOL
        self.user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HANDLE,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self.user32.CreateWindowExW.restype = wintypes.HWND
        self.user32.DestroyWindow.argtypes = [wintypes.HWND]
        self.user32.DestroyWindow.restype = wintypes.BOOL
        self.user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.DefWindowProcW.restype = ctypes.c_ssize_t
        self.user32.RegisterRawInputDevices.argtypes = [
            ctypes.POINTER(_RawInputDevice),
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.RegisterRawInputDevices.restype = wintypes.BOOL
        self.user32.GetRawInputData.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.UINT),
            wintypes.UINT,
        ]
        self.user32.GetRawInputData.restype = wintypes.UINT
        self.user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.GetMessageW.restype = ctypes.c_int
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        self.user32.TranslateMessage.restype = wintypes.BOOL
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        self.user32.DispatchMessageW.restype = ctypes.c_ssize_t
        self.user32.PostThreadMessageW.argtypes = [
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.PostThreadMessageW.restype = wintypes.BOOL

    def _read(self, raw_handle: int, host_time_ns: int) -> Optional[dict]:
        size = wintypes.UINT(0)
        header_size = ctypes.sizeof(_RawInputHeader)
        result = self.user32.GetRawInputData(
            wintypes.HANDLE(raw_handle),
            _RID_INPUT,
            None,
            ctypes.byref(size),
            header_size,
        )
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(size.value)
        result = self.user32.GetRawInputData(
            wintypes.HANDLE(raw_handle),
            _RID_INPUT,
            buffer,
            ctypes.byref(size),
            header_size,
        )
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if result != size.value:
            raise RuntimeError(
                "Windows returned {} of {} raw-input bytes".format(
                    result, size.value
                )
            )
        raw = ctypes.cast(buffer, ctypes.POINTER(_RawInput)).contents
        return decode_raw_input(raw, host_time_ns)

    def run(
        self,
        emit: Callable[[dict], None],
        ready_event: threading.Event,
    ) -> None:
        self._thread_id = int(self.kernel32.GetCurrentThreadId())
        instance = self.kernel32.GetModuleHandleW(None)
        class_name = "AriaTraceRawInput_{}_{}".format(os.getpid(), id(self))

        def window_proc(hwnd, message, wparam, lparam):
            if message == _WM_INPUT:
                try:
                    record = self._read(int(lparam), time.perf_counter_ns())
                    if record is not None:
                        emit(record)
                except Exception as exc:
                    emit(
                        {
                            "kind": "pc_raw_input_error",
                            "host_time_ns": time.perf_counter_ns(),
                            "payload": {
                                "error": "{}: {}".format(
                                    type(exc).__name__, exc
                                )
                            },
                        }
                    )
            return self.user32.DefWindowProcW(
                hwnd, message, wparam, lparam
            )

        self._wndproc = _RAW_WNDPROC(window_proc)
        window_class = _RawWindowClass()
        window_class.lpfnWndProc = self._wndproc
        window_class.hInstance = instance
        window_class.lpszClassName = class_name
        atom = self.user32.RegisterClassW(ctypes.byref(window_class))
        if not atom:
            raise ctypes.WinError(ctypes.get_last_error())

        hwnd = None
        registered = False
        try:
            hwnd = self.user32.CreateWindowExW(
                0,
                class_name,
                class_name,
                0,
                0,
                0,
                0,
                0,
                wintypes.HWND(-3),
                None,
                instance,
                None,
            )
            if not hwnd:
                raise ctypes.WinError(ctypes.get_last_error())
            devices = (_RawInputDevice * 2)(
                _RawInputDevice(0x01, 0x02, _RIDEV_INPUTSINK, hwnd),
                _RawInputDevice(0x01, 0x06, _RIDEV_INPUTSINK, hwnd),
            )
            if not self.user32.RegisterRawInputDevices(
                devices, len(devices), ctypes.sizeof(_RawInputDevice)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            registered = True
            ready_event.set()

            message = wintypes.MSG()
            while True:
                result = self.user32.GetMessageW(
                    ctypes.byref(message), None, 0, 0
                )
                if result == 0:
                    break
                if result == -1:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.user32.TranslateMessage(ctypes.byref(message))
                self.user32.DispatchMessageW(ctypes.byref(message))
        finally:
            ready_event.set()
            if registered:
                removals = (_RawInputDevice * 2)(
                    _RawInputDevice(0x01, 0x02, _RIDEV_REMOVE, None),
                    _RawInputDevice(0x01, 0x06, _RIDEV_REMOVE, None),
                )
                self.user32.RegisterRawInputDevices(
                    removals,
                    len(removals),
                    ctypes.sizeof(_RawInputDevice),
                )
            if hwnd:
                self.user32.DestroyWindow(hwnd)
            self.user32.UnregisterClassW(class_name, instance)
            self._thread_id = None
            self._wndproc = None

    def stop(self) -> None:
        thread_id = self._thread_id
        if thread_id is not None:
            self.user32.PostThreadMessageW(
                thread_id, _WM_QUIT, 0, 0
            )


class WindowsRawKeyboardMouseSource(InputSource):
    """Capture keyboard transitions and true relative mouse input."""

    def __init__(
        self,
        window_title: str,
        source_id: str = "pc-raw-input",
        exact_title: bool = False,
        desktop_api=None,
        raw_input_api=None,
    ) -> None:
        self.window_title = window_title
        self.source_id = source_id
        self.exact_title = bool(exact_title)
        self.desktop_api = desktop_api
        self.raw_input_api = raw_input_api
        self.hwnd = None
        self.matched_title = None
        self._thread = None
        self._ready_event = threading.Event()
        self._errors = []
        self._raw_packets_received = 0
        self._raw_packets_accepted = 0
        self._raw_packets_rejected_foreground = 0
        self._filter_foreground = True
        self._foreground_predicate = None
        self._foreground_authority = "selected_window_hwnd"

    def set_foreground_predicate(self, predicate: Callable[[], bool]) -> None:
        """Use the orchestrator's focus decision instead of a second HWND check."""
        self._filter_foreground = True
        self._foreground_predicate = predicate
        self._foreground_authority = "orchestrator_gate"

    def disable_foreground_filter(self) -> None:
        """Accept Raw Input for the recorder's already-bounded active lifetime."""
        self._filter_foreground = False
        self._foreground_predicate = None
        self._foreground_authority = "active_capture_lifecycle"

    def _is_foreground(self) -> bool:
        if self._foreground_predicate is not None:
            return bool(self._foreground_predicate())
        if hasattr(self.desktop_api, "is_foreground"):
            return bool(self.desktop_api.is_foreground(self.hwnd))
        return bool(self.desktop_api.input_snapshot(self.hwnd)["foreground"])

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        self.desktop_api = self.desktop_api or WindowsDesktopApi()
        self.raw_input_api = self.raw_input_api or WindowsRawInputApi()
        self.hwnd, self.matched_title = select_window(
            self.desktop_api.list_windows(),
            self.window_title,
            self.exact_title,
        )
        self._ready_event.clear()
        self._errors = []
        self._raw_packets_received = 0
        self._raw_packets_accepted = 0
        self._raw_packets_rejected_foreground = 0

        def handle(record: dict) -> None:
            self._raw_packets_received += 1
            if record["kind"] == "pc_raw_input_error":
                emit(
                    InputPacket(
                        self.source_id,
                        record["kind"],
                        record["host_time_ns"],
                        record["payload"],
                    )
                )
                return
            if self._filter_foreground and not self._is_foreground():
                self._raw_packets_rejected_foreground += 1
                return
            self._raw_packets_accepted += 1
            payload = dict(record["payload"])
            payload.update(
                {
                    "window_title": self.matched_title,
                    "foreground": True,
                }
            )
            emit(
                InputPacket(
                    self.source_id,
                    record["kind"],
                    record["host_time_ns"],
                    payload,
                )
            )

        def run() -> None:
            try:
                self.raw_input_api.run(handle, self._ready_event)
            except Exception as exc:
                self._errors.append(exc)
                emit(
                    InputPacket(
                        self.source_id,
                        "pc_raw_input_error",
                        time.perf_counter_ns(),
                        {"error": "{}: {}".format(type(exc).__name__, exc)},
                    )
                )
                self._ready_event.set()

        self._thread = threading.Thread(
            target=run,
            name="pc-raw-input-message-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(2.0):
            self.stop()
            raise RuntimeError("Windows Raw Input did not initialize")
        if self._errors:
            error = self._errors[0]
            self.stop()
            raise RuntimeError(
                "Windows Raw Input initialization failed: {}".format(error)
            )

    def stop(self) -> None:
        if self.raw_input_api is not None:
            self.raw_input_api.stop()
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
                "foreground_only": (
                    self._foreground_authority != "active_capture_lifecycle"
                ),
                "acceptance_policy": self._foreground_authority,
                "keyboard": "raw_make_break_scan_code_and_virtual_key",
                "mouse_motion": "raw_relative_delta",
                "mouse_buttons": "raw_button_transitions_and_wheel",
                "timing": "pc_monotonic_per_raw_event",
                "behavior_fidelity": "keyboard_and_locked_camera_mouse",
                "raw_input_diagnostics": {
                    "packets_received": self._raw_packets_received,
                    "packets_accepted": self._raw_packets_accepted,
                    "packets_rejected_foreground": (
                        self._raw_packets_rejected_foreground
                    ),
                    "foreground_authority": self._foreground_authority,
                },
            }
        )
        return result
