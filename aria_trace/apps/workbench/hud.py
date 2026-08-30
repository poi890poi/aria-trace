"""Capture-safe, always-on-top Windows HUD for the acquisition workbench."""

import argparse
import base64
import ctypes
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin
from typing import Callable, Optional


HUD_WIDTH = 390
HUD_HEIGHT = 92
MAP_HUD_WIDTH = 390
MAP_HUD_HEIGHT = 350
HUD_MARGIN = 22
FOREGROUND_LOSS_GRACE_S = 0.8
TRANSPARENT_KEY = "#ff00ff"

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GA_ROOT = 2


class _HudWindow:
    """Render recorder state without taking focus or entering screen capture."""

    def __init__(
        self,
        status_provider: Callable[[], dict],
        refresh_ms: int = 200,
        foreground_loss_grace_s: float = FOREGROUND_LOSS_GRACE_S,
    ) -> None:
        self.status_provider = status_provider
        self.refresh_ms = int(refresh_ms)
        self.foreground_loss_grace_s = float(foreground_loss_grace_s)
        self._error: Optional[BaseException] = None
        self._hwnd = None
        self.display_affinity = None
        self._last_target_foreground_s = None

    def run_forever(self) -> None:
        """Run Tk on the current thread; used by the isolated HUD process."""
        if os.name != "nt":
            raise RuntimeError("The in-game HUD is available only on Windows")
        self._run()
        if self._error is not None:
            raise RuntimeError("In-game HUD failed: {}".format(self._error))

    @staticmethod
    def _target_position(
        user32,
        root,
        window_title: Optional[str],
        overlay_width: int = HUD_WIDTH,
        overlay_height: int = HUD_HEIGHT,
    ):
        if window_title:
            hwnd = user32.FindWindowW(None, str(window_title))
            if hwnd:
                from ctypes import wintypes

                rect = wintypes.RECT()
                origin = wintypes.POINT(0, 0)
                if user32.GetClientRect(hwnd, ctypes.byref(rect)) and user32.ClientToScreen(
                    hwnd, ctypes.byref(origin)
                ):
                    width = int(rect.right - rect.left)
                    height = int(rect.bottom - rect.top)
                    if width >= overlay_width + HUD_MARGIN * 2 and height >= overlay_height:
                        return (
                            int(origin.x + width - overlay_width - HUD_MARGIN),
                            int(origin.y + HUD_MARGIN),
                        )
        return (
            max(0, int(root.winfo_screenwidth()) - overlay_width - HUD_MARGIN),
            HUD_MARGIN,
        )

    @staticmethod
    def _target_is_foreground(user32, window_title: Optional[str]) -> bool:
        """Show the overlay only while its exact target game owns focus."""
        if not window_title:
            return False
        target = user32.FindWindowW(None, str(window_title))
        return bool(target and user32.GetForegroundWindow() == target)

    def _debounced_visibility(
        self,
        requested_visible: bool,
        target_foreground: bool,
        currently_visible: bool,
        now_s: Optional[float] = None,
    ) -> bool:
        """Ignore one-off foreground misses without delaying deliberate hiding."""

        if not requested_visible:
            self._last_target_foreground_s = None
            return False
        now_s = time.monotonic() if now_s is None else float(now_s)
        if target_foreground:
            self._last_target_foreground_s = now_s
            return True
        if not currently_visible or self._last_target_foreground_s is None:
            return False
        return (
            now_s - self._last_target_foreground_s
            <= self.foreground_loss_grace_s
        )

    @staticmethod
    def _target_client_rect(user32, window_title: Optional[str]):
        if not window_title:
            return None
        hwnd = user32.FindWindowW(None, str(window_title))
        if not hwnd:
            return None
        from ctypes import wintypes

        rect = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not (
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            and user32.ClientToScreen(hwnd, ctypes.byref(origin))
        ):
            return None
        return (
            int(origin.x),
            int(origin.y),
            int(rect.right - rect.left),
            int(rect.bottom - rect.top),
        )

    def _run(self) -> None:
        root = None
        route_root = None
        try:
            import tkinter as tk
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            user32.FindWindowW.restype = wintypes.HWND
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetClientRect.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.RECT),
            ]
            user32.GetClientRect.restype = wintypes.BOOL
            user32.ClientToScreen.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.POINT),
            ]
            user32.ClientToScreen.restype = wintypes.BOOL
            user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            user32.GetAncestor.restype = wintypes.HWND
            user32.SetWindowLongW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_long,
            ]
            user32.SetWindowLongW.restype = ctypes.c_long
            user32.SetWindowDisplayAffinity.argtypes = [
                wintypes.HWND,
                wintypes.DWORD,
            ]
            user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
            user32.GetWindowDisplayAffinity.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.DWORD),
            ]
            user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
            user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.SetWindowPos.restype = wintypes.BOOL
            root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.configure(background="#081018")
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.78)
            root.geometry("{}x{}+0+0".format(HUD_WIDTH, HUD_HEIGHT))

            shell = tk.Frame(
                root,
                background="#081018",
                highlightbackground="#33475f",
                highlightthickness=1,
                padx=13,
                pady=8,
            )
            shell.pack(fill="both", expand=True)
            title = tk.Label(
                shell,
                background="#081018",
                foreground="#9fb4ca",
                anchor="w",
                font=("Segoe UI", 9, "bold"),
            )
            title.pack(fill="x")
            status_label = tk.Label(
                shell,
                background="#081018",
                foreground="#59d7e8",
                anchor="w",
                font=("Segoe UI Semibold", 18),
            )
            status_label.pack(fill="x")
            detail = tk.Label(
                shell,
                background="#081018",
                foreground="#d4dfeb",
                anchor="w",
                font=("Segoe UI", 9),
            )
            detail.pack(fill="x")
            map_label = tk.Label(
                shell,
                background="#081018",
                borderwidth=0,
            )
            map_photo = None
            route_root = tk.Toplevel(root)
            route_root.withdraw()
            route_root.overrideredirect(True)
            route_root.configure(background=TRANSPARENT_KEY)
            route_root.attributes("-topmost", True)
            route_root.attributes("-alpha", 0.88)
            route_root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
            route_label = tk.Label(
                route_root,
                background=TRANSPARENT_KEY,
                borderwidth=0,
                highlightthickness=0,
            )
            route_label.pack(fill="both", expand=True)
            route_photo = None
            root.update_idletasks()

            def configure_capture_safe(window):
                widget_hwnd = int(window.winfo_id())
                hwnd = int(user32.GetAncestor(widget_hwnd, GA_ROOT) or widget_hwnd)
                style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                style |= WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
                if not user32.SetWindowDisplayAffinity(
                    hwnd, WDA_EXCLUDEFROMCAPTURE
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                affinity = wintypes.DWORD(0)
                if not user32.GetWindowDisplayAffinity(
                    hwnd, ctypes.byref(affinity)
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if int(affinity.value) != WDA_EXCLUDEFROMCAPTURE:
                    raise RuntimeError(
                        "Windows did not apply capture exclusion (affinity={})".format(
                            int(affinity.value)
                        )
                    )
                return hwnd, int(affinity.value)

            self._hwnd, self.display_affinity = configure_capture_safe(root)
            route_hwnd, _ = configure_capture_safe(route_root)
            print(
                "ARIATRACE_HUD_READY affinity={}".format(self.display_affinity),
                flush=True,
            )

            visible = False
            route_visible = False

            def refresh() -> None:
                nonlocal visible, map_photo, route_photo, route_visible
                try:
                    value = dict(self.status_provider() or {})
                    if value.get("shutdown"):
                        root.destroy()
                        return
                    should_show = self._debounced_visibility(
                        bool(value.get("visible")),
                        self._target_is_foreground(
                            user32, value.get("window_title")
                        ),
                        visible,
                    )
                    if should_show:
                        title.config(text=str(value.get("title") or "ARIATRACE"))
                        status_label.config(
                            text=str(value.get("status") or "READY"),
                            foreground=str(value.get("color") or "#59d7e8"),
                        )
                        detail.config(text=str(value.get("detail") or ""))
                        encoded_map = value.get("map_overlay_png_base64")
                        if encoded_map:
                            map_photo = tk.PhotoImage(data=encoded_map)
                            map_label.config(image=map_photo)
                            if not map_label.winfo_manager():
                                map_label.pack(fill="both", expand=True, pady=(8, 0))
                            overlay_width = MAP_HUD_WIDTH
                            overlay_height = MAP_HUD_HEIGHT
                        else:
                            map_label.pack_forget()
                            map_photo = None
                            overlay_width = HUD_WIDTH
                            overlay_height = HUD_HEIGHT
                        x, y = self._target_position(
                            user32,
                            root,
                            value.get("window_title"),
                            overlay_width,
                            overlay_height,
                        )
                        root.geometry(
                            "{}x{}+{}+{}".format(
                                overlay_width, overlay_height, x, y
                            )
                        )
                        if not visible:
                            root.deiconify()
                            visible = True
                        user32.SetWindowPos(
                            self._hwnd,
                            HWND_TOPMOST,
                            x,
                            y,
                            0,
                            0,
                            SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                        )
                        encoded_route = value.get(
                            "minimap_route_overlay_png_base64"
                        )
                        crop = value.get("minimap_crop_xywh") or ()
                        frame_size = value.get("frame_size_wh") or ()
                        client = self._target_client_rect(
                            user32, value.get("window_title")
                        )
                        route_ready = bool(
                            encoded_route
                            and len(crop) == 4
                            and len(frame_size) == 2
                            and client
                            and frame_size[0]
                            and frame_size[1]
                        )
                        if route_ready:
                            origin_x, origin_y, client_w, client_h = client
                            scale_x = client_w / float(frame_size[0])
                            scale_y = client_h / float(frame_size[1])
                            # The guide is calibrated in captured-frame pixels.
                            # Hide rather than misregister it under DPI/letterbox
                            # scaling that this lightweight HUD cannot resample.
                            route_ready = (
                                abs(scale_x - 1.0) <= 0.03
                                and abs(scale_y - 1.0) <= 0.03
                            )
                        if route_ready:
                            route_photo = tk.PhotoImage(data=encoded_route)
                            route_label.config(image=route_photo)
                            route_x = origin_x + int(round(float(crop[0]) * scale_x))
                            route_y = origin_y + int(round(float(crop[1]) * scale_y))
                            route_root.geometry(
                                "{}x{}+{}+{}".format(
                                    route_photo.width(),
                                    route_photo.height(),
                                    route_x,
                                    route_y,
                                )
                            )
                            if not route_visible:
                                route_root.deiconify()
                                route_visible = True
                            user32.SetWindowPos(
                                route_hwnd,
                                HWND_TOPMOST,
                                route_x,
                                route_y,
                                0,
                                0,
                                SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                            )
                        elif route_visible:
                            route_root.withdraw()
                            route_visible = False
                    elif visible:
                        root.withdraw()
                        visible = False
                        if route_visible:
                            route_root.withdraw()
                            route_visible = False
                except Exception as exc:
                    title.config(text="ARIATRACE HUD")
                    status_label.config(text="HUD ERROR", foreground="#ff7b84")
                    detail.config(text="{}: {}".format(type(exc).__name__, exc))
                    if visible:
                        root.withdraw()
                        visible = False
                    if route_visible:
                        route_root.withdraw()
                        route_visible = False
                root.after(self.refresh_ms, refresh)

            root.after(0, refresh)
            root.mainloop()
        except BaseException as exc:
            self._error = exc
        finally:
            self._hwnd = None
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
            if route_root is not None:
                try:
                    route_root.destroy()
                except Exception:
                    pass


class _UrlStatusProvider:
    def __init__(self, state_url: str) -> None:
        self.state_url = state_url
        self.failures = 0
        self._last_value = None
        self._media_cache = {}

    def __call__(self) -> dict:
        try:
            with urllib.request.urlopen(self.state_url, timeout=0.5) as response:
                value = json.loads(response.read().decode("utf-8"))
            for url_key, data_key in (
                ("map_overlay_url", "map_overlay_png_base64"),
                (
                    "minimap_route_overlay_url",
                    "minimap_route_overlay_png_base64",
                ),
            ):
                overlay_url = value.get(url_key)
                if not overlay_url:
                    self._media_cache.pop(data_key, None)
                    continue
                try:
                    with urllib.request.urlopen(
                        urljoin(self.state_url, str(overlay_url)), timeout=0.75
                    ) as response:
                        self._media_cache[data_key] = base64.b64encode(
                            response.read()
                        ).decode("ascii")
                except (OSError, ValueError, urllib.error.URLError):
                    # A missed refresh must not remove an already visible overlay.
                    # The next successful fetch replaces this cached frame.
                    pass
                cached = self._media_cache.get(data_key)
                if cached:
                    value[data_key] = cached
            self.failures = 0
            self._last_value = dict(value)
            return value
        except (OSError, ValueError, urllib.error.URLError):
            self.failures += 1
            if self.failures < 20 and self._last_value is not None:
                return dict(self._last_value)
            return {
                "visible": False,
                "shutdown": self.failures >= 20,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-url", required=True)
    args = parser.parse_args()
    _HudWindow(_UrlStatusProvider(args.state_url)).run_forever()


if __name__ == "__main__":
    main()
