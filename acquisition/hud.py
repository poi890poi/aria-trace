"""Capture-safe, always-on-top Windows HUD for the acquisition workbench."""

import argparse
import ctypes
import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional


HUD_WIDTH = 390
HUD_HEIGHT = 92
HUD_MARGIN = 22

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
    ) -> None:
        self.status_provider = status_provider
        self.refresh_ms = int(refresh_ms)
        self._error: Optional[BaseException] = None
        self._hwnd = None
        self.display_affinity = None

    def run_forever(self) -> None:
        """Run Tk on the current thread; used by the isolated HUD process."""
        if os.name != "nt":
            raise RuntimeError("The in-game HUD is available only on Windows")
        self._run()
        if self._error is not None:
            raise RuntimeError("In-game HUD failed: {}".format(self._error))

    @staticmethod
    def _target_position(user32, root, window_title: Optional[str]):
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
                    if width >= HUD_WIDTH + HUD_MARGIN * 2 and height >= HUD_HEIGHT:
                        return (
                            int(origin.x + width - HUD_WIDTH - HUD_MARGIN),
                            int(origin.y + HUD_MARGIN),
                        )
        return (
            max(0, int(root.winfo_screenwidth()) - HUD_WIDTH - HUD_MARGIN),
            HUD_MARGIN,
        )

    def _run(self) -> None:
        root = None
        try:
            import tkinter as tk
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            user32.FindWindowW.restype = wintypes.HWND
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
            root.attributes("-alpha", 0.92)
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
            root.update_idletasks()
            widget_hwnd = int(root.winfo_id())
            self._hwnd = int(
                user32.GetAncestor(widget_hwnd, GA_ROOT) or widget_hwnd
            )

            style = int(user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE))
            style |= WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, style)
            if not user32.SetWindowDisplayAffinity(
                self._hwnd, WDA_EXCLUDEFROMCAPTURE
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            affinity = wintypes.DWORD(0)
            if not user32.GetWindowDisplayAffinity(
                self._hwnd, ctypes.byref(affinity)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            self.display_affinity = int(affinity.value)
            if self.display_affinity != WDA_EXCLUDEFROMCAPTURE:
                raise RuntimeError(
                    "Windows did not apply capture exclusion (affinity={})".format(
                        self.display_affinity
                    )
                )
            print(
                "ARIATRACE_HUD_READY affinity={}".format(self.display_affinity),
                flush=True,
            )

            visible = False

            def refresh() -> None:
                nonlocal visible
                try:
                    value = dict(self.status_provider() or {})
                    if value.get("shutdown"):
                        root.destroy()
                        return
                    should_show = bool(value.get("visible"))
                    if should_show:
                        title.config(text=str(value.get("title") or "ARIATRACE"))
                        status_label.config(
                            text=str(value.get("status") or "READY"),
                            foreground=str(value.get("color") or "#59d7e8"),
                        )
                        detail.config(text=str(value.get("detail") or ""))
                        x, y = self._target_position(
                            user32, root, value.get("window_title")
                        )
                        root.geometry(
                            "{}x{}+{}+{}".format(HUD_WIDTH, HUD_HEIGHT, x, y)
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
                    elif visible:
                        root.withdraw()
                        visible = False
                except Exception as exc:
                    title.config(text="ARIATRACE HUD")
                    status_label.config(text="HUD ERROR", foreground="#ff7b84")
                    detail.config(text="{}: {}".format(type(exc).__name__, exc))
                    if not visible:
                        root.deiconify()
                        visible = True
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


class _UrlStatusProvider:
    def __init__(self, state_url: str) -> None:
        self.state_url = state_url
        self.failures = 0

    def __call__(self) -> dict:
        try:
            with urllib.request.urlopen(self.state_url, timeout=0.5) as response:
                value = json.loads(response.read().decode("utf-8"))
            self.failures = 0
            return value
        except (OSError, ValueError, urllib.error.URLError):
            self.failures += 1
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
