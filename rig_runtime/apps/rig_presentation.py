"""Presentation-only styling shared by IRIS rig command and GUI surfaces."""

from __future__ import annotations

import builtins
import os
import sys
from typing import Any, Optional, TextIO


_RESET = "\x1b[0m"
_ROLE_CODES = {
    "action": "\x1b[1;96m",
    "error": "\x1b[1;91m",
    "stage": "\x1b[1;96m",
    "success": "\x1b[1;92m",
    "warning": "\x1b[1;93m",
}
_ROLE_PREFIXES = {
    "action": "[ACTION] ",
    "error": "[ERROR] ",
    "stage": "[STEP] ",
    "success": "[OK] ",
    "warning": "[WARN] ",
}
_WINDOWS_VT_ENABLED = False
_WINDOWS_VT_ATTEMPTED = False


def message_role(message: str) -> Optional[str]:
    """Choose a concise visual and textual status cue for a rig message."""

    value = str(message).strip().lower()
    if not value or value.startswith(("{", "[")):
        return None
    if value.startswith("rig repeatability policy:"):
        # This is a configuration summary.  It describes the threshold at
        # which a future save would be blocked; it is not a blocked operation.
        return None
    if (
        "warning" in value
        or value.startswith(
            (
                "display wake warning:",
                "rig reuse was skipped:",
                "calibration ended without saving.",
            )
        )
    ):
        return "warning"
    if any(
        token in value
        for token in (
            " failed",
            " failure",
            " blocked",
            " disabled after ",
            " rejected",
            " cancelled",
            " could not ",
            " cannot ",
            " did not ",
        )
    ) or value.startswith(("failure", "error", "cannot", "save blocked")):
        return "error"
    if any(
        token in value
        for token in (
            " succeeded",
            " complete",
            " completed",
            " confirmed",
            " configured",
            " ready",
            " saved",
            " skipped",
        )
    ) or value.startswith(("selected ", "saved ", "complete ")):
        return "success"
    if value.startswith(("focus:", "positioning pause:")):
        return "action"
    if value.startswith(
        (
            "benchmarking ",
            "calibrating ",
            "checking ",
            "reading ",
            "running ",
            "showing ",
            "full rig calibration ",
        )
    ):
        return "stage"
    return None


def _enable_windows_virtual_terminal(stream: TextIO) -> bool:
    global _WINDOWS_VT_ATTEMPTED, _WINDOWS_VT_ENABLED
    if os.name != "nt":
        return True
    if _WINDOWS_VT_ATTEMPTED:
        return _WINDOWS_VT_ENABLED
    _WINDOWS_VT_ATTEMPTED = True
    try:
        import ctypes

        handle_number = -12 if stream is sys.stderr else -11
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(handle_number)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        _WINDOWS_VT_ENABLED = bool(
            kernel32.SetConsoleMode(handle, int(mode.value) | 0x0004)
        )
    except (AttributeError, OSError, ValueError):
        _WINDOWS_VT_ENABLED = False
    return _WINDOWS_VT_ENABLED


def console_styling_enabled(stream: TextIO) -> bool:
    """Return true only for a terminal that can safely consume VT styling."""

    if "NO_COLOR" in os.environ or os.environ.get("TERM", "").lower() == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not bool(isatty()):
        return False
    return _enable_windows_virtual_terminal(stream)


def styled_console_text(
    message: str,
    role: Optional[str] = None,
    enabled: bool = True,
) -> str:
    """Add a semantic cue and terminal attributes while retaining the message."""

    selected = role or message_role(message)
    code = _ROLE_CODES.get(str(selected)) if selected else None
    prefix = _ROLE_PREFIXES.get(str(selected), "") if selected else ""
    return (
        "{}{}{}{}".format(code, prefix, message, _RESET)
        if enabled and code
        else message
    )


def console_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[TextIO] = None,
    flush: bool = False,
    role: Optional[str] = None,
) -> None:
    """Print a rig message with interactive-terminal cues when supported."""

    stream = file or sys.stdout
    message = sep.join(str(value) for value in values)
    rendered = styled_console_text(
        message,
        role=role,
        enabled=console_styling_enabled(stream),
    )
    builtins.print(rendered, end=end, file=stream, flush=flush)


RIG_CALIBRATOR_STYLE_SHEET = """
QMainWindow {
    font-size: 10pt;
}
QLabel#rigStatus {
    background: palette(base);
    color: palette(text);
    border: 1px solid palette(mid);
    border-left: 5px solid palette(highlight);
    border-radius: 4px;
    padding: 11px 13px;
    font-weight: 600;
}
QLabel#rigStatus[messageKind="error"] {
    border: 2px solid palette(highlight);
    border-left-width: 6px;
}
QGroupBox {
    font-weight: 600;
    margin-top: 11px;
    padding-top: 7px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 4px;
}
QPushButton {
    min-height: 28px;
    padding: 3px 9px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 25px;
}
QTextEdit#rigResults {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10pt;
}
"""
