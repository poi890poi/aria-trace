"""Fresh display, overlay and focus checks beside an on-device tap."""

import math
import re
import shlex

from .device import DeviceError


def checked_tap(device, x, y, width, height, package, blocked_window):
    """Return True if sent, False for an overlay, None if awk is unavailable.

    None is established before any input. Every other failure is terminal:
    callers must never replay an input with a lost acknowledgement.
    """
    if not isinstance(package, str) or not re.fullmatch(r"[\w.]+", package):
        raise ValueError("Invalid package")
    if any(not math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError("Nonfinite input geometry")
    x, y, width, height = map(round, (x, y, width, height))
    if not (width > 0 and height > 0 and 0 <= x < width and 0 <= y < height):
        raise ValueError("Touch outside captured display")
    awk = '''index($0,"Viewport INTERNAL") && !seen {seen=1; good=index($0,geometry)>0}
index($0,blocked) && index($0,"name=") && !index($0,"NOT_VISIBLE") && !index($0,"NOT_TOUCHABLE") {overlay=1}
END {print !good ? "DISPLAY_CHANGED" : overlay ? "OVERLAY" : "READY"}'''
    # Consume both dumps fully: exiting awk early can produce a broken-pipe
    # diagnostic on Android. Read focus last, closest to the actual injection.
    command = f"""
command -v awk >/dev/null 2>&1 || {{ printf 'UNSUPPORTED\\n'; exit; }}
atk_gate=$(dumpsys input | awk -v geometry={shlex.quote(f'logicalFrame=[0, 0, {width}, {height}]')} -v blocked={shlex.quote(blocked_window)} {shlex.quote(awk)})
atk_focus=$(dumpsys window | awk '/mCurrentFocus/ && !seen {{print; seen=1}}')
case "$atk_focus" in *{shlex.quote(' '+package+'/')}*) ;; *) printf 'FOCUS_CHANGED\\n'; exit ;; esac
if [ "$atk_gate" != READY ]; then printf '%s\\n' "$atk_gate"; exit; fi
if input tap {x} {y}; then printf 'SENT\\n'; else printf 'INPUT_ERROR\\n'; fi
"""
    response = device.run("shell", command).decode("utf-8", "replace").strip()
    if response == "UNSUPPORTED":
        return None
    if response == "OVERLAY":
        return False
    if response != "SENT":
        raise DeviceError("Guarded input withheld or failed: " + response)
    return True
