"""Isolated E2E candidates; no policy is selected by importing this module."""

import math
import re
import shlex
import time

from swiftest import DeviceError
from examples.genshin.profile import Phone


def checked_tap(device, x, y, width, height, package, blocked_window="GameBooster Lock Screen", *, compact=False, narrow=False):
    """Perform fresh geometry, overlay and focus checks adjacent to injection."""
    if not re.fullmatch(r"[\w.]+", package):
        raise ValueError("Invalid package")
    if any(not math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError("Nonfinite input geometry")
    x, y, width, height = map(round, (x, y, width, height))
    if not (width > 0 and height > 0 and 0 <= x < width and 0 <= y < height):
        raise ValueError("Touch outside captured display")
    # All variable arguments are constrained numbers or quoted literals. The
    # current focus is read after the input-state dump, closest to injection.
    command = f"""
atk_input=$(dumpsys input) || {{ printf 'STATE_ERROR\\n'; exit; }}
atk_view=$(printf '%s\\n' "$atk_input" | grep 'Viewport INTERNAL' | head -1)
atk_overlay=$(printf '%s\\n' "$atk_input" | grep -F {shlex.quote(blocked_window)} | grep "name='" | grep -v 'NOT_VISIBLE' | grep -v 'NOT_TOUCHABLE')
atk_focus=$(dumpsys window | grep mCurrentFocus | head -1)
case "$atk_focus" in *{shlex.quote(' '+package+'/')}*) ;; *) printf 'FOCUS_CHANGED\\n'; exit ;; esac
case "$atk_view" in *{shlex.quote(f'logicalFrame=[0, 0, {width}, {height}]')}*) ;; *) printf 'DISPLAY_CHANGED\\n'; exit ;; esac
if [ -n "$atk_overlay" ]; then printf 'OVERLAY\\n'; exit; fi
if input tap {x} {y}; then printf 'SENT\\n'; else printf 'INPUT_ERROR\\n'; fi
"""
    if compact:
        awk = '''index($0,"Viewport INTERNAL") && !seen {seen=1; good=index($0,geometry)>0}
index($0,blocked) && index($0,"name=") && !index($0,"NOT_VISIBLE") && !index($0,"NOT_TOUCHABLE") {overlay=1}
END {print !good ? "DISPLAY_CHANGED" : overlay ? "OVERLAY" : "READY"}'''
        focus_command = "dumpsys window displays" if narrow else "dumpsys window"
        command = f"""
atk_gate=$(dumpsys input | awk -v geometry={shlex.quote(f'logicalFrame=[0, 0, {width}, {height}]')} -v blocked={shlex.quote(blocked_window)} {shlex.quote(awk)})
atk_focus=$({focus_command} | awk '/mCurrentFocus/ && !seen {{print; seen=1}}')
"""
        if narrow:
            command += """
if [ -z "$atk_focus" ]; then atk_focus=$(dumpsys window | awk '/mCurrentFocus/ && !seen {print; seen=1}'); fi
"""
        command += f"""
case "$atk_focus" in *{shlex.quote(' '+package+'/')}*) ;; *) printf 'FOCUS_CHANGED\\n'; exit ;; esac
if [ "$atk_gate" != READY ]; then printf '%s\\n' "$atk_gate"; exit; fi
if input tap {x} {y}; then printf 'SENT\\n'; else printf 'INPUT_ERROR\\n'; fi
"""
    response = device.run("shell", command).decode("utf-8", "replace").strip()
    if response == "OVERLAY":
        return False
    if response != "SENT":
        raise DeviceError("Guarded input withheld or failed: " + response)
    return True


class GuardedPhone(Phone):
    def __init__(self, session, *, compact=False, narrow=False):
        super().__init__(session)
        self.compact, self.narrow = compact, narrow

    def tap(self, x, y, width, height):
        # Retain the shared Session policy if the caller registered extra rules.
        if self.session.popups or self.session.busy_states:
            return super().tap(x, y, width, height)
        if not checked_tap(self.session.device, x, y, width, height, self.session.device.package,
                           compact=self.compact, narrow=self.narrow):
            if self.protection_released_at is not None and time.monotonic()-self.protection_released_at < 10:
                raise DeviceError("Samsung Game Booster touch protection still covers the game.")
            if not self.in_focus():
                raise DeviceError("Focus changed before overlay recovery")
            cx, cy = str(width//2), str(height//2)
            self.shell("input", "tap", cx, cy)
            self.shell("input", "swipe", cx, cy, str(int(width*.78)), cy, "450")
            self.protection_released_at = time.monotonic()
            return False
        self.session.stats["input_events"] += 1
        self.session.events.append({"touch": [round(x), round(y)]})
        return True
