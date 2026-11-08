"""Optional named-screen transition rules for application-specific workflows."""

import json
import math
import time
import re
import unicodedata
from pathlib import Path
from typing import NamedTuple


class Rule(NamedTuple):
    ready: str
    result: str
    retry_safe: bool = False
    require_loading: bool = False


def load_intent(path, rules):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(data, dict)
        or set(data) != {"open", "touch"}
        or not isinstance(data["touch"], list)
        or not data["touch"]
    ):
        raise ValueError("An intent needs an app to open and a nonempty touch list.")
    for step in data["touch"]:
        if not isinstance(step, dict) or set(step) - {
            "element",
            "position",
            "template",
        }:
            raise ValueError("Touch targets accept element, position or template.")
        if (
            not isinstance(step.get("element"), str)
            or step["element"] not in rules
            or ("position" in step and "template" in step)
        ):
            raise ValueError("Unknown element or conflicting selectors.")
        if "position" in step:
            p = step["position"]
            if (
                not isinstance(p, list)
                or len(p) != 2
                or not all(
                    type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                    for v in p
                )
            ):
                raise ValueError("Position requires two normalized coordinates.")
        if "template" in step:
            if not isinstance(step["template"], str):
                raise ValueError("Template must be an image path.")
            file = (path.parent / step["template"]).resolve()
            if not file.is_file():
                raise ValueError("Template file does not exist: " + str(file))
            step["template"] = str(file)
    return data


def point_for(step, image, default):
    """Resolve the optional user override on the already captured frame."""
    from types import SimpleNamespace
    from .targets import Box, Position, Template

    h, w = image.shape[:2]
    obs = SimpleNamespace(
        image=image, screen=Box(0, 0, w, h), viewport=Box(0, 0, w, h), base=Path.cwd()
    )
    if "position" in step:
        return Position(*step["position"]).matches(obs)[0].center
    if "template" not in step:
        return default()
    target = Template(step["template"])
    matches = target.matches(obs)
    return target.point(matches[0]) if len(matches) == 1 else None


class TouchSequence:
    """One observation can confirm a previous touch and resolve the next target."""

    def __init__(self, steps, rules, transition_timeout):
        self.steps = steps
        self.rules = rules
        self.timeout = transition_timeout
        self.index = 0
        self.sent = None
        self.attempts = 0
        self.loading = False
        self.arrivals = 0
        self.missing_since = None

    def advance(self, state, loading, resolve, tap, now=None):
        now = time.monotonic() if now is None else now
        while self.index < len(self.steps):
            step = self.steps[self.index]
            rule = self.rules[step["element"]]
            if self.sent is not None:
                self.loading |= loading
                if rule.require_loading:
                    self.arrivals = (
                        self.arrivals + 1
                        if state == rule.result and self.loading
                        else 0
                    )
                    confirmed = self.arrivals >= 2
                else:
                    confirmed = state == rule.result
                if confirmed:
                    self.index += 1
                    self.sent = None
                    self.attempts = self.arrivals = 0
                    self.loading = False
                    continue
                transition_timeout = (
                    self.timeout if rule.require_loading else min(self.timeout, 20)
                )
                if now - self.sent > transition_timeout:
                    raise RuntimeError(
                        "Could not verify " + step["element"] + " transition."
                    )
                if not (
                    rule.retry_safe
                    and state == rule.ready
                    and self.attempts < 2
                    and now - self.sent >= 6
                ):
                    return False
            elif rule.retry_safe and state == rule.result:
                # The map may already be open when the script starts.
                self.index += 1
                continue
            if state == rule.ready:
                point = resolve(step)
                if point is not None:
                    self.missing_since = None
                    if tap(point, step["element"]) is not False:
                        self.sent = now
                        self.attempts += 1
                elif self.missing_since is None:
                    self.missing_since = now
                elif now - self.missing_since >= min(20, self.timeout):
                    raise RuntimeError(
                        "No unique visible target for " + step["element"] + "."
                    )
            else:
                self.missing_since = None
            return False
        return True


def interval(state, repeats, pending, base=2.0, *, max_interactive=None):
    if max_interactive is not None and (
            not math.isfinite(max_interactive) or max_interactive <= 0):
        raise ValueError("Interactive polling cap must be finite and positive.")
    if max_interactive is not None and state not in ("downloading", "preparing"):
        return min(max_interactive, interval(state, repeats, pending, base))
    if pending:
        return min(3.0, 0.25 * 2 ** min(4, repeats))
    if state in ("downloading", "preparing"):
        return min(20.0, base * (2 ** min(4, repeats)))
    if state == "unknown":
        return min(6.0, base * (1.5 ** min(4, repeats)))
    return min(3.0, 0.5 * 1.5 ** min(5, repeats))


def download_delay(lines, fallback):
    text = (
        unicodedata.normalize("NFKC", "".join(lines))
        .lower()
        .replace(" ", "")
        .replace(",", ".")
    )
    rate = re.search(r"(\d+(?:\.\d+)?)kb/s", text)
    size = re.search(r"(\d+(?:\.\d+)?)mb/(\d+(?:\.\d+)?)mb", text)
    if not rate or not size:
        return fallback
    speed = float(rate[1])
    remaining = float(size[2]) - float(size[1])
    if speed <= 0 or remaining <= 0:
        return fallback
    # Recheck well before the estimated end; always tolerate OCR or speed errors.
    return min(60, max(1, remaining * 1024 / speed / 4))
