"""Selectors return screen-space boxes. Scoped children use their parent's bounds."""

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import re

import cv2
import numpy as np
from .textmatch import TextPattern


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    input_point: tuple = None

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, other):
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.x + other.w <= self.x + self.w
            and other.y + other.h <= self.y + self.h
        )


def unit(values):
    if not all(math.isfinite(n) and 0 <= n <= 1 for n in values):
        raise ValueError("Normalized coordinates must be in [0,1].")


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def __post_init__(self):
        unit((self.x, self.y))

    def matches(self, observation, scope=None):
        area = scope or observation.viewport
        return [
            Box(
                area.x + round(self.x * (area.w - 1)),
                area.y + round(self.y * (area.h - 1)),
                1,
                1,
            )
        ]


@dataclass(frozen=True)
class Pixel:
    x: int
    y: int

    def matches(self, observation, scope=None):
        area = scope or observation.viewport
        box = Box(area.x + self.x, area.y + self.y, 1, 1)
        return [box] if area.contains(box) else []


@dataclass(frozen=True)
class Node:
    resource_id: str = None
    text: str = None
    description: str = None
    class_name: str = None
    enabled: bool = True

    def __post_init__(self):
        if all(
            v is None
            for v in (self.resource_id, self.text, self.description, self.class_name)
        ):
            raise ValueError("Specify at least one node attribute.")
        for value in (self.resource_id, self.text, self.description, self.class_name):
            if value is None or isinstance(value, TextPattern):
                continue
            values = value if isinstance(value, (tuple, list)) else [value]
            if not values or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("Node attributes need nonempty strings, aliases or TextPattern.")

    def matches(self, observation, scope=None):
        filters = {
            "resource-id": self.resource_id,
            "text": self.text,
            "content-desc": self.description,
            "class": self.class_name,
        }
        found = []
        for node in observation.tree.iter("node"):
            if self.enabled and node.get("enabled") == "false":
                continue
            if any(
                value is not None and not (
                    value.matches(node.get(key, "")) if isinstance(value, TextPattern)
                    else node.get(key) in value if isinstance(value, (tuple, list))
                    else node.get(key) == value)
                for key, value in filters.items()
            ):
                continue
            bounds = re.fullmatch(
                r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", node.get("bounds", "")
            )
            if bounds:
                x, y, right, bottom = map(int, bounds.groups())
                box = Box(x, y, right - x, bottom - y)
                if (
                    box.w > 0
                    and box.h > 0
                    and x >= 0
                    and y >= 0
                    and (scope is None or scope.contains(box))
                ):
                    found.append(box)
        return found


class Text:
    """Find a line of rendered text using the configured local OCR engine."""

    def __init__(self, value=(), exact=True, *, regex=None, max_edits=0,
                 region=None, min_confidence=0.0, padding=8):
        if isinstance(value, TextPattern) and (regex is not None or max_edits):
            raise ValueError("Configure regex and edits on the supplied TextPattern.")
        self.pattern = value if isinstance(value, TextPattern) else TextPattern(
            value, regex=regex, max_edits=max_edits)
        if not exact and self.pattern.max_edits:
            raise ValueError("Fuzzy text matching must match the whole label.")
        if region is not None:
            unit(region)
            if len(region) != 4 or not (region[0] < region[2] and region[1] < region[3]):
                raise ValueError("Empty text region.")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("OCR confidence must be in [0,1].")
        self.region, self.min_confidence = region, min_confidence
        self.exact = exact
        if type(padding) is not int or padding < 0:
            raise ValueError("Text crop padding must be a nonnegative integer.")
        self.padding = padding

    def matches(self, observation, scope=None):
        if self.pattern.max_edits and scope is None and self.region is None:
            raise ValueError("Fuzzy OCR requires a region or a parent dialog.")
        if self.region is not None:
            parent = scope or observation.screen
            l, t, r, b = self.region
            x, y = round(l*parent.w), round(t*parent.h)
            scope = Box(parent.x+x, parent.y+y, round(r*parent.w)-x, round(b*parent.h)-y)
        result = []
        words = (observation.words_in(scope, self.padding) if hasattr(observation, "words_in")
                 else observation.words)
        for value, box in words:
            if self.pattern.matches(value, self.exact) and (
                box.score >= self.min_confidence and (scope is None or scope.contains(box))
            ):
                result.append(box)
        return result


def overlap(a, b):
    w = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    h = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return w * h / min(a.w * a.h, b.w * b.h)


@lru_cache(maxsize=64)
def _asset(path, stamp, flag):
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if result is None:
        raise ValueError(f"Invalid image {path}")
    return result


def asset(obs, filename, flag):
    path = (obs.base / filename).resolve()
    return _asset(str(path), path.stat().st_mtime_ns, flag)


class Template:
    def __init__(
        self,
        filename,
        *,
        region=None,
        scales=(1.0,),
        mask=None,
        threshold=0.90,
        expected=None,
        tolerance=(0, 0),
        size=None,
        anchor=(0.5, 0.5),
        scale_range=None,
    ):
        """region is normalized (left,top,right,bottom); size is pixel (minW,minH,maxW,maxH)."""
        self.filename, self.mask = filename, mask
        if scale_range is not None:
            low, high, step = scale_range
            if (
                not all(math.isfinite(n) for n in scale_range)
                or not (0 < low <= high and step > 0)
                or (high - low) / step > 128
            ):
                raise ValueError(
                    "Use a finite positive scale range with at most 129 steps."
                )
            scales = [
                min(high, low + i * step)
                for i in range(math.ceil((high - low) / step) + 1)
            ]
        self.scales = tuple(scales)
        if not self.scales or any(not math.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("Scales must be finite and positive.")
        self.region, self.expected, self.tolerance, self.size, self.anchor = (
            region,
            expected,
            tolerance,
            size,
            anchor,
        )
        self.threshold = {"strict": 0.95, "relaxed": 0.82}.get(threshold, threshold)
        if not 0 < self.threshold <= 1:
            raise ValueError("Threshold must be in (0,1].")
        unit(anchor)
        unit(tolerance)
        if expected is not None:
            unit(expected)
        if region is not None:
            unit(region)
            if not (region[0] < region[2] and region[1] < region[3]):
                raise ValueError("Empty template region.")
        if size and not (0 < size[0] <= size[2] and 0 < size[1] <= size[3]):
            raise ValueError("Invalid size limits.")

    def point(self, box):
        return box.x + int((box.w - 1) * self.anchor[0] + 0.5), box.y + int(
            (box.h - 1) * self.anchor[1] + 0.5
        )

    def matches(self, obs, scope=None):
        cache = getattr(getattr(obs, "session", None), "_visual_cache", None)
        if cache is None:
            return self._matches_uncached(obs, scope)
        image = obs.image
        parent = scope or obs.screen
        l, t, r, b = self.region or (0, 0, 1, 1)
        x0, y0 = parent.x + round(l * parent.w), parent.y + round(t * parent.h)
        x1, y1 = parent.x + round(r * parent.w), parent.y + round(b * parent.h)
        files = []
        for filename in (self.filename, self.mask):
            if filename is not None:
                path = (obs.base / filename).resolve()
                stat = path.stat()
                files.append((str(path), stat.st_mtime_ns, stat.st_size))
        key = ("template", type(self), tuple(files), image.shape,
               parent.x, parent.y, parent.w, parent.h, x0, y0, x1, y1,
               tuple(self.scales), self.threshold, tuple(self.expected or ()),
               tuple(self.tolerance), tuple(self.size or ()), tuple(self.anchor),
               getattr(self, "color", True))
        return cache.evaluate(key, image[y0:y1, x0:x1], lambda: self._matches_uncached(obs, scope))

    def _matches_uncached(self, obs, scope=None):
        parent = scope or obs.screen
        l, t, r, b = self.region or (0, 0, 1, 1)
        x0, y0 = parent.x + round(l * parent.w), parent.y + round(t * parent.h)
        x1, y1 = parent.x + round(r * parent.w), parent.y + round(b * parent.h)
        source = asset(obs, self.filename, cv2.IMREAD_COLOR)
        mask = asset(obs, self.mask, cv2.IMREAD_GRAYSCALE) if self.mask else None
        if mask is not None and (mask.shape != source.shape[:2] or not np.any(mask)):
            raise ValueError("Mask must match the template and include some pixels.")
        peaks = []
        for scale in self.scales:
            w, h = (
                max(1, round(source.shape[1] * scale)),
                max(1, round(source.shape[0] * scale)),
            )
            if (
                w > x1 - x0
                or h > y1 - y0
                or (
                    self.size
                    and not (
                        self.size[0] <= w <= self.size[2]
                        and self.size[1] <= h <= self.size[3]
                    )
                )
            ):
                continue
            sx, sy, ex, ey = x0, y0, x1, y1
            if self.expected is not None:
                cx = parent.x + self.expected[0] * (parent.w - 1)
                cy = parent.y + self.expected[1] * (parent.h - 1)
                dx, dy = self.tolerance[0] * parent.w, self.tolerance[1] * parent.h
                sx, ex = (
                    max(sx, math.ceil(cx - dx - w // 2)),
                    min(ex, math.floor(cx + dx - w // 2) + w),
                )
                sy, ey = (
                    max(sy, math.ceil(cy - dy - h // 2)),
                    min(ey, math.floor(cy + dy - h // 2) + h),
                )
            if ex - sx < w or ey - sy < h:
                continue
            target = cv2.resize(source, (w, h))
            binary = (
                None
                if mask is None
                else (
                    cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                ).astype(np.uint8)
            )
            crop = obs.image[sy:ey, sx:ex]
            channels = 3
            if not getattr(self, "color", True):
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                channels = 1
            active = (
                target.reshape(-1, channels)
                if binary is None
                else target[binary.astype(bool)].reshape(-1, channels)
            )
            if not len(active):
                continue
            if np.max(np.std(active, axis=0)) < 1:
                sq = cv2.matchTemplate(crop, target, cv2.TM_SQDIFF, mask=binary)
                scores = 1 - np.sqrt(np.maximum(sq, 0) / (len(active) * channels)) / 255
            else:
                scores = cv2.matchTemplate(
                    crop, target, cv2.TM_CCOEFF_NORMED, mask=binary
                )
            scores[~np.isfinite(scores)] = -1
            while True:
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if score < self.threshold:
                    break
                peaks.append(Box(sx + x, sy + y, w, h, score))
                if len(peaks) > 512:
                    raise ValueError("Too many candidates; narrow the template region.")
                scores[
                    max(0, y - h // 2) : y + h // 2 + 1,
                    max(0, x - w // 2) : x + w // 2 + 1,
                ] = -1
        kept = []
        for box in sorted(peaks, key=lambda item: -item.score):
            if not any(overlap(box, other) > 0.5 for other in kept):
                kept.append(box)
        return kept


@dataclass(frozen=True)
class Dialog:
    parent: object
    child: object

    @property
    def condition_only(self):
        return any(
            getattr(t, "condition_only", False) for t in (self.parent, self.child)
        )

    def matches(self, obs, scope=None):
        parents = self.parent.matches(obs, scope)
        if len(parents) != 1:
            return []
        return self.child.matches(obs, parents[0])

    def point(self, box):
        return self.child.point(box) if hasattr(self.child, "point") else box.center


@dataclass(frozen=True)
class Absent:
    target: object
    condition_only = True

    def matches(self, obs, scope=None):
        return [] if self.target.matches(obs, scope) else [Box(0, 0, 1, 1)]


class AnyOf:
    """Resolve the first available alternative, such as a localized resource label."""

    def __init__(self, *alternatives):
        if not alternatives:
            raise ValueError("Provide at least one alternative.")
        self.alternatives = alternatives

    @property
    def condition_only(self):
        return any(getattr(t, "condition_only", False) for t in self.alternatives)

    def point(self, box):
        return box.input_point or box.center

    def matches(self, obs, scope=None):
        for alternative in self.alternatives:
            boxes = alternative.matches(obs, scope)
            if boxes:
                if hasattr(alternative, "point"):
                    return [replace(b, input_point=alternative.point(b)) for b in boxes]
                return boxes
        return []


class Variants(AnyOf):
    """Union of equivalent selectors; distinct visible controls remain ambiguous.

    All alternatives share one observation. Only almost-identical boxes with
    compatible input points are coalesced; a containing dialog is not a button.
    """

    def matches(self, obs, scope=None):
        kept = []
        for alternative in self.alternatives:
            for box in alternative.matches(obs, scope):
                point = alternative.point(box) if hasattr(alternative, "point") else box.center
                box = replace(box, input_point=point)
                duplicate = False
                for other in kept:
                    iw = max(0, min(box.x+box.w, other.x+other.w)-max(box.x, other.x))
                    ih = max(0, min(box.y+box.h, other.y+other.h)-max(box.y, other.y))
                    intersection = iw*ih
                    union = box.w*box.h + other.w*other.h - intersection
                    same_point = max(abs(a-b) for a, b in zip(point, other.input_point)) <= 2
                    if union > 0 and intersection/union >= .8 and same_point:
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(box)
        return kept
