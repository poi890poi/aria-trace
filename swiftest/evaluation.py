"""Bounded reuse of pure visual evaluation on exactly unchanged pixels."""

from collections import OrderedDict
import numpy as np


class VisualCache:
    def __init__(self, max_bytes=16 * 1024 * 1024, max_entries=64):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def evaluate(self, key, image, compute):
        old = self.entries.pop(key, None)
        if old is not None:
            pixels, result = old
            self.bytes -= pixels.nbytes
            if pixels.dtype == image.dtype and np.array_equal(pixels, image):
                self.entries[key] = old
                self.bytes += pixels.nbytes
                self.hits += 1
                return list(result)
        self.misses += 1
        # Never cache errors, and never retain a view of a mutable capture buffer.
        result = tuple(compute())
        if image.nbytes <= self.max_bytes and self.max_entries:
            while self.entries and (self.bytes + image.nbytes > self.max_bytes
                                    or len(self.entries) >= self.max_entries):
                _, (pixels, _) = self.entries.popitem(last=False)
                self.bytes -= pixels.nbytes
            pixels = image.copy()
            self.entries[key] = (pixels, result)
            self.bytes += pixels.nbytes
        return list(result)

    def clear(self):
        self.entries.clear()
        self.bytes = 0
