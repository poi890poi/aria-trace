"""Slow scene-context retrieval for a prepared route; never a steering error."""
import cv2
import numpy as np
from .destination import correspondence


def descriptor_summary(descriptors, centers):
    if not len(descriptors):
        return np.zeros(centers.size, np.float32)
    nearest = np.argmax(descriptors @ centers.T, axis=1)
    out = np.zeros_like(centers)
    for i in range(len(centers)):
        use = descriptors[nearest == i]
        if len(use):
            out[i] = use.sum(axis=0) - len(use) * centers[i]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    out = (np.sign(out) * np.sqrt(abs(out))).ravel()
    return out / max(np.linalg.norm(out), 1e-8)


def scene_support(reference, query):
    links = correspondence(reference, query)
    if len(links) < 12:
        return 0
    a = reference[0][links[:, 0]]; b = query[0][links[:, 1]]
    cv2.setRNGSeed(914)
    h, mask = cv2.findHomography(a, b, cv2.RANSAC, 3., maxIters=1000, confidence=.99)
    if h is None or mask is None:
        return 0
    good = mask.ravel() > 0; b = b[good]
    if len(b) < 12 or good.mean() < .4:
        return 0
    span = np.ptp(b, axis=0)
    cells = set((int(p[0] // 110), int(p[1] // 85)) for p in b)
    return len(b) if span[0] >= 100 and span[1] >= 45 and len(cells) >= 3 else 0


class RouteContext:
    def __init__(self, meta, arrays):
        self.views = meta['views']; self.arrays = arrays
        self.signatures = arrays['signatures']; self.centers = arrays['centers']
        self.phase = None; self.last_fresh = -999.; self.retry = -999.

    def step(self, query, capture_s):
        order = np.argsort(-(self.signatures @ descriptor_summary(query[1], self.centers)))
        if self.phase is not None and capture_s - self.last_fresh < 1:
            indices = [int(i) for i in order if -.055 <= self.views[i]['phase'] - self.phase <= .13][:8]
        else:
            indices = [int(i) for i in order[:8]]
        candidates = []

        def check(indices):
            for i in indices:
                key = self.views[i]['key']
                count = scene_support((self.arrays[key+'p'], self.arrays[key+'d']), query)
                if count:
                    candidates.append((count, i))

        check(indices)
        if not candidates and self.phase is not None and capture_s >= self.retry:
            self.retry = capture_s + .5
            check([int(i) for i in order[:8] if i not in indices])
        # Stable ordering preserves the prototype's tie behavior.
        candidates.sort(key=lambda item: -item[0])
        if not candidates:
            return dict(view=None, capture_s=capture_s, support=0)
        support, index = candidates[0]
        self.phase = self.views[index]['phase']; self.last_fresh = capture_s
        return dict(view=index, capture_s=capture_s, support=support)
