"""Prepared physical destination observation, independent of route/XY estimation."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "aria-destination-1"
MODEL = "aliked-n16rot"
MODEL_SHA256 = "ddf3abbf38e86f6a74540d214e1a9712c54b2d8551abc864542199f2347d7332"
MAX_AGE_S = 0.5
DEFAULT_MASK = {"include": [110, 25, 550, 275], "exclude": [[275, 155, 380, 360]]}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("Invalid destination ID")
    return value


def mask_points(points, mask):
    x, y = np.asarray(points).reshape(-1, 2).T
    x0, y0, x1, y1 = mask["include"]
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    for x0, y0, x1, y1 in mask["exclude"]:
        keep &= ~((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    return keep


def load_package(path, game_id=None):
    path = Path(path)
    meta = read_json(path / "destination.json")
    if meta.get("schema") != SCHEMA or meta.get("model") != MODEL:
        raise ValueError("Unsupported destination package or feature model")
    checked_id(meta.get("destination_id")); checked_id(meta.get("game_profile_id"))
    if game_id is not None and meta["game_profile_id"] != game_id:
        raise ValueError("Destination belongs to another game")
    if meta.get("image_size_wh") != [640, 360] or meta.get("model_sha256") != MODEL_SHA256:
        raise ValueError("Destination image geometry or model identity is incompatible")
    mask = meta.get("mask", {})
    boxes = [mask.get("include", [])] + mask.get("exclude", [])
    if not boxes or len(boxes) > 16:
        raise ValueError("Invalid destination mask")
    for box in boxes:
        if len(box) != 4 or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 360):
            raise ValueError("Invalid destination mask rectangle")
    views = meta.get("views", [])
    if not 1 <= len(views) <= 128:
        raise ValueError("Destination needs between 1 and 128 prepared views")
    bank_path = path / "bank.npz"
    with zipfile.ZipFile(bank_path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError("Destination bank exceeds 64 MiB")
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != meta.get("bank_sha256"):
        raise ValueError("Destination bank checksum mismatch")
    arrays = {}
    seen = set()
    with np.load(str(bank_path), allow_pickle=False) as bank:
        for view in views:
            key = checked_id(view.get("key"))
            if key in seen:
                raise ValueError("Duplicate destination view")
            seen.add(key)
            pp, dd, own = (bank[key + suffix] for suffix in ("p", "d", "o"))
            if pp.ndim != 2 or pp.shape[1] != 2 or not 6 <= len(pp) <= 900 or dd.shape != (len(pp), 128) or own.shape != (len(pp),) or own.dtype != np.bool_:
                raise ValueError("Invalid destination feature arrays")
            aim = np.asarray(view.get("aim"), dtype=float)
            width = float(view.get("width", 0))
            if aim.shape != (2,) or not np.isfinite(aim).all() or not np.isfinite(width) or width <= 0 or not np.isfinite(pp).all() or not np.isfinite(dd).all() or own.sum() < 6:
                raise ValueError("Invalid destination goal or feature values")
            arrays[key + "p"] = np.float32(pp)
            arrays[key + "d"] = np.float32(dd)
            arrays[key + "o"] = own
    return meta, arrays


def correspondence(reference, query):
    pp, dd = reference; qp, qd = query
    if min(len(pp), len(qp)) < 2:
        return np.empty((0, 2), int)
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(dd, qd, k=2)
    reverse = matcher.knnMatch(qd, dd, k=2)
    back = {z.queryIdx: z.trainIdx for z, w in reverse if z.distance < .75 * w.distance}
    return np.array([(z.queryIdx, z.trainIdx) for z, w in forward if z.distance < .75 * w.distance and back.get(z.trainIdx) == z.queryIdx], int).reshape(-1, 2)


def local_fit(a, b, aim):
    if len(a) < 6:
        return None
    cv2.setRNGSeed(913)
    mat, inliers = cv2.estimateAffinePartial2D(np.float32(a), np.float32(b), method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1000, confidence=.99)
    if mat is None or inliers is None or inliers.sum() < 6 or inliers.mean() < .6:
        return None
    scale = float(np.linalg.norm(mat[0, :2])); good = inliers.ravel() > 0
    target = mat @ np.r_[aim, 1.]
    if not .4 <= scale <= 3.5 or not np.isfinite(target).all():
        return None
    return dict(aim=target.tolist(), reference=np.asarray(a)[good], points=np.asarray(b)[good], support=int(good.sum()), scale=scale)


def flow(previous, current, points):
    points = np.float32(points).reshape(-1, 2)
    if not len(points):
        return points, np.zeros(0, bool)
    options = dict(winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, .01))
    q, good, _ = cv2.calcOpticalFlowPyrLK(previous, current, points.reshape(-1, 1, 2), None, **options)
    if q is None:
        return points, np.zeros(len(points), bool)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, q, None, **options)
    if back is None:
        return points, np.zeros(len(points), bool)
    q = q.reshape(-1, 2)
    keep = (good.ravel() > 0) & (reverse.ravel() > 0) & (np.linalg.norm(back.reshape(-1, 2) - points, axis=1) <= 1.5)
    keep &= (q[:, 0] >= 10) & (q[:, 0] < 630) & (q[:, 1] >= 10) & (q[:, 1] < 350)
    return q, keep


class DestinationObserver:
    """Same multi-view + LK rule as the tested observer, with explicit gap reset."""
    def __init__(self, metadata, arrays):
        self.meta = metadata; self.bank = metadata["views"]; self.arrays = arrays
        self.state = None; self.previous = None; self.previous_time = None
        self.anchor = -99.; self.attempt = -99.; self.retry = -99.; self.active = None

    def recognition_due(self, t):
        return t - self.attempt >= .2

    def _candidates(self, query, indices):
        result = []
        for i in indices:
            view = self.bank[i]; key = view["key"]
            pp, dd, own = (self.arrays[key + suffix] for suffix in ("p", "d", "o"))
            links = correspondence((pp, dd), query); links = links[own[links[:, 0]]]
            fit = local_fit(pp[links[:, 0]], query[0][links[:, 1]], view["aim"])
            if fit:
                fit.update(view=i, base_aim=view["aim"], width=view["width"] * fit["scale"])
                result.append(fit)
        return result

    def step(self, image, query, t):
        if self.previous_time is not None and (t <= self.previous_time or t - self.previous_time > MAX_AGE_S):
            self.state = None; self.previous = None
        flowed = None; chosen = None; kind = "unavailable"
        if self.state is not None and self.previous is not None and t - self.anchor <= MAX_AGE_S:
            old = self.state
            xy, good = flow(self.previous, image, old["points"])
            good &= mask_points(xy, self.meta["mask"])
            flowed = local_fit(old["reference"][good], xy[good], old["base_aim"])
            if flowed:
                flowed.update(view=old["view"], base_aim=old["base_aim"], width=self.bank[old["view"]]["width"] * flowed["scale"])
                kind = "flow"
        if query is not None and self.recognition_due(t):
            self.attempt = t
            indices = [] if self.active is None else list(range(max(0, self.active - 1), min(len(self.bank), self.active + 2)))
            candidates = self._candidates(query, indices)
            if not candidates and t >= self.retry:
                candidates = self._candidates(query, [i for i in range(len(self.bank)) if i not in indices]); self.retry = t + .5
            if candidates:
                chosen = max(candidates, key=lambda a: (a["support"], -a["view"]))
                self.anchor = t; self.active = chosen["view"]; kind = "anchor"
        self.state = chosen or flowed; self.previous = image; self.previous_time = t
        state = self.state
        return dict(state="tracking" if state else "unavailable", kind=kind, aim=None if state is None else state["aim"], bearing=None if state is None else (state["aim"][0] - 320) / 320, width=None if state is None else state["width"], support=0 if state is None else state["support"], view=None if state is None else state["view"], anchor_age_s=None if state is None else t - self.anchor, points=[] if state is None else state["points"].tolist(), travel_relation="unknown", commands_issued=0)
