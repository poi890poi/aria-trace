"""Optional learned-feature process. Stdin carries frames; stdout carries JSON."""
import argparse
import base64
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .destination import DestinationObserver, MODEL, MODEL_SHA256, load_package, mask_points


class LearnedFeatures:
    def __init__(self, runtime_root, device):
        import faulthandler
        faulthandler.dump_traceback_later(15, file=sys.stderr)
        import torch
        root = Path(runtime_root)
        weight = root / "model-hub/checkpoints" / (MODEL + ".pth")
        if not weight.is_file() or hashlib.sha256(weight.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Install the verified ALIKED-N16rot weights before enabling destination observation")
        sys.path.insert(0, str(root / "lightglue-upstream"))
        from lightglue import ALIKED
        torch.set_num_threads(1)
        torch.hub.set_dir(str(root / "model-hub"))
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.net = ALIKED(model_name=MODEL, max_num_keypoints=900).eval().to(self.device)
        faulthandler.cancel_dump_traceback_later()

    def extract(self, image, mask):
        import torch
        with torch.inference_mode():
            tensor = (torch.from_numpy(np.array(image)).float()[None, None] / 255).to(self.device)
            out = self.net({"image": tensor})
            points = out["keypoints"][0].cpu().numpy().astype(np.float32)
            descriptors = out["descriptors"][0].cpu().numpy().astype(np.float32)
            scores = out.get("scores", out.get("keypoint_scores"))[0].cpu().numpy().reshape(-1)
        descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
        keep = np.flatnonzero(mask_points(points, mask))
        keep = keep[np.argsort(-scores[keep], kind="stable")[:900]]
        return points[keep], descriptors[keep]


def write(value):
    sys.stdout.write(json.dumps(value, allow_nan=False) + "\n")
    sys.stdout.flush()


def exact(stream, length):
    chunks = []; remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == length:
                return None
            raise EOFError("Incomplete destination frame")
        chunks.append(chunk); remaining -= len(chunk)
    return b"".join(chunks)


def run(package, runtime_root, device):
    cv2.setNumThreads(1)
    print("Loading destination package", file=sys.stderr, flush=True)
    meta, arrays = load_package(package)
    print("Loading learned feature model", file=sys.stderr, flush=True)
    features = LearnedFeatures(runtime_root, device)
    print("Warming feature kernels", file=sys.stderr, flush=True)
    warmup = np.random.RandomState(913).randint(0, 255, (360, 640), dtype=np.uint8)
    for _ in range(2):
        features.extract(warmup, meta["mask"])
    print("Feature model and kernels ready on " + features.device, file=sys.stderr, flush=True)
    observer = DestinationObserver(meta, arrays)
    write(dict(ready=True, device=features.device, model=MODEL))
    while True:
        header = exact(sys.stdin.buffer, 8)
        if header is None:
            return
        capture_ns = struct.unpack("!Q", header)[0]
        data = exact(sys.stdin.buffer, 640 * 360)
        if data is None:
            raise EOFError("Missing destination image")
        image = np.frombuffer(data, np.uint8).reshape(360, 640)
        started = time.perf_counter(); t = capture_ns / 1e9; extraction_ms = 0.; query = None
        if observer.recognition_due(t):
            query = features.extract(image, meta["mask"])
            extraction_ms = (time.perf_counter() - started) * 1000
        result = observer.step(image, query, t)
        result.update(capture_ns=capture_ns, compute_ms=(time.perf_counter() - started) * 1000, extraction_ms=extraction_ms)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result["image_jpeg"] = base64.b64encode(encoded).decode("ascii") if ok else None
        write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        run(args.package, args.runtime_root, args.device)
    except Exception as exc:
        write(dict(error="{}: {}".format(type(exc).__name__, exc)))
        sys.exit(1)
