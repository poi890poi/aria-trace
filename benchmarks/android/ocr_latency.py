"""Compare CPU thread counts on fixed screenshots; keep raw output outside Git."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.genshin.profile import classify, has_world_hud, ocr_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 4, 2])
    parser.add_argument("--detector-size", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images = [(p, ocr_image(cv2.imread(str(p)))) for p in args.images]
    rows = []
    for trial, threads in enumerate(args.threads):
        began = time.perf_counter()
        options = dict(det_limit_type="max", det_limit_side_len=args.detector_size) if args.detector_size else {}
        if args.max_side:
            options.update(max_side_len=args.max_side, det_limit_type="max")
        engine = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1, **options)
        initialization = time.perf_counter()-began
        for path, image in images:
            began = time.perf_counter()
            lines, stages = engine(image)
            elapsed = time.perf_counter()-began
            observation = dict(width=image.shape[1], height=image.shape[0], lines=[])
            for corners, text, confidence in lines or []:
                xs, ys = zip(*corners)
                observation["lines"].append(dict(text=text, words=[dict(
                    x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys))]))
            rows.append(dict(trial=trial, threads=threads, detector_size=args.detector_size,
                             max_side=args.max_side, path=str(path),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             seconds=elapsed, initialization_seconds=initialization,
                             stages=stages, lines=lines,
                             state=classify(observation, has_world_hud(image, observation))))
            print(json.dumps({k: rows[-1][k] for k in ("threads", "path", "seconds", "state")}), flush=True)
        del engine
        (args.output/"results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for threads in sorted(set(args.threads)):
        print(threads, statistics.median(r["seconds"] for r in rows if r["threads"] == threads))


if __name__ == "__main__":
    main()
