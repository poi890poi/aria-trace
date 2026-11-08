"""Compare complete OCR query paths on fixed images; never send device input."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest.textmatch import TextPattern
from examples.genshin.profile import ocr_image


def prepare_resource(destination):
    import shutil
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    keys = engine.text_rec.postprocess_op.character
    assert keys[0] == "blank" and keys[-1] == " "
    models = destination / "model/ocr"
    models.mkdir(parents=True, exist_ok=False)
    (models / "keys.txt").write_text("\n".join(keys[1:-1])+"\n", encoding="utf-8")
    source = Path(rapidocr_onnxruntime.__file__).parent / "models"
    for old, new in (("ch_PP-OCRv4_det_infer.onnx", "det.onnx"),
                     ("ch_PP-OCRv4_rec_infer.onnx", "rec.onnx")):
        shutil.copy2(source / old, models / new)
    (destination / "pipeline").mkdir()
    (destination / "pipeline/probe.json").write_text(json.dumps({
        "probe": {"recognition": "OCR", "action": "DoNothing"}}))
    (destination / "provenance.json").write_text(json.dumps({
        "source": "rapidocr-onnxruntime " + importlib.metadata.version("rapidocr-onnxruntime"),
        "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir()},
    }, indent=2))


class MaaReader:
    def __init__(self, resource_path, output):
        from maa.controller import CustomController
        from maa.define import LoggingLevelEnum
        from maa.resource import Resource
        from maa.tasker import Tasker

        class ReadOnlyFrame(CustomController):
            def connect(self): return True
            def request_uuid(self): return "offline-ocr-solution-probe"
            def get_features(self): return 0
            def screencap(self): raise RuntimeError("Use the explicitly supplied image.")

        Tasker.set_log_dir(output / "maa-log")
        Tasker.set_stdout_level(LoggingLevelEnum.Error)
        Tasker.set_save_draw(False)
        Tasker.set_debug_mode(False)
        self.resource = Resource()
        assert self.resource.use_cpu()
        assert self.resource.post_bundle(resource_path).wait().succeeded
        self.controller = ReadOnlyFrame()
        assert self.controller.post_connection().wait().succeeded
        self.tasker = Tasker()
        assert self.tasker.bind(self.resource, self.controller)

    def __call__(self, image, region=None):
        from maa.pipeline import JOCR
        roi = (0, 0, 0, 0) if region is None else tuple(region)
        job = self.tasker.post_recognition("OCR", JOCR(roi=roi, threshold=.5), image).wait()
        detail = job.get()
        if detail is None or not detail.nodes or detail.nodes[0].recognition is None:
            raise RuntimeError("MaaFramework returned no recognition detail.")
        # Retain ALL filtered candidates; do not silently accept Maa's first result.
        return detail.nodes[0].recognition.raw_detail["filtered"]


class TestKitReader:
    def __init__(self):
        from swiftest import LocalOCR
        self.ocr = LocalOCR(backend="openvino", threads=4, max_side=1280)
        self.ocr._initialize()

    def __call__(self, image, region=None):
        x, y, w, h = region or (0, 0, image.shape[1], image.shape[0])
        return [dict(text=text, score=box.score, box=[box.x+x, box.y+y, box.w, box.h])
                for text, box in self.ocr(image[y:y+h, x:x+w])]


class RecoveryReader:
    def __init__(self):
        from types import SimpleNamespace
        from ocr_backends import make_engine
        self.metadata = {}
        self.engine = make_engine(SimpleNamespace(engine="openvino", model="small",
                                                  threads=4, max_side=1280), self.metadata)

    def __call__(self, image, region=None):
        x, y, w, h = region or (0, 0, image.shape[1], image.shape[0])
        lines, _ = self.engine(image[y:y+h, x:x+w])
        words = []
        for corners, text, confidence in lines or ():
            xs, ys = zip(*corners)
            words.append(dict(text=text, score=float(confidence),
                              box=[min(xs)+x, min(ys)+y, max(xs)-min(xs), max(ys)-min(ys)]))
        return words


class CascadeReader(TestKitReader):
    def __init__(self):
        super().__init__()
        self.recovery = RecoveryReader()


def match(words, query, image):
    h, w = image.shape[:2]
    left, top, right, bottom = query.get("region", [0, 0, 1, 1])
    pattern = TextPattern(query["labels"])
    found = []
    for word in words:
        x, y, bw, bh = word["box"]
        if (word["score"] >= .5 and pattern.matches(word["text"])
                and left*w <= x and top*h <= y and x+bw <= right*w and y+bh <= bottom*h):
            found.append(word)
    return found


def query_image(reader, image, query, strategy):
    calls, fallback = 1, False
    if strategy == "full":
        words = reader(image)
    else:
        h, w = image.shape[:2]
        l, t, r, b = query.get("region", [0, 0, 1, 1])
        x, y = round(l*w), round(t*h)
        words = reader(image, [x, y, round(r*w)-x, round(b*h)-y])
        if strategy == "roi-fallback" and not match(words, query, image):
            words, calls, fallback = reader(image), 2, True
    found = match(words, query, image)
    route = "primary"
    # Ambiguity is terminal. A second reader cannot overrule conflicting targets.
    if not found and isinstance(reader, CascadeReader):
        words = reader.recovery(image)
        found = match(words, query, image)
        calls, fallback, route = calls+1, True, "alternate-full-frame"
    return dict(words=words, matches=found, calls=calls, fallback=fallback,
                route=route,
                outcome="missing" if not found else "unique" if len(found) == 1 else "ambiguous")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--engine", choices=["testkit", "maa-cpu", "recovery", "cascade"])
    parser.add_argument("--strategy", choices=["full", "roi", "roi-fallback"], default="full")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resource", type=Path, default=ROOT / "artifacts/ocr-solutions/resource")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.prepare:
        prepare_resource(args.prepare)
        return
    if not all((args.engine, args.manifest, args.output)):
        parser.error("engine, manifest and output are required for a probe")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = [ocr_image(cv2.imread(case["image"])) for case in cases]
    start = time.perf_counter()
    factories = {"testkit": TestKitReader, "recovery": RecoveryReader, "cascade": CascadeReader}
    reader = factories[args.engine]() if args.engine in factories else MaaReader(
        args.resource, args.output)
    metadata = dict(engine=args.engine, strategy=args.strategy, device_policy="CPU-only",
                    constructor_seconds=time.perf_counter()-start,
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    model_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (args.resource / "model/ocr").iterdir()}, packages={})
    if args.engine in ("cascade", "recovery"):
        metadata["recovery_model_hashes"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "artifacts/android-ocr/models").rglob("*small*.onnx")}
    for name in ("maafw", "rapidocr", "rapidocr-onnxruntime", "openvino", "opencv-python", "numpy"):
        try: metadata["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for repeat in range(args.rounds):
        for case, image in zip(cases, images):
            start = time.perf_counter()
            result = query_image(reader, image, case["query"], args.strategy)
            elapsed = time.perf_counter()-start
            passed = result["outcome"] == case["expected"]
            if passed and case.get("target_box") and result["outcome"] == "unique":
                x, y, w, h = result["matches"][0]["box"]
                tx, ty, tw, th = case["target_box"]
                passed = tx <= x+w/2 <= tx+tw and ty <= y+h/2 <= ty+th
            row = dict(case=case["name"], repeat=repeat, seconds=elapsed, passed=passed,
                       expected=case["expected"], image_sha256=hashlib.sha256(Path(case["image"]).read_bytes()).hexdigest(), **result)
            rows.append(row)
            (args.output / "results.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: row[k] for k in ("case", "repeat", "seconds", "passed", "outcome", "fallback")}), flush=True)
    warm = [r for r in rows if r["repeat"] > 0]
    summary = dict(passed=sum(r["passed"] for r in rows), total=len(rows),
                   warm_median_seconds=statistics.median(r["seconds"] for r in warm),
                   warm_p95_seconds=statistics.quantiles([r["seconds"] for r in warm], n=20, method="inclusive")[-1],
                   failures=sorted({r["case"] for r in rows if not r["passed"]}))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
