"""CPU-only fixed-input OCR experiment. Historical GPU results remain archived."""
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
from examples.genshin.profile import classify, has_world_hud, ocr_image, entry_button, matching_line


def make_engine(args, metadata):
    if args.engine not in ("legacy", "legacy-openvino", "legacy-no-spin", "legacy-arena", "onnx", "openvino"):
        raise ValueError("OCR experiments are CPU-only; GPU backends are excluded.")
    metadata["device_policy"] = "CPU-only"
    if args.engine.startswith("legacy"):
        from rapidocr_onnxruntime import RapidOCR
        from rapidocr_onnxruntime.utils import infer_engine
        original = infer_engine.InferenceSession
        metadata["sessions"] = []
        def create(path, sess_options, providers):
            if args.engine == "legacy-no-spin":
                sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            if args.engine == "legacy-arena":
                sess_options.enable_cpu_mem_arena = True
            session = original(path, sess_options=sess_options, providers=["CPUExecutionProvider"])
            metadata["sessions"].append(dict(model=Path(path).name, providers=session.get_providers(),
                                             options=session.get_provider_options()))
            return session
        infer_engine.InferenceSession = create
        options = dict(intra_op_num_threads=args.threads, inter_op_num_threads=1)
        if args.max_side:
            options.update(max_side_len=args.max_side, det_limit_type="max")
        engine = RapidOCR(**options)
        if args.engine == "legacy-openvino":
            from swiftest.ocr_acceleration import accelerate
            engine = accelerate(engine, args.threads)
            metadata["active_backend"] = "openvino"
        return engine
    from rapidocr import RapidOCR, EngineType, OCRVersion, ModelType
    params = {"Global.log_level": "warning", "Global.model_root_dir": str(ROOT/"artifacts/android-ocr/models"),
              "EngineConfig.onnxruntime.intra_op_num_threads": args.threads,
              "EngineConfig.onnxruntime.inter_op_num_threads": 1,
              "EngineConfig.onnxruntime.use_cuda": False,
              "EngineConfig.onnxruntime.use_dml": False,
              "EngineConfig.onnxruntime.use_cann": False,
              "EngineConfig.onnxruntime.use_coreml": False,
              "EngineConfig.openvino.inference_num_threads": args.threads,
              "EngineConfig.openvino.performance_hint": "LATENCY",
              "EngineConfig.openvino.num_streams": 1}
    if args.max_side:
        params.update({"Global.max_side_len": args.max_side, "Det.limit_type": "max",
                       "Det.limit_side_len": args.max_side})
    old_models = ROOT/".venv/androidtestkit/Lib/site-packages/rapidocr_onnxruntime/models"
    files = dict(Det="ch_PP-OCRv4_det_infer.onnx", Cls="ch_ppocr_mobile_v2.0_cls_infer.onnx",
                 Rec="ch_PP-OCRv4_rec_infer.onnx")
    for stage in ("Det", "Cls", "Rec"):
        params[stage+".engine_type"] = EngineType.OPENVINO if args.engine == "openvino" else EngineType.ONNXRUNTIME
        if args.model == "v4" or stage == "Cls":
            params.update({stage+".ocr_version": OCRVersion.PPOCRV4, stage+".model_type": ModelType.MOBILE,
                           stage+".model_path": str(old_models/files[stage])})
        else:
            params.update({stage+".ocr_version": OCRVersion.PPOCRV6, stage+".model_type": ModelType(args.model)})
    metadata["params"] = {k: str(v) for k,v in params.items()}
    modern = RapidOCR(params=params)
    def engine(image):
        result = modern(image)
        lines = [] if result.boxes is None else [[box.tolist(), str(text), float(score)]
                  for box,text,score in zip(result.boxes, result.txts, result.scores)]
        return lines, result.elapse_list
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["legacy", "legacy-openvino", "legacy-no-spin", "legacy-arena", "onnx", "openvino"], required=True)
    parser.add_argument("--model", choices=["v4", "tiny", "small"], default="v4")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(configuration={k:str(v) for k,v in vars(args).items()}, packages={})
    for name in ("rapidocr", "rapidocr-onnxruntime", "onnxruntime", "onnxruntime-directml", "openvino", "numpy", "opencv-python", "opencv-python-headless"):
        try:
            metadata["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    images = [(p, ocr_image(cv2.imread(str(p)))) for p in args.images]
    start = time.perf_counter()
    engine = make_engine(args, metadata)
    metadata["construction_seconds"] = time.perf_counter()-start
    rows = []
    for repeat in range(args.rounds):
        for path, image in images:
            start = time.perf_counter()
            lines, stages = engine(image)
            elapsed = time.perf_counter()-start
            observation = dict(width=image.shape[1], height=image.shape[0], lines=[])
            for corners, text, confidence in lines or []:
                xs, ys = zip(*corners)
                observation["lines"].append(dict(text=text, words=[dict(x=min(xs), y=min(ys),
                    width=max(xs)-min(xs), height=max(ys)-min(ys))]))
            state = classify(observation, has_world_hud(image, observation))
            row = dict(repeat=repeat, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       seconds=elapsed, stages=stages, lines=lines, state=state,
                       entry=entry_button(observation),
                       teleport=matching_line(observation, ("傳送", "传送", "teleport"), (.65,.65,1,1), exact=True))
            rows.append(row)
            print(json.dumps(dict(repeat=repeat, image=path.name, seconds=round(elapsed,4),
                                  state=state, stages=stages)), flush=True)
            (args.output/"results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    warm = [r["seconds"] for r in rows if r["repeat"] > 0]
    print("Warm median:", statistics.median(warm) if warm else "no warm round")


if __name__ == "__main__":
    main()
