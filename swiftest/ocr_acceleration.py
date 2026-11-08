"""Per-instance OpenVINO execution for the validated PP-OCRv4 model interfaces."""
from importlib.metadata import version
from pathlib import Path


class _Inference:
    def __init__(self, request):
        self.request = request

    def __call__(self, tensor):
        self.request.infer(inputs=[tensor])
        # The legacy detector/classifier/recognizer expect a one-output list.
        return [self.request.get_output_tensor().data]


def accelerate(engine, threads):
    # These integration points are pinned and checked, not patched globally.
    if version("rapidocr-onnxruntime") != "1.4.4":
        raise RuntimeError("OpenVINO integration requires rapidocr-onnxruntime 1.4.4.")
    import openvino
    import rapidocr_onnxruntime

    root = Path(rapidocr_onnxruntime.__file__).parent / "models"
    core = openvino.Core()
    replacements = []
    for component, attribute, filename in (
        (engine.text_det, "infer", "ch_PP-OCRv4_det_infer.onnx"),
        (engine.text_cls, "infer", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
        (engine.text_rec, "session", "ch_PP-OCRv4_rec_infer.onnx"),
    ):
        model = core.read_model(root / filename)
        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise RuntimeError("Unexpected OCR model interface; acceleration was not applied.")
        compiled = core.compile_model(model, "CPU", {
            "INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
        })
        replacements.append((component, attribute, _Inference(compiled.create_infer_request())))
    # Keep the original engine intact if any model fails to compile.
    for component, attribute, inference in replacements:
        setattr(component, attribute, inference)
    return engine
