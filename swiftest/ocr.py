"""Lazy, local CPU OCR. Used only when a script requests rendered text."""

from .targets import Box


class LocalOCR:
    def __init__(self, *, threads=2, max_side=None, backend="auto"):
        if type(threads) is not int or threads < 1:
            raise ValueError("OCR threads must be a positive integer.")
        if max_side is not None and (type(max_side) is not int or max_side < 32):
            raise ValueError("OCR max_side must be an integer of at least 32 pixels.")
        if backend not in ("auto", "cpu", "openvino"):
            raise ValueError("OCR backend must be auto, cpu or openvino.")
        self.threads, self.max_side = threads, max_side
        self.requested_backend = backend
        self.backend = None
        self.fallback = None
        self.engine = None

    @property
    def info(self):
        return dict(requested_backend=self.requested_backend, backend=self.backend,
                    model="PP-OCRv4", threads=self.threads, max_side=self.max_side,
                    fallback=self.fallback)

    @staticmethod
    def _has_openvino():
        from importlib.util import find_spec
        return find_spec("openvino") is not None

    def _make_cpu(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "Install swiftest[ocr] to read rendered text."
            ) from exc
        options = {} if self.max_side is None else dict(
            max_side_len=self.max_side, det_limit_type="max")
        return RapidOCR(intra_op_num_threads=self.threads,
                        inter_op_num_threads=1, **options)

    def _initialize(self):
        self.engine = self._make_cpu()
        self.backend = "cpu"
        if self.requested_backend == "cpu":
            return
        if not self._has_openvino():
            if self.requested_backend == "openvino":
                self.engine, self.backend = None, None
                raise RuntimeError("Install swiftest[ocr-fast] to use OpenVINO.")
            self.fallback = "Optional OpenVINO package is not installed."
            return
        try:
            from .ocr_acceleration import accelerate
            self.engine = accelerate(self.engine, self.threads)
            self.backend = "openvino"
        except (ImportError, RuntimeError, OSError) as error:
            if self.requested_backend != "auto":
                self.engine, self.backend = None, None
                raise
            self.fallback = "OpenVINO initialization failed: " + str(error)

    def __call__(self, image):
        if self.engine is None:
            self._initialize()
        try:
            lines, _ = self.engine(image)
        except (ImportError, RuntimeError, OSError) as error:
            if self.requested_backend != "auto" or self.backend != "openvino":
                raise
            self.fallback = "OpenVINO inference failed: " + str(error)
            self.engine, self.backend = self._make_cpu(), "cpu"
            lines, _ = self.engine(image)
        result = []
        for corners, text, confidence in lines or ():
            xs, ys = zip(*corners)
            l, t, r, b = round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))
            result.append((text, Box(l, t, r - l, b - t, float(confidence))))
        return result
