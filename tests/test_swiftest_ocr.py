"""Backend failures must preserve recognition contracts and explicit selection."""
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from swiftest.ocr import LocalOCR
from swiftest.ocr_acceleration import accelerate, _Inference


class BackendTests(unittest.TestCase):
    def test_auto_missing_is_lazy_and_reports_cpu_fallback(self):
        cpu = Mock(return_value=([], None))
        with patch.object(LocalOCR, "_make_cpu", return_value=cpu) as factory, \
             patch.object(LocalOCR, "_has_openvino", return_value=False):
            ocr = LocalOCR()
            self.assertIsNone(ocr.info["backend"])
            factory.assert_not_called()
            self.assertEqual(ocr(object()), [])
            self.assertEqual(ocr.info["backend"], "cpu")
            self.assertIn("not installed", ocr.info["fallback"])

    def test_explicit_missing_never_silently_uses_cpu_on_retry(self):
        with patch.object(LocalOCR, "_make_cpu", return_value=Mock()), \
             patch.object(LocalOCR, "_has_openvino", return_value=False):
            ocr = LocalOCR(backend="openvino")
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "ocr-fast"):
                    ocr(object())
                self.assertIsNone(ocr.engine)
                self.assertIsNone(ocr.backend)

    def test_initialization_failure_respects_backend_selection(self):
        cpu = Mock(return_value=([], None))
        with patch.object(LocalOCR, "_make_cpu", return_value=cpu), \
             patch.object(LocalOCR, "_has_openvino", return_value=True), \
             patch("swiftest.ocr_acceleration.accelerate", side_effect=RuntimeError("compile")):
            auto = LocalOCR()
            self.assertEqual(auto(object()), [])
            self.assertIs(auto.engine, cpu)
            self.assertIn("compile", auto.fallback)
            strict = LocalOCR(backend="openvino")
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "compile"):
                    strict(object())

    def test_auto_retries_same_image_once_and_keeps_coordinates(self):
        image = object()
        cpu = Mock(return_value=([([[12, 20], [36, 20], [36, 30], [12, 30]], "Continue", .98)], None))
        accelerated = Mock(side_effect=RuntimeError("inference"))
        with patch.object(LocalOCR, "_make_cpu", return_value=cpu) as factory, \
             patch.object(LocalOCR, "_has_openvino", return_value=True), \
             patch("swiftest.ocr_acceleration.accelerate", return_value=accelerated):
            ocr = LocalOCR()
            words = ocr(image)
            self.assertEqual(words[0][0], "Continue")
            self.assertEqual(words[0][1].center, (24, 25))
            self.assertEqual(words[0][1].score, .98)
            accelerated.assert_called_once_with(image)
            cpu.assert_called_once_with(image)
            self.assertEqual(factory.call_count, 2)
            ocr(image)
            self.assertEqual(accelerated.call_count, 1)
            self.assertEqual(ocr.backend, "cpu")
            self.assertIn("inference", ocr.fallback)

    def test_explicit_and_cpu_inference_errors_propagate(self):
        for backend in ("cpu", "openvino"):
            engine = Mock(side_effect=RuntimeError("failed"))
            ocr = LocalOCR(backend=backend)
            ocr.engine, ocr.backend = engine, backend
            with self.assertRaisesRegex(RuntimeError, "failed"):
                ocr(object())
            self.assertEqual(engine.call_count, 1)


class AdapterTests(unittest.TestCase):
    def engine(self):
        return NS(text_det=NS(infer=object()), text_cls=NS(infer=object()), text_rec=NS(session=object()))

    def test_compilation_failure_leaves_all_original_sessions_intact(self):
        engine = self.engine()
        before = (engine.text_det.infer, engine.text_cls.infer, engine.text_rec.session)
        core = Mock()
        core.read_model.return_value = NS(inputs=[1], outputs=[1])
        core.compile_model.side_effect = [Mock(), RuntimeError("compile")]
        with patch("swiftest.ocr_acceleration.version", return_value="1.4.4"), \
             patch.dict(sys.modules, {"openvino": NS(Core=lambda: core),
                                      "rapidocr_onnxruntime": NS(__file__="package/__init__.py")}):
            with self.assertRaisesRegex(RuntimeError, "compile"):
                accelerate(engine, 4)
        self.assertEqual(before, (engine.text_det.infer, engine.text_cls.infer, engine.text_rec.session))

    def test_acceleration_is_per_instance_and_preserves_output_protocol(self):
        engine, other = self.engine(), self.engine()
        untouched = other.text_det.infer
        request = Mock()
        tensor, output = object(), object()
        request.get_output_tensor.return_value = NS(data=output)
        core = Mock()
        core.read_model.return_value = NS(inputs=[1], outputs=[1])
        core.compile_model.return_value.create_infer_request.return_value = request
        with patch("swiftest.ocr_acceleration.version", return_value="1.4.4"), \
             patch.dict(sys.modules, {"openvino": NS(Core=lambda: core),
                                      "rapidocr_onnxruntime": NS(__file__="package/__init__.py")}):
            self.assertIs(accelerate(engine, 4), engine)
        self.assertEqual(core.compile_model.call_count, 3)
        self.assertTrue(all(call.args[1] == "CPU" for call in core.compile_model.call_args_list))
        self.assertEqual(engine.text_rec.session(tensor), [output])
        request.infer.assert_called_once_with(inputs=[tensor])
        self.assertIs(other.text_det.infer, untouched)

    def test_unvalidated_library_version_is_rejected(self):
        with patch("swiftest.ocr_acceleration.version", return_value="2.0"):
            with self.assertRaisesRegex(RuntimeError, "1.4.4"):
                accelerate(self.engine(), 4)


if __name__ == "__main__":
    unittest.main()
