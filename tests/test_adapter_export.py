import importlib.util
import ast
import base64
import json
import tempfile
import unittest
from pathlib import Path

from aria_trace.adapters.filesystem.profile_registry import (
    AdapterRequest,
    ProfileContext,
    ProfileRegistry,
)
from aria_trace.workflows.adapter_export import _literal, export_resolved_adapter


class AdapterExportTests(unittest.TestCase):
    def test_binary_literal_chunks_large_payload_without_changing_bytes(self):
        payload = bytes(range(256)) * 4096
        encoded_literal = _literal(payload)
        encoded = ast.literal_eval(encoded_literal)
        self.assertEqual(payload, base64.b64decode(encoded))
        self.assertLessEqual(
            max(len(line.strip().strip("'")) for line in encoded_literal.splitlines()[1:-1]),
            96,
        )

    def test_export_embeds_runtime_data_and_never_reads_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "source" / "hik_camera_calibration.json"
            calibration.parent.mkdir()
            calibration.write_text(
                json.dumps(
                    {
                        "camera": {
                            "device_id": "CAM-1",
                            "full_sensor_mode": {
                                "width_px": 8, "height_px": 8, "fps": 30,
                            },
                            "hardware_roi_xywh": [0, 0, 8, 8],
                        },
                        "phone": {"serial": "PHONE-1"},
                        "imaging": {
                            "exposure_us": 1000,
                            "gain": 1,
                            "white_balance": {
                                "ratio_red": 1024,
                                "ratio_green": 1024,
                                "ratio_blue": 1024,
                            },
                        },
                        "normalization": {
                            "output_size_px": [8, 8],
                            "full_sensor_camera_to_output_3x3": [
                                [1, 0, 0], [0, 1, 0], [0, 0, 1]
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            context = ProfileContext(
                camera_id="CAM-1",
                phone_id="PHONE-1",
                panel_display={"natural_panel_px": [100, 200]},
            )
            registry = ProfileRegistry(root / "profiles")
            profile = registry.publish(
                "rig",
                context,
                {"profile_kind": "rig"},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted",
                activate=True,
            )
            output = root / "standalone_hikcam.py"
            result = export_resolved_adapter(
                output,
                registry=registry,
                context=context,
                request=AdapterRequest(mode="full", color_policy="rig_locked"),
            )
            source = output.read_text(encoding="utf-8")
            self.assertNotIn(str(calibration), source)
            self.assertIn("Registry reads: none", source)
            self.assertEqual(profile["revision_id"], result["profile_revisions"]["rig"])

            specification = importlib.util.spec_from_file_location(
                "generated_hikcam", str(output)
            )
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            paths = module._materialize()
            self.assertTrue(paths["hik_camera_calibration.json"].is_file())
            self.assertEqual(0, result["registry_reads_at_runtime"])


if __name__ == "__main__":
    unittest.main()
