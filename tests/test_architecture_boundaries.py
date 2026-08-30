import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imports_in(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_new_contract_package_has_no_legacy_or_platform_dependencies(self):
        forbidden = ("acquisition", "replay", "poc", "cv2", "numpy", "PySide6")
        violations = []
        for layer in ("domain", "ports"):
            for path in (ROOT / "aria_trace" / layer).rglob("*.py"):
                for imported in imports_in(path):
                    if imported.startswith(forbidden):
                        violations.append((str(path.relative_to(ROOT)), imported))
        self.assertEqual([], violations)

    def test_no_new_production_imports_from_poc_are_added(self):
        allowed_debt = {
            ("acquisition/live_tracker.py", "poc.pose_fusion"),
            ("acquisition/live_tracker.py", "poc.yaw_estimation"),
            ("acquisition/scene_yaw_calibration.py", "poc.yaw_estimation"),
        }
        violations = set()
        for package in ("acquisition", "replay", "aria_trace"):
            for path in (ROOT / package).rglob("*.py"):
                for imported in imports_in(path):
                    if imported == "poc" or imported.startswith("poc."):
                        violations.add((path.relative_to(ROOT).as_posix(), imported))
        self.assertEqual(set(), violations - allowed_debt)


if __name__ == "__main__":
    unittest.main()
