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
        violations = set()
        for package in ("acquisition", "replay", "aria_trace"):
            for path in (ROOT / package).rglob("*.py"):
                for imported in imports_in(path):
                    if imported == "poc" or imported.startswith("poc."):
                        violations.add((path.relative_to(ROOT).as_posix(), imported))
        self.assertEqual(set(), violations)

    def test_poc_imports_are_compatibility_aliases_for_promoted_services(self):
        from aria_trace.services.tracking import PoseFusionGate as production_fusion
        from aria_trace.services.vision import KltAngularYawEstimator as production_yaw
        from poc.pose_fusion import PoseFusionGate as compatibility_fusion
        from poc.yaw_estimation import KltAngularYawEstimator as compatibility_yaw

        self.assertIs(production_fusion, compatibility_fusion)
        self.assertIs(production_yaw, compatibility_yaw)

    def test_workbench_server_legacy_exports_are_exact_compatibility_aliases(self):
        from acquisition import workbench as legacy
        from aria_trace.apps.workbench import server

        self.assertEqual(server.WORKBENCH_SERVICE, legacy.WORKBENCH_SERVICE)
        self.assertIs(server.WorkbenchHttpServer, legacy.WorkbenchHttpServer)
        self.assertIs(
            server.discover_workbench_instance,
            legacy.discover_workbench_instance,
        )
        self.assertIs(server.is_client_disconnect, legacy.is_client_disconnect)
        self.assertIs(server.occupied_port_message, legacy.occupied_port_message)


if __name__ == "__main__":
    unittest.main()
