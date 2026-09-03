import unittest
from unittest import mock

import iris_tools


class IrisToolsTests(unittest.TestCase):
    def test_public_helper_forwards_arguments_to_owned_module(self):
        module = mock.Mock()
        module.main.return_value = 7
        with mock.patch.object(
            iris_tools.importlib, "import_module", return_value=module
        ) as importer:
            result = iris_tools.zigzag_acquisition(["--moves", "4"])
        importer.assert_called_once_with(
            "rig_runtime.workflows.minimap_capture"
        )
        module.main.assert_called_once_with(["--moves", "4"])
        self.assertEqual(7, result)

    def test_dispatcher_uses_user_python_helper_command(self):
        helper = mock.Mock(return_value=3)
        with mock.patch.dict(
            iris_tools.COMMANDS,
            {"zigzag-acquisition": helper},
            clear=True,
        ):
            result = iris_tools.main(
                ["zigzag-acquisition", "--android-package", "org.example.game"]
            )
        helper.assert_called_once_with(
            ["--android-package", "org.example.game"]
        )
        self.assertEqual(3, result)

    def test_dispatcher_forwards_selected_tool_help(self):
        helper = mock.Mock(return_value=0)
        with mock.patch.dict(
            iris_tools.COMMANDS,
            {"rig-calibration": helper},
            clear=True,
        ):
            result = iris_tools.main(["rig-calibration", "--help"])
        helper.assert_called_once_with(["--help"])
        self.assertEqual(0, result)


if __name__ == "__main__":
    unittest.main()
