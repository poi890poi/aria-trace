# SwifTest starter project

Copy this directory to start tests for your own Android app. This is a template,
not an installed sample application. Replace `com.example.app` in both test files
and the resource IDs in `ui.json` with values from your app's native hierarchy.

The sample opens a details screen, verifies its title, returns and verifies the
original control. Python and JSON are equivalent alternatives sharing one catalog.
Keep either test file for a single run; running the directory with both files
executes the flow twice.

```text
python -m swiftest hierarchy artifacts/inspection/ui.xml --serial YOUR_DEVICE_SERIAL
python -m swiftest run my-tests/test_navigation.py --serial YOUR_DEVICE_SERIAL --output artifacts/my-app-tests --failure-screenshot
python -m swiftest run my-tests/test_navigation.json --serial YOUR_DEVICE_SERIAL --output artifacts/my-app-tests --failure-screenshot
```

Use the interpreter where the framework is installed. The CLI loads `ui.json`
beside each test and supplies its session. Reports are written under
`artifacts/my-app-tests/<run-id>/report.json`. These native-only examples need no
OCR, template assets, app profile or game benchmark configuration.

See [the framework guide](../../docs/android-usage.md) for installation, selectors,
pop-ups, embedding and reports, and [requirements](../../docs/android-requirements.md)
for framework contracts. Live use requires adapting the placeholders first; the
examples' Python/JSON execution and catalog were checked against an offline device
fixture, not against an arbitrary installed application.
