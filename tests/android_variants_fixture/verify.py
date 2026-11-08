"""Exercise the INSTALLED wheel on a real device, with three independent UI variants."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import swiftest
from swiftest import Session
from swiftest.intent import execute, load_catalog

PACKAGE = "org.swiftest.variantsfixture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/androidtestkit/variants-live"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    rows = []
    print("Package location:", swiftest.__file__, flush=True)
    plan = json.loads((root/"examples/variants/test_continue.json").read_text())
    for variant in range(3):
        with Session(args.serial, timeout=60) as session:
            session.device.shell("am", "force-stop", PACKAGE)
            session.device.shell("input", "keyevent", "224")
            session.device.shell("input", "keyevent", "82")
            session.device.shell("am", "start", "-W", "-n", PACKAGE+"/.MainActivity",
                                 "--ei", "variant", str(variant))
            load_catalog(session, root/"examples/variants/ui.json")
            try:
                execute(session, plan)
            except Exception as error:
                (args.output/f"variant-{variant}-failure.json").write_text(json.dumps(
                    dict(error=repr(error), metrics=session.report()), indent=2))
                raise
            report = session.report()
            assert report["input_events"] == 2, report
            assert report["captures"] == report["ocr_calls"] == 0, report
            assert any(e.get("status") == "dismissed" for e in report["actions"]), report
            rows.append(dict(variant=variant, status="passed", metrics=report,
                             version=importlib.metadata.version("swiftest")))
            (args.output/"results.json").write_text(json.dumps(rows, indent=2))
            # Independent screenshot after assertions; outside the measured session counters.
            (args.output/f"variant-{variant}-arrival.png").write_bytes(
                session.device.run("exec-out", "screencap", "-p"))
            print(json.dumps(rows[-1]), flush=True)
    print("Three native variants passed.")


if __name__ == "__main__":
    main()
