"""Cold-start A/B experiment: only the screenshot transport differs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from compare_genshin import ROOT, PACKAGE, command, run_trial, write_json
from full_comparison import source_hashes


def worker():
    sys.argv.remove("--worker")
    variant = sys.argv[sys.argv.index("--variant")+1]
    index = sys.argv.index("--variant")
    del sys.argv[index:index+2]
    from binary_capture import BinaryCapture
    if variant == "binary":
        import swiftest.session as session
        session.Capture = BinaryCapture
    from full_comparison_trial import main as trial
    trial()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", default="ascii,binary,binary,ascii")
    parser.add_argument("--rest", type=float, default=60)
    parser.add_argument("--max-rest", type=float, default=300)
    args = parser.parse_args()
    if args.serial != "RFCR91GWXLX":
        raise ValueError("Experiment is scoped to the authorized phone")
    order = args.order.split(",")
    if any(v not in ("ascii","binary") for v in order):
        raise ValueError("Unknown transport")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.adb = "adb"
    args.assets = ROOT/"artifacts/airtest-benchmark/assets"
    args.testkit_python = ROOT/".venv/androidtestkit/Scripts/python.exe"
    args.testkit_script = Path(__file__).resolve()
    adb = [args.adb,"-s",args.serial]
    focus = command(adb+["shell","dumpsys","window"]).decode("utf-8","replace")
    prior = re.search(r"mCurrentFocus=.*?\s([\w.]+/[\w.$]+)",focus)
    prior = prior[1] if prior else None
    def hashes():
        return dict(source_hashes(args.assets), **{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(),Path(__file__).with_name("binary_capture.py").resolve())})
    baseline = hashes()
    write_json(args.output/"metadata.json",dict(order=order,hashes=baseline,prior=prior,
        timing="Fresh process and force-stopped app, through verified arrival, evidence and cleanup",
        capture="Identical production acquisition, JPEG80, exact-repeat policy; framing/transport only",
        setup="Both helper builds cached before cohort; helper launch/push/forward included"))
    results = []
    try:
        for index,variant in enumerate(order,1):
            args.testkit_options = ["--worker","--variant",variant,"--engine","testkit","--mode","cold"]
            row = run_trial(args,index,"testkit")
            row["variant"] = variant
            results.append(row)
            write_json(args.output/"results.json",results)
            assert hashes()==baseline, "Source changed during cohort"
            assert row["success"], "Failed trial retained; cohort stopped"
            assert row["task"]["metrics"]["capture_acquisition"]=="direct-bitmap"
            assert row["task"]["metrics"]["fallback"] is None
            assert row["rest_gate_met"], "Thermal gate missed; cohort stopped"
        write_json(args.output/"complete.json",dict(tasks=len(results),source_unchanged=True))
    finally:
        command(adb+["shell","am","force-stop",PACKAGE])
        if prior:
            command(adb+["shell","input","keyevent","224"])
            command(adb+["shell","am","start","-W","-n",prior])


if __name__=="__main__":
    worker() if "--worker" in sys.argv else main()
