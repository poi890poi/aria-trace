"""Cooled same-task ablations and holdouts; each named case is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_genshin import ROOT, PACKAGE, command, run_trial, write_json

BASELINE = ROOT/"artifacts/android-speed/baseline/examples/genshin/profile.py"
CASES = {
    "baseline": ["--profile", str(BASELINE)],
    "ocr": ["--profile", str(BASELINE), "--ocr-max-side", "1280"],
    "entry": [],
    "poll": ["--profile", str(BASELINE), "--poll-cap", ".25"],
    "combined": ["--ocr-max-side", "1280", "--ocr-threads", "4", "--poll-cap", ".25"],
    "combined-cpu": ["--ocr-max-side", "1280", "--ocr-threads", "4", "--poll-cap", ".25", "--ocr-backend", "cpu"],
    "combined-openvino": ["--ocr-max-side", "1280", "--ocr-threads", "4", "--poll-cap", ".25", "--ocr-backend", "openvino"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--testkit-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--airtest-python", type=Path,
                        default=ROOT/".venv/airtest-benchmark/Scripts/python.exe")
    parser.add_argument("--assets", type=Path, default=ROOT/"artifacts/airtest-benchmark/assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--rest", type=float, default=60)
    parser.add_argument("--max-rest", type=float, default=300)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    order = args.order.split(",")
    if any(case not in (*CASES, "airtest") for case in order):
        raise ValueError("Unknown case")
    args.output.mkdir(parents=True, exist_ok=False)
    args.testkit_script = ROOT/"benchmarks/android/speed_trial.py"
    paths = list((ROOT/"swiftest").rglob("*.py"))
    paths += list((ROOT/"benchmarks/android").glob("*.py"))
    paths += [BASELINE, ROOT/"examples/genshin/profile.py"] + list(args.assets.glob("*"))
    metadata = dict(order=order, cases=CASES, installed=args.installed, hashes={str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
    write_json(args.output/"metadata.json", metadata)
    results = []
    adb = [args.adb, "-s", args.serial]
    try:
        for index, case in enumerate(order, 1):
            args.testkit_options = CASES.get(case, []) + (["--installed"] if args.installed else [])
            row = run_trial(args, index, "airtest" if case == "airtest" else "testkit")
            row["case"] = case
            results.append(row)
            write_json(args.output/"results.json", results)
            if not row["success"]:
                print("Stopped on failure; preserve and diagnose before a new cohort.", flush=True)
                break
    finally:
        command(adb+["shell", "am", "force-stop", PACKAGE])
        command(adb+["shell", "input", "keyevent", "223"])


if __name__ == "__main__":
    main()
