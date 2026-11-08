"""Run alternating cold-app-start trials and sample real Android temperatures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "com.miHoYo.GenshinImpact"


def command(argv, timeout=20):
    return subprocess.check_output([str(v) for v in argv], timeout=timeout,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def parse_temperatures(battery, thermal):
    result = {}
    for line in battery.splitlines():
        match = re.match(r"\s*(temperature|level|status|USB powered):\s*(.+)", line)
        if match:
            key, value = match.groups()
            if key == "temperature":
                result["battery_c"] = int(value) / 10
            else:
                result[key] = value
    # Prefer the fresh HAL section over cached listener temperatures.
    current = thermal.split("Current temperatures from HAL:")[-1]
    for value, kind, name, status in re.findall(
            r"Temperature\{mValue=([\d.-]+), mType=(\d+), mName=([^,]+), mStatus=(\d+)\}", current):
        if name in ("AP", "SKIN", "BAT"):
            result[name + "_c"] = float(value)
            result[name + "_status"] = int(status)
    match = re.search(r"Thermal Status:\s*(\d+)", thermal)
    if match:
        result["thermal_status"] = int(match[1])
    return result


def sample(adb):
    began = time.perf_counter()
    battery = command(adb + ["shell", "dumpsys", "battery"]).decode()
    thermal = command(adb + ["shell", "dumpsys", "thermalservice"]).decode()
    return dict(time=time.time(), read_seconds=time.perf_counter()-began,
                **parse_temperatures(battery, thermal))


def temperature_summary(samples):
    summary = {}
    for name in ("battery_c", "AP_c", "SKIN_c"):
        values = [s[name] for s in samples if name in s]
        if values:
            summary[name] = dict(start=values[0], end=values[-1], peak=max(values),
                                 delta=round(values[-1]-values[0], 3))
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def cool(adb, rest, maximum):
    command(adb + ["shell", "am", "force-stop", PACKAGE])
    command(adb + ["shell", "input", "keyevent", "223"])
    began = time.monotonic()
    samples = []
    while True:
        row = sample(adb)
        samples.append(row)
        elapsed = time.monotonic()-began
        ready = row.get("battery_c", 100) <= 34.5 and row.get("SKIN_c", 100) <= 37.0
        print(json.dumps({"phase": "rest", "seconds": round(elapsed),
                          "battery_c": row.get("battery_c"), "skin_c": row.get("SKIN_c")}), flush=True)
        if elapsed >= rest and (ready or elapsed >= maximum):
            break
        time.sleep(min(10, max(0.1, maximum-elapsed)))
    command(adb + ["shell", "input", "keyevent", "224"])
    command(adb + ["shell", "input", "keyevent", "82"])
    return dict(seconds=time.monotonic()-began, gate_met=ready, samples=samples)


def run_trial(args, index, implementation):
    directory = args.output / f"{index:02d}-{implementation}"
    directory.mkdir(parents=True, exist_ok=False)
    adb = [args.adb, "-s", args.serial]
    rest = (dict(seconds=0, gate_met=None, samples=[], reason="Warm continuation; no app stop")
            if getattr(args, "skip_cooldown", False) else cool(adb, args.rest, args.max_rest))
    write_json(directory / "rest.json", rest)
    if implementation == "airtest":
        argv = [args.airtest_python, getattr(args, "airtest_script", ROOT / "examples/genshin/airtest_teleport.py"),
                "--assets", args.assets]
        argv += getattr(args, "airtest_options", [])
    else:
        argv = [args.testkit_python, getattr(args, "testkit_script", ROOT / "benchmarks/android/testkit_teleport.py"),
                "--intent", args.assets / "teleport.json"]
        argv += getattr(args, "testkit_options", [])
    argv += ["--serial", args.serial, "--output", directory / "task", "--timeout", "240"]
    samples = [sample(adb)]
    stop = threading.Event()

    def monitor():
        while not stop.wait(2):
            try:
                samples.append(sample(adb))
            except Exception as error:
                samples.append(dict(time=time.time(), error=repr(error)))

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    began = time.perf_counter()
    print(json.dumps({"phase": "trial", "index": index, "tool": implementation}), flush=True)
    timeout = False
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(v) for v in argv], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while True:
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.perf_counter()-began
                    print(json.dumps({"phase": "running", "index": index,
                                      "seconds": round(elapsed), "temperature": samples[-1]}), flush=True)
                    if elapsed > 300:
                        timeout = True
                        process.kill()
                        code = process.wait(timeout=10)
                        break
        finally:
            elapsed = time.perf_counter()-began
            stop.set()
            worker.join(timeout=45)
    samples.append(sample(adb))
    write_json(directory / "temperatures.json", samples)
    task_file = directory / "task/result.json"
    task = json.loads(task_file.read_text()) if task_file.is_file() else {}
    result = dict(index=index, implementation=implementation, process_seconds=elapsed,
                  exit_code=code, timeout=timeout, rest_gate_met=rest["gate_met"],
                  success=code == 0 and task.get("status") == "teleport_completed",
                  temperature=temperature_summary(samples), task=task)
    write_json(directory / "trial.json", result)
    # Independent evidence is deliberately outside the timed task and thermal window.
    (directory / "independent-arrival.png").write_bytes(
        command(adb + ["exec-out", "screencap", "-p"]))
    print(json.dumps({"phase": "finished", "index": index, "tool": implementation,
                      "success": result["success"], "seconds": round(elapsed, 3),
                      "temperature": result["temperature"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--airtest-python", type=Path, required=True)
    parser.add_argument("--testkit-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/airtest-benchmark/assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rest", type=float, default=60)
    parser.add_argument("--max-rest", type=float, default=300)
    parser.add_argument("--order", default="airtest,testkit,testkit,airtest,testkit,airtest,airtest,testkit")
    args = parser.parse_args()
    for name in ("airtest_python", "testkit_python", "assets", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    adb = [args.adb, "-s", args.serial]
    metadata = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "serial": args.serial,
                "protocol": "README.md", "order": args.order.split(","), "hashes": {},
                "host": platform.platform(), "processor": platform.processor(),
                "python": sys.version}
    files = list((ROOT / "benchmarks/android").glob("*.py")) + list(args.assets.glob("*"))
    files += [ROOT / "examples/genshin/airtest_teleport.py", ROOT / "examples/genshin/profile.py"]
    files += list((ROOT / "swiftest").rglob("*.py"))
    for path in files:
        metadata["hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, argv in {
        "model": ["getprop", "ro.product.model"], "android": ["getprop", "ro.build.version.release"],
        "brightness": ["settings", "get", "system", "screen_brightness"],
        "brightness_mode": ["settings", "get", "system", "screen_brightness_mode"],
        "size": ["wm", "size"],
    }.items():
        metadata[name] = command(adb+["shell", *argv]).decode().strip()
    for name, python in (("airtest", args.airtest_python), ("testkit", args.testkit_python)):
        metadata[name + "_packages"] = command([python, "-m", "pip", "freeze"]).decode().splitlines()
    write_json(args.output / "metadata.json", metadata)
    results = []
    try:
        for index, implementation in enumerate(metadata["order"], 1):
            if implementation not in ("airtest", "testkit"):
                raise ValueError("Unknown implementation")
            results.append(run_trial(args, index, implementation))
            write_json(args.output / "results.json", results)
            if not results[-1]["success"]:
                print("Stopping on failed holdout: preserve failure and diagnose before restarting.", flush=True)
                break
    finally:
        command(adb + ["shell", "am", "force-stop", PACKAGE])
        command(adb + ["shell", "input", "keyevent", "223"])
    print(json.dumps({"phase": "complete", "trials": len(results)}), flush=True)


if __name__ == "__main__":
    main()
