"""Warm, paired command costs. No input is sent by this probe."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest.device import Android
from swiftest.shell import Shell


def summary(values):
    values = sorted(values)
    pos = (len(values) - 1) * .95
    lo = int(pos)
    return dict(n=len(values), mean_ms=statistics.mean(values),
                median_ms=statistics.median(values),
                p95_ms=values[lo] + (values[min(lo + 1, len(values)-1)]-values[lo]) * (pos-lo),
                max_ms=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    phone = Android(args.serial, persistent_shell=False)
    commands = {
        "noop": "true",
        "focus": "dumpsys window | grep mCurrentFocus",
        "focus_displays": "dumpsys window displays | grep mCurrentFocus",
        "display": 'dumpsys input | grep "Viewport INTERNAL" | head -1',
        "overlay": "dumpsys input | grep 'GameBooster Lock Screen' | head -1",
        "input_help": "input --help",
        "cmd_input_help": "cmd input help",
    }
    channel = Shell(phone.adb, phone.serial)
    rows = []
    try:
        for i in range(args.rounds + 1):
            for name, command in commands.items():
                if i == 0:
                    print("Warming " + name, flush=True)
                order = ("process", "persistent") if i % 2 == 0 else ("persistent", "process")
                for mode in order:
                    start = time.perf_counter()
                    data = phone.run("shell", command) if mode == "process" else channel.run(command)
                    ms = (time.perf_counter() - start) * 1000
                    if i:
                        rows.append(dict(round=i, command=name, mode=mode, ms=ms,
                                         output=data.decode("utf-8", "replace")))
        result = {f"{name}/{mode}": summary([r["ms"] for r in rows
                  if r["command"] == name and r["mode"] == mode])
                  for name in commands for mode in ("process", "persistent")}
        (args.output / "result.json").write_text(json.dumps(dict(
            serial=args.serial, python=sys.version, host=platform.platform(),
            commands=commands, rows=rows, summary=result,
            hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT/"swiftest/device.py", ROOT/"swiftest/shell.py"]}), indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        channel.close()
        phone.close()


if __name__ == "__main__":
    main()
