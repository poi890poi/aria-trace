"""Read-only Android 13 screenshot experiment, retaining equal JPEG/repeat policy."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest.device import Android
from steady_latency import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--force-encode", action="store_true")
    parser.add_argument("--compare-bitmap", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sdk = Path(os.environ["LOCALAPPDATA"]) / "Android/Sdk"
    android = sorted(sdk.glob("platforms/*/android.jar"))[-1]
    d8 = sorted(sdk.glob("build-tools/*/lib/d8.jar"))[-1]
    source = Path(__file__).with_name("IntentScreenshot.java")
    for command in (
        ["javac", "-source", "8", "-target", "8", "-classpath", str(android), "-d", str(args.output), str(source)],
        ["java", "-cp", str(d8), "com.android.tools.r8.D8", "--min-api", "26", "--lib", str(android),
         "--output", str(args.output/"probe.jar"), str(args.output/"IntentScreenshot.class")],
    ):
        subprocess.run(command, check=True, capture_output=True, timeout=90)
    phone = Android(args.serial)
    remote = "/data/local/tmp/testkit-capture-probe-" + uuid.uuid4().hex + ".jar"
    process = None
    rows = []
    modes = ("direct", "bitmap") if args.compare_bitmap else ("screencap", "direct")
    try:
        phone.push(args.output/"probe.jar", remote)
        with (args.output/"device-timings.csv").open("wb") as log:
            process = subprocess.Popen([phone.adb, "-s", phone.serial, "shell", "-T", "env",
                "CLASSPATH=" + remote, "app_process", "/system/bin", "IntentScreenshot"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            responses = queue.Queue()

            def receive():
                while True:
                    line = process.stdout.readline(32*1024*1024)
                    responses.put(line)
                    if not line:
                        break

            threading.Thread(target=receive, daemon=True).start()
            previous = None
            for i in range(args.rounds + 1):
                for mode in (modes if i % 2 == 0 else modes[::-1]):
                    start = time.perf_counter()
                    request = {"screencap":b"C", "direct":b"D", "bitmap":b"R"}[mode]
                    process.stdin.write(request.lower() if args.force_encode else request)
                    process.stdin.flush()
                    wire = responses.get(timeout=15)
                    if not wire:
                        raise RuntimeError("Capture probe stopped; inspect device-timings.csv")
                    data = base64.b64decode(wire.rstrip(), validate=True)
                    repeated = not data
                    if repeated:
                        assert previous is not None
                        data = previous
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    assert image is not None
                    elapsed = (time.perf_counter()-start)*1000
                    previous = data
                    if i:
                        rows.append(dict(round=i, mode=mode, ms=elapsed, repeated=repeated,
                                         shape=list(image.shape), wire_bytes=len(wire)))
                    if i == 1:
                        cv2.imwrite(str(args.output/(mode + ".png")), image)
            result = {mode: summary([r["ms"] for r in rows if r["mode"] == mode])
                      for mode in modes}
            (args.output/"result.json").write_text(json.dumps(dict(rows=rows, summary=result,
                serial=args.serial, force_encode=args.force_encode,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), indent=2))
            print(json.dumps(result, indent=2), flush=True)
    finally:
        if process:
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
        phone.shell("rm", "-f", remote)
        phone.close()


if __name__ == "__main__":
    main()
