"""Diagnostic frame/stage trace; instrumentation makes these non-benchmark runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker():
    sys.argv.remove("--worker")
    import cv2
    from swiftest.ocr import LocalOCR
    from examples.genshin.profile import Runner, Phone
    from full_comparison_trial import main as trial
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    warm = sys.argv[sys.argv.index("--mode") + 1] == "warm"
    rows, writes, stages = [], [], []
    initialize = LocalOCR._initialize

    class Timed:
        def __init__(self, inner, name):
            self.inner, self.name = inner, name

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __call__(self, *args, **kwargs):
            began = time.perf_counter()
            result = self.inner(*args, **kwargs)
            stages.append(dict(stage=self.name, seconds=time.perf_counter()-began,
                               count=len(result[0]) if result[0] is not None else 0))
            return result

    def initialized(self):
        initialize(self)
        for name in ("text_det", "text_cls", "text_rec"):
            setattr(self.engine, name, Timed(getattr(self.engine, name), name))
    LocalOCR._initialize = initialized
    phone_init = Phone.__init__

    def phone_ready(self, session):
        phone_init(self, session)
        if warm:
            assert session.device.foreground() == "com.miHoYo.GenshinImpact"
    Phone.__init__ = phone_ready
    original_tap, original_observe = Runner.tap, Runner.observe

    def tap(self, point, reason):
        value = original_tap(self, point, reason)
        if reason == "teleport" and value:
            self._diagnostic_teleport = time.perf_counter()
        return value
    Runner.tap = tap
    with ThreadPoolExecutor(max_workers=1) as writer:
        def observe(self, waiting_for_loading=False):
            stages.clear()
            before_bytes = self.session.report()["capture_bytes"]
            before_ocr = self.ocr_calls
            began = time.perf_counter()
            value = original_observe(self, waiting_for_loading)
            ended = time.perf_counter()
            if warm or getattr(self, "_diagnostic_teleport", None):
                path = self.directory / ("diagnostic-%04d.png" % self.index)
                writes.append(writer.submit(cv2.imwrite, str(path), value[2],
                                             [cv2.IMWRITE_PNG_COMPRESSION, 1]))
                rows.append(dict(path=str(path), started=began, ended=ended,
                    state=value[0], observation=value[1], waiting=waiting_for_loading,
                    repeated=self.session.capture_repeated,
                    ocr_calls=self.ocr_calls-before_ocr, stages=list(stages),
                    bytes=self.session.report()["capture_bytes"]-before_bytes,
                    teleport_ended=getattr(self, "_diagnostic_teleport", None)))
            return value
        Runner.observe = observe
        try:
            trial()
        finally:
            for future in writes:
                if not future.result():
                    raise OSError("Diagnostic image write failed")
            output.mkdir(parents=True, exist_ok=True)
            (output / "diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    from compare_genshin import command, PACKAGE, sample, write_json
    from full_comparison import source_hashes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-runs", type=int, default=3)
    args = parser.parse_args()
    if args.serial != "RFCR91GWXLX":
        raise ValueError("This diagnostic is scoped to the authorized phone")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    adb = ["adb", "-s", args.serial]
    focus = command(adb + ["shell", "dumpsys", "window"]).decode("utf-8", "replace")
    prior = re.search(r"mCurrentFocus=.*?\s([\w.]+/[\w.$]+)", focus)
    prior = prior[1] if prior else None
    assets = ROOT / "artifacts/airtest-benchmark/assets"
    hashes = source_hashes(assets)
    write_json(args.output / "metadata.json", dict(hashes=hashes, prior=prior,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat="Diagnostic only: asynchronous lossless frame writes and stage instrumentation",
        temperature_before=sample(adb)))
    results = []
    try:
        command(adb + ["shell", "input", "keyevent", "224"])
        command(adb + ["shell", "am", "force-stop", PACKAGE])
        for index, mode in enumerate(["cold"] + ["warm"] * args.warm_runs):
            directory = args.output / ("%02d-%s" % (index, mode))
            with (args.output / ("%02d-console.txt" % index)).open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(Path(__file__)), "--worker",
                    "--engine", "testkit", "--mode", mode, "--serial", args.serial,
                    "--assets", str(assets), "--output", str(directory)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            result = json.loads((directory / "result.json").read_text())
            results.append(dict(mode=mode, seconds=result["task_seconds"],
                                observations=len(result["observations"])))
            print(json.dumps(results[-1]), flush=True)
            assert hashes == source_hashes(assets), "Runtime changed during probe"
            write_json(args.output / "results.json", results)
        write_json(args.output / "complete.json", dict(source_unchanged=True,
            temperature_after=sample(adb), tasks=len(results)))
    finally:
        command(adb + ["shell", "am", "force-stop", PACKAGE])
        if prior:
            command(adb + ["shell", "am", "start", "-W", "-n", prior])


if __name__ == "__main__":
    worker() if "--worker" in sys.argv else main()
