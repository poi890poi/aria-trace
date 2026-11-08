"""Isolate OCR and scheduling changes, retaining the frozen baseline profile."""
import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if "--installed" in sys.argv:
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
else:
    sys.path.insert(0, str(ROOT))
import swiftest
from swiftest import LocalOCR, Session
from swiftest.workflow import interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--profile", type=Path, default=ROOT/"examples/genshin/profile.py")
    parser.add_argument("--ocr-max-side", type=int)
    parser.add_argument("--ocr-threads", type=int, default=2)
    parser.add_argument("--ocr-backend", choices=["auto", "cpu", "openvino"], default="auto")
    parser.add_argument("--poll-cap", type=float)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("measured_profile", args.profile)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    # One configurable scheduler for the frozen and current profiles.
    profile.interval = lambda *values, **ignored: interval(*values, max_interactive=args.poll_cap)
    started = time.perf_counter()
    observations, inputs = [], []
    latest = {}
    result = {"status": "failed"}
    with Session(args.serial, timeout=args.timeout,
                 ocr=LocalOCR(max_side=args.ocr_max_side, threads=args.ocr_threads,
                              backend=args.ocr_backend)) as session:
        runner = profile.Runner(profile.Phone(session), SimpleNamespace(
            intent=args.intent, timeout=args.timeout, arrival_timeout=120,
            inspect=False, poll=2, poll_cap=args.poll_cap, output=args.output), session)
        frame, observe, tap = session.frame, runner.observe, runner.tap

        def capture():
            latest["capture_start"] = time.perf_counter()
            image = frame()
            latest["captured"] = time.perf_counter()
            return image

        def observation(waiting_for_loading=False):
            ocr_before = runner.ocr_seconds
            value = observe(waiting_for_loading)
            latest["recognized"] = time.perf_counter()
            observations.append(dict(**latest, state=value[0],
                                     ocr_seconds=runner.ocr_seconds-ocr_before))
            if value[0] == "loading":
                import cv2
                cv2.imwrite(str(runner.directory/"loading.png"), value[2])
            return value

        def dispatch(point, reason):
            began = time.perf_counter()
            sent = tap(point, reason)
            ended = time.perf_counter()
            inputs.append(dict(reason=reason, sent=sent, seconds=ended-began,
                               ready_frame_to_input_seconds=ended-latest["captured"],
                               capture_start_to_input_seconds=ended-latest["capture_start"]))
            return sent

        session.frame, runner.observe, runner.tap = capture, observation, dispatch
        try:
            result = runner.run()
            if result["status"] == "teleport_completed":
                if session.report()["input_events"] != 5 or runner.actions != 5:
                    raise AssertionError("Expected exactly two entry taps and three task taps.")
                if sum(i["reason"] == "teleport" and i["sent"] for i in inputs) != 1:
                    raise AssertionError("Expected one Teleport command.")
        except Exception as error:
            result["error"] = repr(error)
            raise
        finally:
            result.update(task_seconds=time.perf_counter()-started, metrics=session.report(),
                          capture_seconds=runner.capture_seconds, ocr_seconds=runner.ocr_seconds,
                          observations=observations, inputs=inputs,
                          package_path=swiftest.__file__,
                          package_version=importlib.metadata.version("swiftest"),
                          evidence_directory=str(runner.directory))
            (args.output/"result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
