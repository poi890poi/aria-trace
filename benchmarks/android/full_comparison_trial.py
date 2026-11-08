"""Instrument the real Airtest/TestKit workflows without changing their policies."""
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate(result, engine, mode, inputs, observations, events):
    expected = 5 if mode == "cold" else 3
    if result.get("status") != "teleport_completed":
        raise AssertionError("Workflow did not complete")
    if len(inputs) != expected or not all(i["sent"] for i in inputs):
        raise AssertionError("Unexpected recovery, retry or action count")
    if sum(i["reason"] == "teleport" for i in inputs) != 1:
        raise AssertionError("Expected one Teleport")
    if [i["reason"] for i in inputs[-3:]] != ["minimap", "waypoint", "teleport"]:
        raise AssertionError("Wrong task action order")
    if engine == "testkit":
        states = [o["state"] for o in observations]
        if "loading" not in states or states[-2:] != ["world", "world"]:
            raise AssertionError("Missing loading or consecutive arrival")
    else:
        names = [e["event"] for e in events]
        required = ["world_ready", "map_ready", "waypoint_selected", "loading", "arrival"]
        if [n for n in names if n in required] != required:
            raise AssertionError("Missing or unordered Airtest postconditions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("airtest", "testkit"), required=True)
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--assets", type=Path, default=ROOT/"artifacts/airtest-benchmark/assets")
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup_started = time.perf_counter()
    task_started = setup_started
    result = dict(status="failed", engine=args.engine, mode=args.mode)
    observations, inputs, frames, latest = [], [], [], {}
    runner = session = None
    loading_saved = False
    try:
        if args.engine == "testkit":
            import swiftest
            from swiftest import Session, LocalOCR
            from examples.genshin.profile import Phone, Runner
            session = Session(args.serial, timeout=args.timeout,
                              ocr=LocalOCR(max_side=1280, threads=4, backend="openvino"))
            session.__enter__()
            config = SimpleNamespace(intent=args.intent or args.assets/"teleport.json",
                output=args.output, timeout=args.timeout, arrival_timeout=120,
                inspect=False, poll=2)
            runner = Runner(Phone(session), config, session)
            if args.mode == "warm":
                session.device.package = "com.miHoYo.GenshinImpact"
                ready = time.perf_counter()+15
                while runner.observe()[0] != "world":
                    if time.perf_counter() >= ready:
                        raise AssertionError("Warm TestKit did not reach verified gameplay")
                    time.sleep(.25)
                runner.phone.launch = lambda: None
                runner.capture_seconds = runner.ocr_seconds = 0
                runner.ocr_calls = 0
            read, observe = session.frame, runner.observe

            def capture():
                latest["capture_start"] = time.perf_counter()
                image = read()
                latest["captured"] = time.perf_counter()
                frames.append(dict(**latest, repeated=session.capture_repeated))
                return image

            def observation(waiting=False):
                nonlocal loading_saved
                value = observe(waiting)
                observations.append(dict(**latest, state=value[0], ended=time.perf_counter()))
                if value[0] == "loading" and not loading_saved:
                    import cv2
                    if not cv2.imwrite(str(runner.directory/"loading.png"), value[2]):
                        raise OSError("Loading evidence failed")
                    loading_saved = True
                return value

            session.frame, runner.observe = capture, observation
            result.update(package_path=swiftest.__file__,
                ocr_config=dict(backend="openvino", threads=4, max_side=1280))
        else:
            from examples.genshin import airtest_teleport as air
            from airtest.core.settings import Settings
            logging.getLogger("airtest").setLevel(logging.WARNING)
            runner = air.Teleport(args.serial, args.assets, args.output, args.timeout)
            if args.mode == "warm":
                ready = time.perf_counter()+15
                while True:
                    runner.snapshot()
                    if runner.world():
                        break
                    if time.perf_counter() >= ready:
                        raise AssertionError("Warm Airtest did not reach verified gameplay")
                    time.sleep(.25)
                air.start_app = lambda *a, **kw: None
                runner.capture_s = runner.match_s = 0
                runner.captures = 0
                runner.events = []
            read = runner.snapshot

            def capture():
                latest["capture_start"] = time.perf_counter()
                image = read()
                latest["captured"] = time.perf_counter()
                frames.append(dict(**latest, repeated=None))
                return image

            runner.snapshot = capture
            result.update(package_version=importlib.metadata.version("airtest"),
                          operation_delay=Settings.OPDELAY)

        tap = runner.tap
        def dispatch(point, reason):
            started = time.perf_counter()
            value = tap(point, reason)
            ended = time.perf_counter()
            inputs.append(dict(reason="waypoint" if reason=="unlocked_waypoint" else reason,
                sent=value is not False, started=started, ended=ended,
                dispatch_seconds=ended-started,
                capture_to_input_seconds=ended-latest["capture_start"],
                ready_frame_to_input_seconds=ended-latest["captured"]))
            return value
        runner.tap = dispatch
        setup_seconds = time.perf_counter()-setup_started
        if args.mode == "warm":
            task_started = time.perf_counter()
        if args.engine == "airtest":
            runner.deadline = time.monotonic()+args.timeout
            runner.started = time.perf_counter()
        task_result = runner.run()
        ended = time.perf_counter()
        result.update(task_result)
        validate(result, args.engine, args.mode, inputs, observations, result.get("events", []))
        result.update(task_seconds=ended-task_started, setup_seconds=setup_seconds,
                      map_to_arrival_seconds=ended-inputs[-3]["started"])
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result.setdefault("task_seconds", time.perf_counter()-task_started)
        result.update(inputs=inputs, frames=frames, observations=observations)
        if runner is not None:
            if args.engine == "testkit":
                result.update(capture_seconds=runner.capture_seconds,
                    recognition_seconds=runner.ocr_seconds, ocr_calls=runner.ocr_calls,
                    metrics=session.report(), evidence_directory=str(runner.directory))
            else:
                result.update(capture_seconds=runner.capture_s,
                              recognition_seconds=runner.match_s, events=runner.events,
                              evidence_directory=str(args.output))
        try:
            if session is not None:
                session.close()
            elif runner is not None:
                runner.close()
        finally:
            (args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
