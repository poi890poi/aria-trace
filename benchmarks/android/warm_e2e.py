"""Warm Genshin task and capture-to-input timings; startup is recorded separately."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest import Android, LocalOCR, Session
from examples.genshin.profile import Phone, Runner, PACKAGE
from examples.genshin.profile import Phone as ProductionPhone, Runner as ProductionRunner
from steady_latency import summary
from e2e_candidates import GuardedPhone

@dataclass(frozen=True)
class Case:
    capture: str = "jpeg"
    guarded: bool = False
    poll: float = .25
    compact: bool = False
    narrow: bool = False
    async_evidence: bool = False
    production: bool = False
    compare_frames: bool = False


CASES = {"baseline": Case(), "direct": Case(capture="direct"),
    "guarded": Case(guarded=True), "combined": Case("direct",True),
    "compact": Case("direct",True,compact=True),
    "narrow": Case("direct",True,compact=True,narrow=True),
    "poll50": Case("direct",True,.05,True), "poll10": Case("direct",True,.01,True),
    "async": Case("direct",True,.05,True,async_evidence=True),
    "fast": Case("direct",True,.05,True,True,True),
    "bitmap": Case("bitmap",True,.05,True,True,True),
    "fast10": Case("direct",True,.01,True,True,True),
    "fast100": Case("direct",True,.1,True,True,True),
    "bitmap_full": Case("bitmap",True,.05,True,False,True),
    "bitmap100": Case("bitmap",True,.1,True,True,True),
    "bitmap10": Case("bitmap",True,.01,True,True,True),
    "bitmap_simple": Case("bitmap",True,.05,False,False,True),
    "production": Case("auto",poll=.1,production=True),
    "production50": Case("auto",poll=.05,production=True),
    "stream": Case("scrcpy",poll=.1,production=True),
    "stream_exact": Case("scrcpy",poll=.1,production=True,compare_frames=True)}

BASELINE = ROOT/"artifacts/android-e2e/baseline/genshin/profile.py"


def hashes():
    paths = list((ROOT/"swiftest").rglob("*.py")) + list((ROOT/"swiftest/assets").glob("*.java"))
    paths += [ROOT/"examples/genshin/profile.py", Path(__file__), Path(__file__).with_name("e2e_candidates.py"), BASELINE]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def arguments(output, poll=.25):
    return SimpleNamespace(intent=ROOT/"artifacts/airtest-benchmark/assets/teleport.json",
        output=output, timeout=240, arrival_timeout=60, inspect=False, poll=2, poll_cap=poll)


def measured(session, args, case):
    config = CASES[case]
    phone = ProductionPhone(session) if config.production else (
        GuardedPhone(session,compact=config.compact,narrow=config.narrow) if config.guarded else Phone(session))
    runner = (ProductionRunner if config.production else Runner)(phone, args, session)
    if config.compare_frames:
        import numpy as np
        backend = session._backend
        read = backend.read
        previous = session._image
        def exact_read():
            nonlocal previous
            current = read()
            backend.repeat = previous is not None and np.array_equal(previous,current)
            previous = current
            return current
        backend.read = exact_read
    writer = ThreadPoolExecutor(max_workers=1) if config.async_evidence else None
    writes = []
    if writer:
        def touch_target(point, element):
            import cv2
            name = {"minimap":"before", "unlocked_waypoint":"map", "teleport":"selected"}[element]
            writes.append(writer.submit(cv2.imwrite,str(runner.directory/(name+".png")),runner.frame))
            return runner.tap(point,element)
        runner.touch_target = touch_target
    # Launch/wake and initialization were completed before the timed task.
    runner.phone.launch = lambda: None
    observations, inputs, latest = [], [], {}
    frame, observe, tap = session.frame, runner.observe, runner.tap
    def capture():
        latest["capture_start"] = time.perf_counter()
        value = frame()
        latest["captured"] = time.perf_counter()
        return value
    def observation(waiting=False):
        before = runner.ocr_seconds
        value = observe(waiting)
        observations.append(dict(**latest, state=value[0],
            ocr_seconds=runner.ocr_seconds-before, ended=time.perf_counter(),
            repeated=session.capture_repeated))
        return value
    def dispatch(point, reason):
        begin = time.perf_counter()
        sent = tap(point, reason)
        end = time.perf_counter()
        inputs.append(dict(reason=reason, sent=sent, dispatch_ms=(end-begin)*1000,
            capture_to_input_ms=(end-latest["capture_start"])*1000))
        return sent
    session.frame, runner.observe, runner.tap = capture, observation, dispatch
    initial_inputs = session.stats["input_events"]
    start = time.perf_counter()
    result = dict(status="failed", case=case)
    try:
        result.update(runner.run())
        if writer:
            assert all(future.result() for future in writes), "Evidence image write failed"
        assert result["status"] == "teleport_completed"
        assert runner.actions == 3 and session.stats["input_events"] - initial_inputs == 3
        assert len(inputs) == 3 and all(i["sent"] for i in inputs), "Unexpected recovery or retry"
        assert sum(i["reason"] == "teleport" and i["sent"] for i in inputs) == 1
        assert any(o["state"] == "loading" for o in observations)
        assert [o["state"] for o in observations[-2:]] == ["world", "world"]
        return result
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if writer:
            writer.shutdown(wait=True)
        result.update(task_ms=(time.perf_counter()-start)*1000,
            capture_ms=runner.capture_seconds*1000, ocr_ms=runner.ocr_seconds*1000,
            observations=observations, inputs=inputs, metrics=session.report(),
            evidence_directory=str(runner.directory))
        (Path(args.output)/"result.json").write_text(json.dumps(result, indent=2))
        session.frame = frame


def main():
    global Phone, Runner
    # The starting profile is immutable local evidence, never the edited
    # production profile. This prevents improvements contaminating the control.
    spec = importlib.util.spec_from_file_location("e2e_frozen_profile", BASELINE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    Phone, Runner = original.Phone, original.Runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--order", default="baseline,baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--scrcpy-server", type=Path)
    args = parser.parse_args()
    order = args.order.split(",")
    if any(c not in CASES for c in order):
        raise ValueError("Unknown case")
    args.output.mkdir(parents=True, exist_ok=False)
    source = hashes()
    (args.output/"metadata.json").write_text(json.dumps(dict(hashes=source,order=order,
        cases={case:asdict(CASES[case]) for case in order},
        serial=args.serial, python=sys.version, intent_sha256=hashlib.sha256(
            (ROOT/"artifacts/airtest-benchmark/assets/teleport.json").read_bytes()).hexdigest()), indent=2))
    admin = Android(args.serial)
    prior = admin.foreground()
    ocr = LocalOCR(max_side=1280, threads=4, backend="openvino")
    results = []
    try:
        if not args.skip_preflight:
            with Session(device=Android(args.serial), ocr=ocr) as session:
                runner = Runner(Phone(session), arguments(args.output/"preflight"), session)
                start = time.perf_counter()
                value = runner.run()
                assert value["status"] == "teleport_completed"
                (args.output/"preflight.json").write_text(json.dumps(dict(**value,
                    excluded_seconds=time.perf_counter()-start)))
        for index, case in enumerate(order):
            directory = args.output/f"{index:02d}-{case}"
            directory.mkdir()
            with Session(device=Android(args.serial), ocr=ocr, capture=CASES[case].capture,
                         scrcpy_server=args.scrcpy_server) as session:
                session.open(PACKAGE)
                warm = Runner(Phone(session), arguments(directory/"warmup"), session)
                state, _, _ = warm.observe()
                ready_started = time.perf_counter()
                while state != "world" and time.perf_counter()-ready_started < 15:
                    time.sleep(.25)
                    state, _, _ = warm.observe()
                (directory/"readiness.json").write_text(json.dumps(dict(
                    state=state, seconds=time.perf_counter()-ready_started)))
                assert state == "world", "Warm task must start in verified gameplay"
                thermal_before = session.device.shell("dumpsys", "battery")
                if CASES[case].capture == "direct":
                    assert session.report()["capture_acquisition"] == "direct"
                if CASES[case].capture == "bitmap":
                    assert session.report()["capture_acquisition"] == "direct-bitmap"
                result = measured(session, arguments(directory, CASES[case].poll), case)
                result["battery_before"] = thermal_before
                result["battery_after"] = session.device.shell("dumpsys", "battery")
                (directory/"result.json").write_text(json.dumps(result,indent=2))
                (directory/"independent-arrival.png").write_bytes(session.device.run("exec-out","screencap","-p"))
                results.append(result)
                (args.output/"results.json").write_text(json.dumps(results,indent=2))
                print(json.dumps(dict(case=case,task_ms=result["task_ms"],
                    input_ms=[i["capture_to_input_ms"] for i in result["inputs"]])),flush=True)
        assert source == hashes(), "Source changed during cohort"
        totals = {case: dict(task=summary([r["task_ms"] for r in results if r["case"]==case]),
            input=summary([i["capture_to_input_ms"] for r in results if r["case"]==case for i in r["inputs"]]))
            for case in dict.fromkeys(order)}
        (args.output/"summary.json").write_text(json.dumps(totals,indent=2))
        print(json.dumps(totals,indent=2),flush=True)
    finally:
        if prior != PACKAGE:
            admin.shell("am", "force-stop", PACKAGE)
        if prior:
            admin.launch(prior)
        admin.close()


if __name__ == "__main__":
    main()
