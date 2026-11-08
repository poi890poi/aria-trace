"""Instrument the existing TestKit Genshin profile without changing its policy."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swiftest import Session
from examples.genshin.profile import Phone, Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    result = {"status": "failed"}
    with Session(args.serial, timeout=args.timeout) as session:
        runner = Runner(Phone(session), SimpleNamespace(
            intent=args.intent, timeout=args.timeout, arrival_timeout=120,
            inspect=False, poll=2, output=args.output), session)
        original = runner.observe

        def observe(waiting_for_loading=False):
            value = original(waiting_for_loading)
            if value[0] == "loading":
                import cv2
                cv2.imwrite(str(runner.directory / "loading.png"), value[2])
            return value

        runner.observe = observe
        try:
            result = runner.run()
        except Exception as error:
            result["error"] = repr(error)
            raise
        finally:
            result.update(task_seconds=time.perf_counter() - began,
                          metrics=session.report(), capture_seconds=runner.capture_seconds,
                          ocr_seconds=runner.ocr_seconds, evidence_directory=str(runner.directory))
            (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
