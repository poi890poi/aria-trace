"""Run Python tests, declarative tests, or convert a recorded Airtest project."""

import argparse
import ast
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
import tokenize
import traceback
from functools import wraps

from .convert import convert
from .device import devices
from .intent import execute, load_catalog
from .session import Session
from .ocr import LocalOCR
from .outcome import json_value


def suite(path):
    if path.is_file():
        return [path]
    if path.suffix == ".air":
        entry = path / (path.stem + ".py")
        return [entry] if entry.is_file() else []
    return sorted(set(path.rglob("test_*.py")) | set(path.rglob("test_*.json")))


def _report_script_errors(function):
    @wraps(function)
    def run(path, session, *args, **kwargs):
        try:
            return function(path, session, *args, **kwargs)
        except (Exception, SystemExit) as exc:
            if session._failure_exception is not exc:
                session.record_failure(path.name, exc, phase="script")
            raise
    return run


@_report_script_errors
def run_script(path, session, failure_dir=None):
    session.reset_outcome()
    if path.suffix == ".json":
        execute(session, json.loads(path.read_text(encoding="utf-8-sig")))
        return [{"name": path.stem, "status": "passed", "outcome": session.complete_outcome()}]
    if path.suffix != ".py":
        raise ValueError("A test must be Python or JSON.")
    with tokenize.open(path) as stream:
        syntax = ast.parse(stream.read(), filename=str(path))
    for node in ast.walk(syntax):
        imported = (
            [item.name for item in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if any(name.split(".")[0] == "airtest" for name in imported):
            raise ValueError(
                "Convert the Airtest script first with swiftest convert."
            )
    old_path = sys.path.copy()
    old_argv = sys.argv
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path)]
    try:
        try:
            ns = runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            if exc.code in (0, None):
                return [{"name": path.stem, "status": "passed", "outcome": session.complete_outcome()}]
            raise AssertionError(f"Script exited with {exc.code}") from exc
        tests = [
            (name, func)
            for name, func in ns.items()
            if name.startswith("test_")
            and inspect.isfunction(func)
            and func.__globals__.get("__file__") == str(path)
        ]
        cases = []
        for name, func in tests:
            session.reset_outcome()
            case = {"name": name, "status": "passed"}
            started = time.monotonic()
            try:
                args = list(inspect.signature(func).parameters)
                if args == ["session"]:
                    value = func(session)
                elif not args:
                    value = func()
                else:
                    raise ValueError(f"{name} must accept zero arguments or session.")
                session.complete_outcome(value, name=name + " return")
            except (Exception, SystemExit) as exc:
                if isinstance(exc, SystemExit) and exc.code in (0, None):
                    session.complete_outcome()
                else:
                    if session._failure_exception is not exc:
                        session.record_failure(name, exc, phase="python")
                    case.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                    if failure_dir:
                        try:
                            session.snapshot(failure_dir / (name + ".png"))
                        except Exception as capture_error:
                            case["capture_error"] = str(capture_error)
            case["outcome"] = session.outcome()
            case["elapsed_s"] = round(time.monotonic() - started, 3)
            cases.append(case)
        return cases or [{"name": path.stem, "status": "passed", "outcome": session.complete_outcome()}]
    finally:
        sys.path[:] = old_path
        sys.argv = old_argv


def run_suite(args):
    files = suite(args.path.resolve())
    if not files:
        raise ValueError("No test files found.")
    out = args.output.resolve() / str(time.time_ns())
    out.mkdir(parents=True)
    results = []
    for index, path in enumerate(files):
        start = time.monotonic()
        result = {"file": str(path), "status": "passed"}
        session = None
        try:
            with Session(
                args.serial,
                adb=args.adb,
                base=path.parent,
                capture=args.capture,
                scrcpy_server=args.scrcpy_server,
                timeout=args.timeout,
                poll_interval=getattr(args, "poll_interval", 0.1),
                ocr=LocalOCR(threads=args.ocr_threads, max_side=args.ocr_max_side,
                             backend=args.ocr_backend),
            ) as session:
                session.output = out
                try:
                    catalog = args.catalog or path.parent / "ui.json"
                    if args.catalog or catalog.is_file():
                        load_catalog(session, catalog)
                    if args.profile:
                        module = runpy.run_path(str(args.profile.resolve()))
                        with session.intent("profile", file=str(path)):
                            value = module["run_intent"](session, path)
                            session.checkpoint(value)
                        result["profile"] = json_value(value)
                        result["outcome"] = session.complete_outcome()
                        result["tests"] = [path.name]
                    else:
                        cases = run_script(
                            path,
                            session,
                            out / f"case-{index}" if args.failure_screenshot else None,
                        )
                        result["tests"] = [case["name"] for case in cases]
                        result["cases"] = cases
                        failures = [
                            case for case in cases if case["status"] == "failed"
                        ]
                        if failures:
                            result["outcome"] = failures[0]["outcome"]
                            result.update(
                                status="failed",
                                error="; ".join(
                                    f"{c['name']}: {c['error']}" for c in failures
                                ),
                            )
                        else:
                            result["outcome"] = cases[-1]["outcome"]
                except Exception:
                    if args.failure_screenshot:
                        try:
                            session.snapshot(out / f"failure-{index}.png")
                        except Exception:
                            pass
                    raise
        except (Exception, SystemExit) as exc:
            if session:
                if session._failure_exception is not exc:
                    session.record_failure(path.name, exc, phase="script or teardown")
                result["outcome"] = session.outcome()
            else:
                result["outcome"] = {"status": "failed", "last_valid_result": None,
                                     "failed_intent": {"intent": "create session", "error": str(exc)}}
            result.update(
                status="failed", error=str(exc), traceback=traceback.format_exc()
            )
        if session:
            result["metrics"] = session.report()
            # File/case outcomes above are authoritative; the session can end on
            # a passing case even when an earlier case failed.
            result["metrics"].pop("outcome", None)
        result["elapsed_s"] = round(time.monotonic() - start, 3)
        results.append(result)
        print(result["status"], path.name, result.get("error", ""))
        if result["status"] == "failed":
            outcome = result["outcome"]
            print("  failed intent:", (outcome.get("failed_intent") or {}).get("intent"))
            print("  last valid result:", json.dumps(outcome.get("last_valid_result"), ensure_ascii=False))
    report = out / "report.json"
    report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(report)
    return int(any(r["status"] == "failed" for r in results))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="swiftest")
    commands = parser.add_subparsers(dest="command", required=True)
    devices_cmd = commands.add_parser("devices")
    devices_cmd.add_argument("--adb", default="adb")
    run = commands.add_parser(
        "run", help="Run a Python file, JSON intent or directory of test_* scripts."
    )
    run.add_argument("path", type=Path)
    run.add_argument("--output", type=Path, default=Path("swiftest-results"))
    run.add_argument("--failure-screenshot", action="store_true")
    run.add_argument("--poll-interval", type=float, default=0.1,
                     help="Maximum polling sleep for on-demand capture (default: 0.1 s).")
    run.add_argument("--ocr-threads", type=int, default=2)
    run.add_argument("--ocr-backend", choices=["auto", "cpu", "openvino"], default="auto")
    run.add_argument("--ocr-max-side", type=int,
                     help="Optional OCR input limit; validate small text for your application.")
    run.add_argument(
        "--profile",
        type=Path,
        help="Trusted Python file exposing run_intent(session, path) for application-specific workflows.",
    )
    run.add_argument(
        "--catalog",
        type=Path,
        help="Shared named selectors and busy states; defaults to a sibling ui.json if present.",
    )
    grab = commands.add_parser(
        "snapshot", help="Save a screenshot or a cropped template."
    )
    grab.add_argument("path", type=Path)
    grab.add_argument(
        "--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT")
    )
    tree = commands.add_parser(
        "hierarchy", help="Save native UI attributes and bounds."
    )
    tree.add_argument("path", type=Path)
    for cmd in (run, grab, tree):
        cmd.add_argument("--serial")
        cmd.add_argument("--adb", default="adb")
        cmd.add_argument("--capture", choices=("auto", "jpeg", "png", "scrcpy"), default="auto")
        cmd.add_argument("--scrcpy-server", type=Path)
        cmd.add_argument("--timeout", type=float, default=12)
    migrate = commands.add_parser(
        "convert",
        help="Convert .air or .py into a new directory and report migration issues.",
    )
    migrate.add_argument("source", type=Path)
    migrate.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_suite(args)
        if args.command == "devices":
            for serial in devices(args.adb):
                print(serial)
        elif args.command == "convert":
            report = convert(args.source, args.out)
            print(json.dumps(report, indent=2))
            return 2 if report["issues"] else 0
        else:
            with Session(
                args.serial,
                adb=args.adb,
                capture=args.capture,
                scrcpy_server=args.scrcpy_server,
            ) as session:
                if args.command == "hierarchy":
                    from xml.etree.ElementTree import ElementTree

                    args.path.parent.mkdir(parents=True, exist_ok=True)
                    ElementTree(session.device.hierarchy()).write(
                        args.path, encoding="utf-8", xml_declaration=True
                    )
                elif args.crop:
                    import cv2

                    image = session.frame()
                    x, y, w, h = args.crop
                    if (
                        min(x, y) < 0
                        or min(w, h) < 1
                        or x + w > image.shape[1]
                        or y + h > image.shape[0]
                    ):
                        raise ValueError("Crop outside captured screen.")
                    args.path.parent.mkdir(parents=True, exist_ok=True)
                    ok, encoded = cv2.imencode(
                        args.path.suffix, image[y : y + h, x : x + w]
                    )
                    if not ok:
                        raise ValueError("Image encoding failed.")
                    args.path.write_bytes(encoded.tobytes())
                else:
                    session.snapshot(args.path)
        return 0
    except (RuntimeError, ValueError, OSError, SyntaxError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
