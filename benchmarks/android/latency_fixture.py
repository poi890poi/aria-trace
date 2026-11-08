"""Paired warm taps in the disposable native fixture, with independent UI checks."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest import Node, Pixel, Session, TextPattern
from swiftest.device import Android, DeviceError
from steady_latency import summary

PACKAGE = "org.swiftest.variantsfixture"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/"swiftest/device.py", ROOT/"swiftest/shell.py",
               ROOT/"tests/android_variants_fixture/MainActivity.java"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    admin = Android(args.serial, persistent_shell=False)
    prior = admin.foreground()
    installed = "package:" + PACKAGE in admin.shell("pm", "list", "packages", PACKAGE).splitlines()
    rows = []
    try:
        if not installed:
            admin.install(ROOT / "artifacts/androidtestkit/variants-fixture/fixture.apk")
        admin.shell("input", "keyevent", "KEYCODE_WAKEUP")
        admin.shell("input", "keyevent", "82")
        for variant in range(3):
            admin.shell("am", "force-stop", PACKAGE)
            admin.shell("am", "start", "-W", "-n", PACKAGE + "/.MainActivity", "--ei", "variant", variant,
                        "--ez", "latency_probe", "true")
            with Session(device=Android(args.serial)) as setup:
                setup.device.package = PACKAGE
                setup.touch(Node(text=TextPattern(("Close", "關閉", "关闭")[variant])), timeout=8)
                point = setup.wait(Node(description="primary-action"), timeout=8)
            if args.capture:
                subprocess.run([sys.executable, str(Path(__file__).with_name("capture_latency.py")),
                    "--serial", args.serial, "--output", str(args.output/f"capture-{variant}"),
                    "--rounds", str(args.rounds), "--force-encode"], check=True, timeout=90)
            expected_clicks = 0
            for persistent in (False, True, True, False):
                phone = Android(args.serial, persistent_shell=persistent)
                with Session(device=phone) as session:
                    phone.package = PACKAGE
                    session.touch(Pixel(*map(round, point)))  # warmup excluded
                    for i in range(args.rounds):
                        parts = {}
                        start = time.perf_counter()
                        check = start
                        assert phone.foreground() == PACKAGE
                        parts["initial_focus_ms"] = (time.perf_counter()-check)*1000
                        check = time.perf_counter()
                        phone.run("shell", "dumpsys input | grep 'GameBooster Lock Screen' | head -1")
                        parts["overlay_ms"] = (time.perf_counter()-check)*1000
                        check = time.perf_counter()
                        session.touch(Pixel(*map(round, point)))
                        parts["session_touch_ms"] = (time.perf_counter()-check)*1000
                        rows.append(dict(variant=variant, mode="persistent" if persistent else "process",
                                         ms=(time.perf_counter()-start)*1000, **parts))
                    expected_clicks += args.rounds + 1
                    expected_text = ("Completed" if variant == 0 else "完成") + " " + str(expected_clicks)
                    assert session.exists(Node(text=expected_text))
                    # Fail closed if focus changes between recognition and dispatch.
                    phone.package = "org.swiftest.wrongpackage"
                    try:
                        session.touch(Pixel(*map(round, point)))
                    except DeviceError:
                        pass
                    else:
                        raise AssertionError("Focus mismatch permitted input")
                    assert session.exists(Node(text=expected_text))
            print("Verified variant", variant, flush=True)
        result = {mode: summary([r["ms"] for r in rows if r["mode"] == mode])
                  for mode in ("process", "persistent")}
        assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (args.output / "result.json").write_text(json.dumps(dict(
            rows=rows, summary=result, serial=args.serial,
            hashes=hashes), indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        admin.shell("am", "force-stop", PACKAGE)
        if not installed:
            admin.uninstall(PACKAGE)
        if prior:
            admin.launch(prior)
        admin.close()


if __name__ == "__main__":
    main()
