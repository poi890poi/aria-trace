"""Validate batched input against native counters before using it in the game."""
import argparse
import json
from pathlib import Path
import sys
from functools import partial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest import Android, Session, Node, TextPattern, DeviceError
from e2e_candidates import checked_tap
from latency_fixture import PACKAGE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--narrow", action="store_true")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    checked = partial(checked_tap, compact=args.compact, narrow=args.narrow)
    if args.production:
        from swiftest.guarded import checked_tap as production_tap
        checked = partial(production_tap, blocked_window="GameBooster Lock Screen")
    args.output.mkdir(parents=True, exist_ok=False)
    admin = Android(args.serial)
    prior = admin.foreground()
    installed = "package:"+PACKAGE in admin.shell("pm", "list", "packages", PACKAGE).splitlines()
    rows = []
    try:
        if not installed:
            admin.install(ROOT/"artifacts/androidtestkit/variants-fixture/fixture.apk")
        for variant in range(3):
            admin.shell("am","force-stop",PACKAGE)
            admin.shell("am","start","-W","-n",PACKAGE+"/.MainActivity","--ei","variant",variant,
                        "--ez","latency_probe","true")
            with Session(device=Android(args.serial), capture="direct") as session:
                session.device.package = PACKAGE
                session.touch(Node(text=TextPattern(("Close","關閉","关闭")[variant])))
                x,y = session.wait(Node(description="primary-action"))
                image = session.frame()
                h,w = image.shape[:2]
                for package, width in (("org.wrong.package",w),(PACKAGE,w+100)):
                    try:
                        checked(session.device,x,y,width,h,package)
                    except DeviceError:
                        pass
                    else:
                        raise AssertionError("Guard did not reject mismatch")
                assert not checked(session.device,x,y,w,h,PACKAGE,blocked_window=PACKAGE)
                for count in range(1,4):
                    assert checked(session.device,x,y,w,h,PACKAGE)
                    expected=("Completed" if variant==0 else "完成")+" "+str(count)
                    assert session.exists(Node(text=expected))
                rows.append(dict(variant=variant,shape=list(image.shape),count=3,
                                 acquisition=session.report()["capture_acquisition"]))
        (args.output/"result.json").write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows),flush=True)
    finally:
        admin.shell("am","force-stop",PACKAGE)
        if not installed:
            admin.uninstall(PACKAGE)
        if prior:
            admin.launch(prior)
        admin.close()


if __name__ == "__main__":
    main()
