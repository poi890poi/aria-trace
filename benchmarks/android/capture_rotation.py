"""Keep one direct-capture helper alive across app rotation and fresh UI changes."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from swiftest import Android, Session, Node, TextPattern
from latency_fixture import PACKAGE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--serial",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--capture",choices=("auto","direct","bitmap","scrcpy"),default="direct")
    parser.add_argument("--scrcpy-server",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    admin=Android(args.serial)
    prior=admin.foreground()
    installed="package:"+PACKAGE in admin.shell("pm","list","packages",PACKAGE).splitlines()
    rows=[]
    try:
        if not installed:
            admin.install(ROOT/"artifacts/androidtestkit/variants-fixture/fixture.apk")
        with Session(device=Android(args.serial),capture=args.capture,
                     scrcpy_server=args.scrcpy_server) as session:
            session.device.package=PACKAGE
            backend=None
            for index,variant in enumerate((0,2,1)):
                admin.shell("am","force-stop",PACKAGE)
                admin.shell("am","start","-W","-n",PACKAGE+"/.MainActivity","--ei","variant",variant,"--ez","latency_probe","true")
                session.touch(Node(text=TextPattern(("Close","關閉","关闭")[variant])))
                before=session.frame().copy()
                if backend is None:
                    backend=session._backend
                assert backend is session._backend
                session.touch(Node(description="primary-action"))
                assert session.exists(Node(text=("Completed" if variant==0 else "完成")+" 1"))
                after=session.frame()
                assert not session.capture_repeated
                assert not np.array_equal(before,after)
                independent=cv2.imdecode(np.frombuffer(admin.run("exec-out","screencap","-p"),np.uint8),cv2.IMREAD_COLOR)
                assert after.shape==independent.shape
                assert (after.shape[1]>after.shape[0]) == (variant==2)
                cv2.imwrite(str(args.output/f"{index}-direct.png"),after)
                cv2.imwrite(str(args.output/f"{index}-independent.png"),independent)
                rows.append(dict(variant=variant,shape=list(after.shape),mode=session.report()["capture_acquisition"],fresh=True))
        (args.output/"result.json").write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows),flush=True)
    finally:
        admin.shell("am","force-stop",PACKAGE)
        if not installed:
            admin.uninstall(PACKAGE)
        if prior:
            admin.launch(prior)
        admin.close()


if __name__=="__main__":
    main()
