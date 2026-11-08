"""Capture fresh popup/wording/layout examples with native UI ground truth."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swiftest import Session, Node, TextPattern

PACKAGE = "org.swiftest.variantsfixture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    with Session(args.serial) as session:
        try:
            session.device.shell("input", "keyevent", "224")
            session.device.shell("input", "keyevent", "82")
            for variant in range(3):
                session.stop(PACKAGE)
                session.device.shell("am", "start", "-W", "-n", PACKAGE+"/.MainActivity", "--ei", "variant", str(variant))
                for phase, labels in (("popup", ["Close", "關閉", "关闭"]),
                                      ("button", ["Continue", "繼續", "继续"])):
                    pattern = TextPattern(labels)
                    target = Node(text=pattern) if phase == "popup" else Node(description="primary-action")
                    session.wait(target, timeout=10)
                    time.sleep(.3)
                    image = args.output / f"variant-{variant}-{phase}.png"
                    image.write_bytes(session.device.run("exec-out", "screencap", "-p"))
                    tree = session.device.hierarchy()
                    nodes = [n for n in tree.iter("node") if (pattern.matches(n.get("text", "")) if phase == "popup"
                             else n.get("content-desc") == "primary-action")]
                    assert len(nodes) == 1
                    l, t, r, b = map(int, re.findall(r"\d+", nodes[0].get("bounds")))
                    cases.append(dict(name=f"variant-{variant}-{phase}", image=str(image.resolve()),
                                      query=dict(labels=labels, region=[.1,.25,.9,.75] if phase == "popup" else [.02,.08,.98,.5]),
                                      expected="unique", target_box=[l,t,r-l,b-t],
                                      reference="Android native hierarchy; fixture source labels/layout", reference_text=nodes[0].get("text")))
                    if phase == "popup": session.touch(target)
            (args.output / "manifest.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
        finally:
            session.stop(PACKAGE)
            session.device.shell("input", "keyevent", "223")


if __name__ == "__main__":
    main()
