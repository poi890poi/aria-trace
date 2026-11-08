"""Replay human-reviewed arrival screenshots; no device input or policy changes."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--engine",choices=("testkit","airtest"),required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=ROOT/"artifacts/airtest-benchmark/assets")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import cv2
    if args.engine=="testkit":
        from swiftest import LocalOCR
        from examples.genshin.profile import classify,has_world_hud,ocr_image
        reader=LocalOCR(max_side=1280,threads=4,backend="openvino")
        def recognize(image):
            prepared=ocr_image(image)
            lines=[dict(text=text,words=[dict(text=text,x=box.x,y=box.y,
                width=box.w,height=box.h)]) for text,box in reader(prepared)]
            observation=dict(width=image.shape[1],height=image.shape[0],lines=lines)
            state=classify(observation,has_world_hud(prepared,observation))
            return dict(world=state=="world",state=state,text=[l["text"] for l in lines])
    else:
        from examples.genshin.airtest_teleport import Teleport
        from airtest.core.cv import Template
        from airtest.core.settings import Settings
        logging.getLogger("airtest").setLevel(logging.WARNING)
        Settings.CVSTRATEGY=["tpl"]
        runner=Teleport.__new__(Teleport)
        runner.manifest=json.loads((args.assets/"manifest.json").read_text())
        runner.templates={name:Template(str(args.assets/spec["file"]),threshold=spec["threshold"])
                          for name,spec in runner.manifest["templates"].items()}
        runner.match_s=0
        def recognize(image):
            runner.frame=image
            return dict(world=bool(runner.world()))
    manifest=json.loads(args.manifest.read_text())
    rows=[]
    for item in manifest:
        path=Path(item["path"])
        assert item["expected"]=="world" and item["human_reviewed"] is True
        image=cv2.imread(str(path))
        assert image is not None
        start=time.perf_counter()
        result=recognize(image)
        rows.append(dict(**item,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         **result,seconds=time.perf_counter()-start))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(engine=args.engine,rows=rows,
        accepted=sum(r["world"] for r in rows),total=len(rows)),indent=2),encoding="utf-8")
    print(json.dumps(dict(engine=args.engine,accepted=sum(r["world"] for r in rows),total=len(rows))))


if __name__=="__main__":
    main()
