"""Paired offline HUD OCR experiment; does not replace full-screen error checks."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import cv2
    from swiftest import LocalOCR
    from examples.genshin.profile import has_world_hud, ocr_image
    reader = LocalOCR(max_side=1280, threads=4, backend="openvino")
    records = json.loads((args.directory/"analysis.json").read_text())["rows"]
    records = [r for r in records if r["case"]!="00-cold" and r["timed"] and not r["repeated"]]
    output=[]
    for index, record in enumerate(records):
        image = cv2.imread(record["path"])
        prepared = ocr_image(image)
        digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        h,w = image.shape[:2]
        def recognize(mode):
            regions = [(0,0,w,h)] if mode=="full" else [
                (int(.7*w),int(.85*h),w,h),
                (int(.3*w),int(.82*h),int(.68*w),int(.98*h))]
            lines=[]
            for l,t,r,b in regions:
                for text,box in reader(prepared[t:b,l:r]):
                    lines.append(dict(text=text,words=[dict(text=text,x=box.x+l,
                        y=box.y+t,width=box.w,height=box.h)]))
            observation=dict(width=w,height=h,lines=lines)
            return dict(hud=has_world_hud(prepared,observation),observation=observation)
        # One shape/model warmup, followed by two alternating paired passes.
        if index==0:
            recognize("full")
            recognize("roi")
        for repeat in range(2):
            modes = ["full","roi"] if (index+repeat)%2==0 else ["roi","full"]
            for mode in modes:
                began=time.perf_counter()
                value=recognize(mode)
                elapsed=time.perf_counter()-began
                output.append(dict(path=record["path"],case=record["case"],
                    original_state=record["state"],mode=mode,repeat=repeat,
                    sha256=digest,seconds=elapsed,**value))
    summary={mode:dict(calls=sum(r["mode"]==mode for r in output),
                median_ms=1000*statistics.median(r["seconds"] for r in output if r["mode"]==mode),
                accepted=sum(r["hud"] for r in output if r["mode"]==mode))
             for mode in ("full","roi")}
    (args.directory/"roi-replay.json").write_text(json.dumps(dict(summary=summary,rows=output),indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
