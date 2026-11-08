"""Same-image CPU OCR ablations, retaining full-screen detection and text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from swiftest import LocalOCR
from steady_latency import summary


def canonical(lines):
    return [(text, np.rint(corners).astype(int).tolist()) for corners,text,score in lines or ()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=json.loads(args.manifest.read_text())
    images=[cv2.imread(c["image"]) for c in cases]
    engines={}
    for threads in (2,4,8):
        reader=LocalOCR(max_side=1280,threads=threads,backend="openvino")
        reader(images[0])
        engines[threads]=reader.engine
    modes=[(4,True),(2,True),(8,True),(4,False)]
    rows=[]
    reference={}
    for index,image in enumerate(images):
        reference[index]=canonical(engines[4](image)[0])
    for repeat in range(4):
        for index,image in enumerate(images):
            for threads,cls in (modes if repeat%2==0 else modes[::-1]):
                start=time.perf_counter()
                lines,stages=engines[threads](image,use_cls=cls)
                ms=(time.perf_counter()-start)*1000
                if repeat:
                    rows.append(dict(name=cases[index]["name"],threads=threads,classification=cls,
                        ms=ms,stages=stages,matches_baseline=canonical(lines)==reference[index],
                        output=canonical(lines)))
    result={f"threads{t}-cls{c}":dict(timing=summary([r["ms"] for r in rows if (r["threads"],r["classification"])==(t,c)]),
        matches=sum(r["matches_baseline"] for r in rows if (r["threads"],r["classification"])==(t,c))) for t,c in modes}
    (args.output/"result.json").write_text(json.dumps(dict(rows=rows,summary=result,
        inputs={c["image"]:hashlib.sha256(Path(c["image"]).read_bytes()).hexdigest() for c in cases},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
