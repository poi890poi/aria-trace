"""Compare unused native map viewports with the rebuilt mosaic at native scale."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import distribution


def run(stitch_path, output):
    output.mkdir(parents=True, exist_ok=True)
    stitch = json.loads(stitch_path.read_text())
    mosaic = cv2.imread(str(stitch_path.parent / "mosaic.png"))
    coverage = cv2.imread(str(stitch_path.parent / "coverage.png"), 0)
    sift = cv2.SIFT_create(nfeatures=16000, contrastThreshold=.015)
    map_keys, map_desc = sift.detectAndCompute(cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY), coverage)
    selected = set(stitch["selected_frame_indices"])
    candidates = [i for i in range(20, stitch["source_frame_count"]-20) if i not in selected]
    wanted = {candidates[i] for i in np.linspace(0, len(candidates)-1, 20).astype(int)}
    video = Path(stitch["provenance"]["source_session_path"]) / "video_main.mkv"
    capture = cv2.VideoCapture(str(video))
    x,y,w,h = stitch["viewport_crop_xywh"]
    rows=[]
    index=0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            wanted.remove(index)
            native = frame[y:y+h,x:x+w]
            points, desc = sift.detectAndCompute(cv2.cvtColor(native, cv2.COLOR_BGR2GRAY), None)
            matches = [a for pair in cv2.BFMatcher().knnMatch(desc,map_desc,k=2) if len(pair)==2 for a,b in [pair] if a.distance < .72*b.distance]
            row={"frame_index":index,"used_in_composition":False,"matches":len(matches)}
            if len(matches)>=12:
                src=np.float32([points[m.queryIdx].pt for m in matches]); dst=np.float32([map_keys[m.trainIdx].pt for m in matches])
                _, inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
                if inliers is not None and int(inliers.sum())>=12:
                    keep=inliers.ravel().astype(bool)
                    shift=np.median(dst[keep]-src[keep],axis=0)
                    residual=np.linalg.norm(dst[keep]-src[keep]-shift,axis=1)
                    aligned=cv2.warpAffine(mosaic,np.float32([[1,0,-shift[0]],[0,1,-shift[1]]]),(w,h))
                    mask=cv2.warpAffine(coverage,np.float32([[1,0,-shift[0]],[0,1,-shift[1]]]),(w,h))>250
                    mask[:20]=False;mask[-20:]=False;mask[:,:20]=False;mask[:,-20:]=False
                    diff=np.abs(native.astype(float)-aligned.astype(float)).mean(axis=2)
                    row.update({"translation_xy":shift.tolist(),"inliers":int(keep.sum()),"translation_residual_px":distribution(residual),"rgb_absolute_error":distribution(diff[mask]),"covered_fraction":float(mask.mean())})
                    # Original pixels remain unscaled; only the mosaic is translated.
                    panel=np.full((h+44,w*2,3),25,np.uint8)
                    panel[44:,:w]=native;panel[44:,w:]=aligned
                    cv2.putText(panel,f"Native frame {index} (unused source)",(12,29),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                    cv2.putText(panel,"Rebuilt mosaic - same area, native scale",(w+12,29),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                    filename=f"frame_{index:06d}.jpg";cv2.imwrite(str(output/filename),panel)
                    row["comparison"]=filename
            rows.append(row)
        index+=1
    capture.release()
    report={"protocol":"20 deterministic frames excluded from source-owner composition; SIFT correspondence and best translation only; local self-consistency, not external geographic truth. RGB differences include animation and tone. RANSAC-selected feature residuals omit unmatched/mismatching content; inspect native-scale comparisons.","stitch":identity(stitch_path),"mosaic":identity(stitch_path.parent/'mosaic.png'),"video":identity(video),"implementation":identity(__file__),"rows":rows}
    (output/"results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps([{k:r[k] for k in ('frame_index','inliers','translation_residual_px','rgb_absolute_error') if k in r} for r in rows],indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stitch",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.stitch,a.output)
