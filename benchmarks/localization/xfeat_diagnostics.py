"""Post-run XFeat positive controls; reference-centered crops are NOT estimators."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from aria_trace.services.tracking.runtime import MinimapExtractor
from benchmarks.localization.xfeat_cpu import ATLAS, CALIBRATION, XFeatAdapter, mnn_matches
from benchmarks.localization.run_workbench_replay import read_rows
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog


def correspondences(a, b, expected_shift):
    pa, da = a
    pb, db = b
    if da is None or db is None:
        return {"features": [len(pa), len(pb)], "mnn": 0, "ratio": 0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    ratio = [p for p, q in pairs if p.distance < .8*q.distance]
    mnn = mnn_matches(da, db)
    result = {"features": [len(pa), len(pb)]}
    for name, matches in (("ratio", ratio), ("mnn", mnn)):
        result[name] = len(matches)
        if matches:
            source = np.float32([pa[m.queryIdx].pt for m in matches])
            target = np.float32([pb[m.trainIdx].pt for m in matches])
            errors = np.linalg.norm(target-source-np.asarray(expected_shift), axis=1)
            result[name+"_within_3px"] = int(sum(errors <= 3))
            result[name+"_median_error_px"] = float(np.median(errors))
        if len(matches) >= 6:
            cv2.setRNGSeed(0)
            affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                ransacReprojThreshold=3, maxIters=20000, confidence=.999)
            result[name+"_ransac_inliers"] = int(inliers.sum()) if inliers is not None else 0
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(".tools/xfeat-upstream").resolve()))
    from modules.xfeat import XFeat
    model = XFeat(weights=str(Path(".tools/xfeat-upstream/weights/xfeat.pt").resolve()))
    adapter = XFeatAdapter(model)
    root = Path("artifacts/workbench")
    atlas = root / "map_atlases/genshin-impact-pc" / ATLAS
    manifest = json.loads((atlas/"map_atlas.json").read_text())
    layers = {r["mode_id"]: r for r in manifest["layers"]}
    config = ProfileCatalog().game("genshin-impact-pc")["minimap_calibration"]
    extractor = MinimapExtractor(config["crop_xywh"], json.loads((root/"minimap_calibrations/genshin-impact-pc"/CALIBRATION/"calibration.json").read_text()))
    refs = json.loads(Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json").read_text())
    results = []
    for number, time_s in ((11, 0), (17, 25)):
        reference = read_rows(Path(refs[str(number)])/"route_states.jsonl")
        target = min(reference, key=lambda r: abs(r["session_time_ns"]/1e9-time_s))
        frame_index = target["source_frame_index"]
        capture = cv2.VideoCapture(str(Path("sessions/workbench/recordings-genshin-impact-pc")/f"run_{number:02d}/video_main.mkv"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("Diagnostic frame unavailable")
        observation, mask = extractor.extract(image)
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        layer = layers[target["mode_id"]]
        mosaic = cv2.imread(str(atlas/layer["localization_mosaic_file"]))
        map_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        coverage = cv2.imread(str(atlas/layer["localization_coverage_file"]), 0)
        transform = np.linalg.inv(layer["localization_to_canonical_3x3"])
        center = transform @ np.array([*target["canonical_xy"], 1])
        h, w = gray.shape
        left, top = int(round(center[0]-w/2)), int(round(center[1]-h/2))
        patch = map_gray[top:top+h, left:left+w].copy()
        a = adapter.detectAndCompute(gray, mask)
        b = adapter.detectAndCompute(map_gray, cv2.erode(coverage, np.ones((5,5), np.uint8)))
        same = adapter.detectAndCompute(gray.copy(), mask)
        pcrop = adapter.detectAndCompute(patch, mask)
        self_crop = adapter.detectAndCompute(patch, mask)
        record = {"session": number, "source_frame_index": frame_index,
            "reference_time_s": target["session_time_ns"]/1e9, "mode": target["mode_id"],
            "role": "reference-centered positive controls only, not localization performance",
            "observation_shape": list(gray.shape),
            "identical_observation": correspondences(a, same, (0,0)),
            "native_to_full_map": correspondences(a, b, (left,top)),
            "native_to_reference_crop": correspondences(a, pcrop, (0,0)),
            "exact_map_crop_to_full_map": correspondences(self_crop, b, (left,top))}
        results.append(record)
        print(json.dumps(record), flush=True)
        # Native minimap versus reference-centered atlas crop, enlarged for QA.
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        pair = np.concatenate([observation, patch_bgr], axis=1)
        pair = cv2.resize(pair, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(args.output/f"run{number:02d}-native-and-reference-crop.png"), pair)
    (args.output/"diagnostics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
