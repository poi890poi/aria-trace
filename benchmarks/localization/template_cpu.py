"""Experimental translation-only global atlas matching, without SIFT proposals."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from aria_trace.services.tracking import runtime
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import run
from benchmarks.localization.xfeat_cpu import ATLAS, CALIBRATION, frozen_method, probe
from benchmarks.localization.known_start import read_start, installed_start


def localize_translation(self, observation, mask, yaw_prior_deg=None,
                         search_center_xy=None, search_radius_px=None):
    started = time.perf_counter()
    if self._cancel.is_set():
        raise RuntimeError("Global localization canceled")
    if mask.shape != observation.shape[:2] or not np.any(mask):
        return self._invalid(started, ("empty-or-invalid-observation-mask",))
    gradient = self._template_representation == "gradient"
    template = runtime._gradient(observation) if gradient else cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if float(np.std(template[mask > 0])) < 1e-6:
        return self._invalid(started, ("constant-observation",))
    atlas = self.map_gradient if gradient else self.map_gray
    h, w = template.shape
    mh, mw = atlas.shape
    if h >= mh or w >= mw:
        return self._invalid(started, ("observation-exceeds-map",))
    left = top = 0
    right, bottom = mw, mh
    if search_center_xy is not None and search_radius_px is not None:
        x, y = self._localization_xy(search_center_xy)
        sx = math.hypot(self.original_to_localization[0,0], self.original_to_localization[1,0])
        sy = math.hypot(self.original_to_localization[0,1], self.original_to_localization[1,1])
        radius = max(16.0, float(search_radius_px)*(sx+sy)/2)
        left, top = max(0, math.floor(x-radius-w/2)), max(0, math.floor(y-radius-h/2))
        right, bottom = min(mw, math.ceil(x+radius+w/2)), min(mh, math.ceil(y+radius+h/2))
        if right-left <= w or bottom-top <= h:
            return self._invalid(started, ("insufficient-bounded-search-area",))
    search = atlas[top:bottom, left:right]
    method = cv2.TM_CCORR_NORMED if gradient else cv2.TM_CCOEFF_NORMED
    response = cv2.matchTemplate(search, template, method, mask=mask)
    response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
    suppressed = response.copy()
    peaks = []
    for _ in range(3):
        _, score, _, location = cv2.minMaxLoc(suppressed)
        if score <= -1.0:
            break
        center = (left+location[0]+w/2, top+location[1]+h/2)
        peaks.append((float(score), location, center))
        cv2.circle(suppressed, location, max(8,min(h,w)//3), -1, -1)
    if not peaks:
        return self._invalid(started, ("no-finite-correlation",))
    score, location, center = peaks[0]
    # A bounded window may have no peak outside the suppression neighborhood.
    # Do not turn its -1 sentinel into a fictitious alternative or extra margin.
    margin = score-max((p[0] for p in peaks[1:]), default=0.0)
    px, py = left+location[0], top+location[1]
    coverage = float(np.mean(self.coverage[py:py+h, px:px+w][mask>0] > 0))
    center_covered = bool(self.coverage[int(center[1]), int(center[0])])
    reasons = []
    if score < .55:
        reasons.append("low-correlation")
    if margin < .06:
        reasons.append("ambiguous-correlation")
    if coverage < .75 or not center_covered:
        reasons.append("insufficient-observed-coverage")
    x, y = self._original_xy(center)
    sx = math.hypot(self.localization_to_original[0,0], self.localization_to_original[1,0])
    sy = math.hypot(self.localization_to_original[0,1], self.localization_to_original[1,1])
    # Do not populate feature inliers or agreement with synthetic successes.
    diagnostics = {"observation": observation.copy(), "mask": mask.copy(),
                   "proposal_method": "translation-only-"+self._template_representation,
                   "coverage_fraction": coverage, "in_layer_scale": 1.0,
                   "in_layer_rotation_deg": 0.0, "feature_verification": "absent",
                   "distinct_peak_count": len(peaks)}
    return runtime.GlobalFix(x,y,0.0,(sx+sy)/2,score,margin,(time.perf_counter()-started)*1000,
        valid=not reasons, rejection_reasons=tuple(reasons),
        alternatives=tuple(dict(zip(("x","y"),self._original_xy(c)),score=s) for s,_,c in peaks),
        search_bounds_xyxy=(left,top,right,bottom), search_area_fraction=search.size/atlas.size,
        diagnostics=diagnostics)


@contextmanager
def installed(representation, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"variant": "template-"+representation, "feature_scale": 1,
        "base": "f607fa6", "python": platform.python_version(), "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(), "setup": [], "implementation": identity(__file__)}
    (output/"harness.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    source = frozen_method("__init__")
    start, end = source.index("    self.sift ="), source.index("    self._cancel =")
    source = source[:start]+'    self.map_gray = cv2.cvtColor(self.mosaic, cv2.COLOR_BGR2GRAY).astype(np.float32)\n'+source[end:]
    namespace = dict(vars(runtime))
    exec(compile(source,str(output/"initializer.py"),"exec"),namespace)
    init = namespace["__init__"]
    (output/"initializer.py").write_text(source, encoding="utf-8")
    (output/"localize.py").write_text(inspect.getsource(localize_translation), encoding="utf-8")
    original_init, original_localize = runtime.GlobalMapLocalizer.__init__, runtime.GlobalMapLocalizer.localize
    calls = (output/"global_calls.jsonl").open("x")
    def initialize(self,*args,**kwargs):
        wall,cpu = time.perf_counter(),time.process_time()
        init(self,*args,**kwargs)
        self._template_representation = representation
        metadata["setup"].append({"map_shape_hw": list(self.mosaic.shape[:2]),
            "wall_ms": (time.perf_counter()-wall)*1000, "process_cpu_ms": (time.process_time()-cpu)*1000})
    def localize(self,observation,mask,*args,**kwargs):
        wall,cpu = time.perf_counter(),time.process_time()
        fix = localize_translation(self,observation,mask,*args,**kwargs)
        calls.write(json.dumps({"map_shape_hw":list(self.mosaic.shape[:2]),
            "observation_sha256":hashlib.sha256(observation.tobytes()).hexdigest(),
            "wall_ms":(time.perf_counter()-wall)*1000,"process_cpu_ms":(time.process_time()-cpu)*1000,
            "valid":fix.valid,"xy":[fix.x,fix.y],"score":fix.score,"margin":fix.margin,
            "reasons":fix.rejection_reasons})+"\n")
        calls.flush()
        return fix
    try:
        runtime.GlobalMapLocalizer.__init__,runtime.GlobalMapLocalizer.localize = initialize,localize
        (output/"manifest.json").write_text(json.dumps(metadata,indent=2))
        yield metadata
    finally:
        runtime.GlobalMapLocalizer.__init__,runtime.GlobalMapLocalizer.localize = original_init,original_localize
        calls.close()
        (output/"manifest.json").write_text(json.dumps(metadata,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation",choices=["existing","gradient","gray"],required=True)
    p.add_argument("--action",choices=["probe","replay"],required=True)
    p.add_argument("--runs",nargs="+",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-seconds",type=float)
    p.add_argument("--rate",type=float,default=1.0)
    p.add_argument("--mode",default="free-roam",choices=["free-roam","route-assisted"])
    p.add_argument("--start-policy",choices=["none","search","prior"],default="none")
    p.add_argument("--start-route",type=int,default=11)
    p.add_argument("--start-offset",nargs=2,type=float,default=[0.,0.])
    p.add_argument("--start-radius",type=float,default=90.)
    args=p.parse_args()
    if args.output.exists():
        raise RuntimeError("Use a new output directory")
    args.atlas,args.calibration=ATLAS,CALIBRATION
    args.scene_yaw="01dbaa74-8e00-4763-a215-9ea37e18b1b2"
    args.cache=Path("artifacts/benchmark_cache/atlas_references")
    args.references=Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json")
    args.references_only=args.record_video=False
    args.loss_error_limit_px=None
    hint=None
    if args.start_policy!="none":
        if args.action!="replay":
            raise ValueError("Known-start tests use the stateful replay loop")
        references=json.loads(args.references.read_text())
        hint=read_start(Path(references[str(args.start_route)])/"route_states.jsonl",args.start_offset)
    candidate = (installed(args.representation,args.output/"candidate-source") if args.representation!="existing"
                 else nullcontext({"variant":"existing","setup":[],"base":"f607fa6"}))
    with candidate as metadata:
        metadata["known_start"]={"policy":args.start_policy,"hint":hint,"radius_px":args.start_radius}
        if hint:
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/"known_start_source.py").write_text(Path(inspect.getfile(read_start)).read_text(),encoding="utf-8")
        args.experiment=metadata
        with installed_start(hint,args.start_policy,args.start_radius) if hint else nullcontext():
            probe(args) if args.action=="probe" else run(args)


if __name__=="__main__":
    main()
