"""Isolated follow-up experiments on production integration b6c6125."""

import argparse
import ast
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import textwrap

from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker as Runtime
from aria_trace.services.localization.route.tracker import RouteVisualTracker as Visual
from benchmarks.localization.tracking_candidates import replace_once
from benchmarks.localization.run_workbench_replay import run

BASE = "b6c6125"
CHANGES = {
    "K": "Wider current-image refinement after a confirmed layer switch, retaining continuity gating until accepted.",
    "L": "Observe representation at up to source 30 Hz while a transition is armed; keep visual confirmation thresholds.",
    "M": "Use first valid initialization hypothesis as a bounded next-search proposal; invalid fixes clear it normally.",
}


@lru_cache(maxsize=None)
def method(cls, name):
    root = Path(__file__).resolve().parents[2]
    path = Path(inspect.getfile(cls)).resolve().relative_to(root).as_posix()
    source = subprocess.check_output(["git", "show", BASE+":"+path], cwd=root, text=True, encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent("".join(source.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]))


def sources(variant):
    if set(variant)-set(CHANGES):
        raise ValueError("Unknown variant: "+variant)
    values = {(c,n): method(c,n) for c,n in (
        (Visual,"track"), (Visual,"confirm_trained_transition_layer"),
        (Runtime,"update"), (Runtime,"_global_search"))}
    if "K" in variant:
        key = Visual,"confirm_trained_transition_layer"
        values[key] = replace_once(values[key], '    pending = self._trained_transition\n',
            '    self._wide_after_mode_change = True\n    pending = self._trained_transition\n')
        key = Visual,"track"
        values[key] = replace_once(values[key], 'self.previous_time_ns is None else self.local_radius_px',
            '(self.previous_time_ns is None or getattr(self, "_wide_after_mode_change", False)) else self.local_radius_px')
        values[key] = replace_once(values[key], '    if measurement_accepted:\n        self.previous_xy',
            '    if measurement_accepted:\n        self._wide_after_mode_change = False\n        self.previous_xy')
    if "L" in variant:
        key = Runtime,"update"
        values[key] = replace_once(values[key], '        >= self.representation_interval_ns\n',
            '        >= (min(self.representation_interval_ns, int(1e9/30))\n'
            '            if ((self._last_representation_observation or {}).get("controller") or {}).get("transition_armed")\n'
            '            else self.representation_interval_ns)\n')
    if "M" in variant:
        key = Runtime,"_global_search"
        values[key] = replace_once(values[key], '    if self.fusion._state is None:\n        return None, None\n',
            '    if self.fusion._state is None:\n'
            '        if self._initial_hypotheses:\n'
            '            fix = self._initial_hypotheses[-1]\n'
            '            return (float(fix.x), float(fix.y)), 150.0\n'
            '        return None, None\n')
    return values


@contextmanager
def installed(variant, output=None):
    saved = {}
    metadata = {"variant":variant or "baseline", "base":BASE,
                "changes":{k:CHANGES[k] for k in variant}, "methods":[]}
    try:
        for (cls,name), source in sources(variant).items():
            saved[cls,name] = inspect.getattr_static(cls,name)
            namespace = vars(inspect.getmodule(cls))
            sentinel = object()
            previous = namespace.get(name,sentinel)
            filename = f"{variant or 'baseline'}-{cls.__name__}-{name}.py"
            try:
                exec(compile(source,filename,"exec"),namespace)
                setattr(cls,name,namespace[name])
            finally:
                if previous is sentinel:
                    namespace.pop(name,None)
                else:
                    namespace[name] = previous
            if output:
                output.mkdir(parents=True,exist_ok=True)
                (output/filename).write_text(source,encoding="utf-8")
            metadata["methods"].append({"class":cls.__name__,"method":name,
                "sha256":hashlib.sha256(source.encode()).hexdigest(),"source_file":filename})
        yield metadata
    finally:
        for (cls,name), original in saved.items():
            setattr(cls,name,original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",default="baseline")
    p.add_argument("--runs",nargs="+",type=int,required=True)
    p.add_argument("--mode",default="free-roam",choices=["free-roam","route-assisted"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-seconds",type=float)
    p.add_argument("--record-video",action="store_true")
    args=p.parse_args()
    args.atlas="08b6f2d6-820a-4bfd-875a-6a55d1986a4e"
    args.calibration="segments-df624035-833-bd07601f-708"
    args.scene_yaw="01dbaa74-8e00-4763-a215-9ea37e18b1b2"
    args.cache=Path("artifacts/benchmark_cache/atlas_references")
    args.references=Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json")
    args.references_only=False
    args.loss_error_limit_px=None
    variant="" if args.variant=="baseline" else "".join(sorted(set(args.variant)))
    with installed(variant,args.output/"candidate-source") as metadata:
        args.experiment=metadata
        (args.output/"candidate-source"/"manifest.json").write_text(json.dumps(metadata,indent=2))
        run(args)


if __name__=="__main__":
    main()
