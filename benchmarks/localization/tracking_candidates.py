"""Explicit runtime-method experiments; baseline production source stays intact.

Each candidate compiles a reviewed method edit and installs it on the actual
production class before Workbench construction. No estimator outputs or future
references are injected. Combined variants apply the same edits cumulatively.
"""

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker
from aria_trace.services.localization.route.tracker import RouteVisualTracker
from benchmarks.localization.run_workbench_replay import run


CANDIDATES = {
    "A": "Pass observation-time position uncertainty to transition authority.",
    "B": "Use reverse-direction transition endpoints as current-image search proposals.",
    "C": "Use the latest consensus observation pose instead of averaging poses across motion.",
    "D": "Preserve global recovery consensus across accepted local-motion updates while recovery is latched.",
    "F": "Remove the automatic hold on transition arming; keep current-layer tracking until visual layer confirmation.",
    "G": "Consume cursor after XY work, without waiting on its own; with E, defer the bounded wait too.",
    "I": "Remove relative-motion accumulation in free-roam; use current-image atlas matching with no route states or proposals.",
    "J": "Use the existing wider recovery radius only for the first image refinement after a global seed.",
    "H": "Remove full-scene KLT and camera-derived minimap rotation trials from free-roam tracking.",
    "E": "Submit then consume same-frame cursor result within the remaining 33.3 ms source deadline; retain asynchronous fallback.",
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Candidate source boundary changed: " + before[:100])
    return source.replace(before, after, 1)


def candidate_sources(letters):
    unknown = set(letters)-set(CANDIDATES)
    if unknown:
        raise ValueError("Unknown candidates: " + str(unknown))
    methods = {}
    def get(cls, name):
        return methods.get((cls, name), textwrap.dedent(inspect.getsource(getattr(cls, name))))
    def put(cls, name, source):
        methods[(cls, name)] = source
    if "A" in letters:
        source = get(TwoRateRealtimeTracker, "_consume_representation_observation")
        source = replace_once(source, '            canonical_xy=(\n',
                              '            position_uncertainty_px=getattr(self, "_representation_position_sigma", 0.0),\n            canonical_xy=(\n')
        put(TwoRateRealtimeTracker, "_consume_representation_observation", source)
        source = get(TwoRateRealtimeTracker, "update")
        source = replace_once(source, '        self._representation_future = self._representation_executor.submit(',
                              '        self._representation_position_sigma = float(state.position_sigma_m)\n        self._representation_future = self._representation_executor.submit(')
        put(TwoRateRealtimeTracker, "update", source)
    if "B" in letters:
        source = get(RouteVisualTracker, "arm_trained_transition")
        source = replace_once(source, '    if not matches:\n        return None\n', '''    if not matches:
        for item in self.package.transitions:
            if (str(item.get("source_mode_id")) == target_mode_id
                    and str(item.get("target_mode_id")) == source_mode_id):
                reverse = dict(item)
                reverse.update(
                    source_mode_id=source_mode_id, target_mode_id=target_mode_id,
                    last_source_state_index=item["first_target_state_index"],
                    last_source_canonical_xy=item.get("first_target_canonical_xy"),
                    first_target_state_index=item["last_source_state_index"],
                    first_target_canonical_xy=item.get("last_source_canonical_xy"))
                matches.append(reverse)
    if not matches:
        return None
''')
        put(RouteVisualTracker, "arm_trained_transition", source)
    if "C" in letters:
        source = get(TwoRateRealtimeTracker, "_mean_fix")
        source = replace_once(source, 'float(np.mean([item.x for item in rows]))', 'float(rows[-1].x)')
        source = replace_once(source, 'float(np.mean([item.y for item in rows]))', 'float(rows[-1].y)')
        source = replace_once(source, '_circular_mean_deg([item.yaw_deg for item in rows])', 'float(rows[-1].yaw_deg)')
        put(TwoRateRealtimeTracker, "_mean_fix", source)
    if "D" in letters:
        source = get(TwoRateRealtimeTracker, "update")
        source = replace_once(source, '        self._local_rejections = 0\n        self._recovery_hypotheses = []\n    elif (',
                              '        self._local_rejections = 0\n        if not self._recovery_request_active:\n            self._recovery_hypotheses = []\n    elif (')
        put(TwoRateRealtimeTracker, "update", source)
    if "F" in letters:
        source = get(RouteVisualTracker, "track")
        source = replace_once(source, '        return self._held_transition_result()\n',
                              '        pending_transition = None  # arming is a proposal, not a reason to freeze XY\n')
        put(RouteVisualTracker, "track", source)
    if "H" in letters:
        source = get(TwoRateRealtimeTracker, "update")
        source = replace_once(source, '    if self.route_visual_tracker is not None:\n        # Route-assisted XY',
                              '    if True:  # north-fixed minimap motion ablation\n        # Route-assisted XY')
        source = replace_once(source, '            "bypassed:route-map-correlation",\n',
                              '            "bypassed:route-map-correlation" if self.route_visual_tracker is not None else "bypassed:scene-motion-ablation",\n')
        put(TwoRateRealtimeTracker, "update", source)
    if "E" in letters:
        source = get(TwoRateRealtimeTracker, "update")
        consume_start = source.index('    cursor_pose_fresh = False\n')
        submit_start = source.index('    cursor_due = (\n', consume_start)
        cursor_end = source.index('    global_fix = None\n', submit_start)
        consume = source[consume_start:submit_start]
        submit = source[submit_start:cursor_end]
        wait = '''    if self._cursor_future is not None and not self._cursor_future.done():
        remaining_s = max(0.0, (timestamp_ns + 1e9/30 - time.perf_counter_ns())/1e9)
        if remaining_s:
            try:
                self._cursor_future.result(timeout=remaining_s)
            except Exception:
                pass  # timeout stays pending; completed errors use existing handling
'''
        source = source[:consume_start] + submit + wait + consume + source[cursor_end:]
        put(TwoRateRealtimeTracker, "update", source)
    if "I" in letters:
        source = get(TwoRateRealtimeTracker, "__init__")
        source += """
    if self.route_visual_tracker is None:
        from types import SimpleNamespace
        from aria_trace.services.localization.route.tracker import RouteVisualTracker
        empty_reference = SimpleNamespace(
            manifest={"motion_envelope": {}}, states=[], transitions=[],
            candidates=lambda *args, **kwargs: [])
        self.route_visual_tracker = RouteVisualTracker(
            empty_reference, self.localizer, score_min=0.0,
            local_radius_px=12.0, recovery_radius_px=55.0)
"""
        put(TwoRateRealtimeTracker, "__init__", source)
    if "J" in letters:
        source = get(RouteVisualTracker, "track")
        source = replace_once(source, '            self.local_radius_px,\n            "continuous-local",',
                              '            self.recovery_radius_px if self.previous_time_ns is None else self.local_radius_px,\n            "continuous-local",')
        put(RouteVisualTracker, "track", source)
    if "G" in letters:
        source = get(TwoRateRealtimeTracker, "update")
        consume_start = source.index('    cursor_pose_fresh = False\n')
        if "E" in letters:
            consume_start = source.index('    if self._cursor_future is not None and not self._cursor_future.done():\n')
            consume_end = source.index('    global_fix = None\n', consume_start)
        else:
            consume_end = source.index('    cursor_due = (\n', consume_start)
        consume = source[consume_start:consume_end]
        source = source[:consume_start] + source[consume_end:]
        source = replace_once(source, '    self.sequence += 1\n', consume + '    self.sequence += 1\n')
        put(TwoRateRealtimeTracker, "update", source)
    return methods


@contextmanager
def installed_candidates(letters, output=None):
    methods = candidate_sources(letters)
    originals = {}
    metadata = {"variant": letters or "baseline", "changes": {k:CANDIDATES[k] for k in letters}, "methods": []}
    try:
        for (cls, name), source in methods.items():
            originals[(cls, name)] = inspect.getattr_static(cls, name)
            namespace = vars(inspect.getmodule(cls))
            filename = f"candidate-{letters}-{cls.__name__}-{name}.py"
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                (output/filename).write_text(source, encoding="utf-8")
            sentinel = object()
            prior_global = namespace.get(name, sentinel)
            try:
                exec(compile(source, filename, "exec"), namespace)
                setattr(cls, name, namespace[name])
            finally:
                if prior_global is sentinel:
                    namespace.pop(name, None)
                else:
                    namespace[name] = prior_global
            metadata["methods"].append({"class":cls.__name__, "method":name,
                                        "sha256":hashlib.sha256(source.encode()).hexdigest(), "source_file":filename})
        yield metadata
    finally:
        for (cls, name), original in originals.items():
            setattr(cls, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--runs", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=["free-roam", "route-assisted"], default="route-assisted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json"))
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    args.atlas = "08b6f2d6-820a-4bfd-875a-6a55d1986a4e"
    args.calibration = "segments-df624035-833-bd07601f-708"
    args.scene_yaw = "01dbaa74-8e00-4763-a215-9ea37e18b1b2"
    args.cache = Path("artifacts/benchmark_cache/atlas_references")
    args.references_only = False
    args.loss_error_limit_px = None
    letters = "" if args.variant == "baseline" else "".join(sorted(set(args.variant)))
    with installed_candidates(letters, args.output/"candidate-source") as metadata:
        args.experiment = metadata
        (args.output/"candidate-source"/"manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output/"candidate-source"/"manifest.json").write_text(json.dumps(metadata, indent=2))
        run(args)


if __name__ == "__main__":
    main()
