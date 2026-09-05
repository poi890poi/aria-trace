"""CPU feature-proposal experiments; no production dependency or reference input."""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
import time

import cv2
import numpy as np

from aria_trace.services.tracking import runtime
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import distribution, read_rows, run
from benchmarks.localization.tracking_candidates import replace_once

BASE = "f607fa6"
UPSTREAM = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
ATLAS = "08b6f2d6-820a-4bfd-875a-6a55d1986a4e"
CALIBRATION = "segments-df624035-833-bd07601f-708"


def frozen_method(name):
    source = subprocess.check_output(["git", "show", BASE + ":aria_trace/services/tracking/runtime.py"], text=True, encoding="utf-8")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "GlobalMapLocalizer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent("".join(source.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]))


class XFeatAdapter:
    def __init__(self, model, top_k=8000, feature_scale=1):
        self.model, self.top_k, self.feature_scale = model, top_k, feature_scale
        self.last_metrics = {}

    def detectAndCompute(self, gray, mask):
        started = time.perf_counter()
        model_input = cv2.resize(gray, None, fx=self.feature_scale, fy=self.feature_scale,
                                interpolation=cv2.INTER_LINEAR) if self.feature_scale != 1 else gray
        out = self.model.detectAndCompute(model_input, top_k=self.top_k)[0]
        xy = out["keypoints"].cpu().numpy() / self.feature_scale
        descriptors = out["descriptors"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        # Test validity before indexing: no clamping can turn an invalid point
        # into an apparently covered one.
        ij = np.rint(xy).astype(np.int64)
        valid = (ij[:, 0] >= 0) & (ij[:, 0] < gray.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < gray.shape[0])
        if mask is not None:
            selected = np.flatnonzero(valid)
            valid[selected] &= mask[ij[selected, 1], ij[selected, 0]] > 0
        points = [cv2.KeyPoint(float(x), float(y), 1.0, response=float(s)) for (x, y), s in zip(xy[valid], scores[valid])]
        self.last_metrics = {"extraction_ms": (time.perf_counter()-started)*1000,
                             "features_before_mask": len(xy), "features_after_mask": len(points)}
        return points, np.ascontiguousarray(descriptors[valid], dtype=np.float32) if points else None


def mnn_matches(query, target):
    # Upstream sparse XFeat uses normalized descriptors and mutual nearest
    # neighbors, without a similarity threshold by default. L2 cross-check
    # has the same ranking for these unit descriptors, using OpenCV CPU.
    return list(cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(query, target))


def mask_feature_input(gray, mask, fill="none", region="both"):
    """Remove excluded query pixels before extraction; never change pose inputs."""
    if fill == "none":
        return gray
    excluded = mask == 0
    if region != "both":
        _, labels = cv2.connectedComponents(excluded.astype(np.uint8), connectivity=8)
        edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        outside = excluded & np.isin(labels, edge_labels)
        excluded = outside if region == "outside" else excluded & ~outside
    value = int(round(float(np.mean(gray[mask > 0])))) if fill == "mean" and np.any(mask > 0) else 0
    result = gray.copy()
    result[excluded] = value
    return result


@contextmanager
def installed(variant, output, upstream, threads=1, feature_scale=1, input_mask="none", mask_region="both"):
    output.mkdir(parents=True, exist_ok=True)
    (output / "harness.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    model = None
    metadata = {"variant": variant, "feature_scale": feature_scale, "input_mask": input_mask,
                "mask_region": mask_region, "base": BASE, "python": platform.python_version(),
                "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
                "implementation": identity(__file__), "methods": [], "setup": []}
    if variant != "sift":
        import torch
        actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual != UPSTREAM:
            raise ValueError("XFeat source revision differs from protocol")
        if torch.version.cuda is not None:
            raise RuntimeError("Use the isolated CPU-only PyTorch build")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(upstream.resolve()))
        from modules.xfeat import XFeat
        started = time.perf_counter()
        weights = upstream / "weights/xfeat.pt"
        model = XFeat(weights=str(weights.resolve()), top_k=8000)
        if str(model.dev) != "cpu":
            raise RuntimeError("XFeat is not on CPU")
        metadata.update({"torch": torch.__version__, "threads": threads, "device": "cpu",
                         "upstream": actual, "weights": identity(weights),
                         "parameters": sum(p.numel() for p in model.net.parameters()),
                         "model_load_ms": (time.perf_counter()-started)*1000,
                         "upstream_sources": [identity(p) for p in sorted((upstream/"modules").rglob("*.py"))]})
        if variant == "xfeat-lighterglue":
            import kornia
            from modules.lighterglue import LighterGlue
            started = time.perf_counter()
            model.lighterglue = LighterGlue().eval()
            # Verify the upstream permissive loader actually found every tensor.
            weights = torch.load(upstream/"weights/xfeat-lighterglue.pt", map_location="cpu", weights_only=True)
            for i in range(model.lighterglue.net.conf.n_layers):
                weights = {k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn").replace(
                    f"cross_attn.{i}", f"transformers.{i}.cross_attn").replace("matcher.", ""): v for k,v in weights.items()}
            expected = model.lighterglue.net.state_dict()
            extra = set(weights)-set(expected)
            missing = set(expected)-set(weights)
            # The upstream training checkpoint also contains the extractor.
            # Kornia computes this nonlearned threshold buffer in __init__.
            if any(not k.startswith("extractor.") for k in extra) or missing != {"confidence_thresholds"}:
                raise RuntimeError(f"Unexpected LighterGlue checkpoint mismatch: {missing}, {extra}")
            checked = {k: weights[k] if k in weights else expected[k] for k in expected}
            model.lighterglue.net.load_state_dict(checked, strict=True)
            metadata.update({"kornia": kornia.__version__, "matcher_weights": identity(upstream/"weights/xfeat-lighterglue.pt"),
                             "matcher_load_ms": (time.perf_counter()-started)*1000,
                             "matcher_parameters": sum(p.numel() for p in model.lighterglue.parameters())})
    sources = {name: frozen_method(name) for name in ("__init__", "localize")}
    if input_mask != "none":
        sources["localize"] = replace_once(sources["localize"],
            "points, descriptors = self.sift.detectAndCompute(observation_gray, mask)",
            "points, descriptors = self.sift.detectAndCompute(_xfeat_mask_input(observation_gray, mask), mask)")
    if model is not None:
        sources["__init__"] = replace_once(sources["__init__"],
            'self.sift = cv2.SIFT_create(\n        nfeatures=8000, contrastThreshold=0.005, edgeThreshold=15\n    )',
            'self.sift = _xfeat_factory()')
    if variant in ("xfeat-mnn", "xfeat-lighterglue"):
        start = sources["localize"].index("    pairs = cv2.BFMatcher")
        end = sources["localize"].index("    ratio_count = len(matches)")
        replacement = "    matches = _xfeat_mnn(descriptors, self.map_descriptors)\n" if variant == "xfeat-mnn" else (
            "    matches = _xfeat_glue(points, descriptors, observation.shape, self.map_points, self.map_descriptors, self.mosaic.shape)\n")
        sources["localize"] = sources["localize"][:start] + replacement + sources["localize"][end:]
    originals = {name: getattr(runtime.GlobalMapLocalizer, name) for name in sources}
    calls = (output / "global_calls.jsonl").open("x", encoding="utf-8")
    def glue(points, descriptors, shape, map_points, map_descriptors, map_shape):
        def data(kps, desc, hw):
            return {"keypoints": torch.tensor([p.pt for p in kps], dtype=torch.float32),
                    "descriptors": torch.from_numpy(desc), "image_size": (hw[1], hw[0])}
        _, _, indices = model.match_lighterglue(data(points, descriptors, shape), data(map_points, map_descriptors, map_shape))
        return [cv2.DMatch(int(a), int(b), 0.0) for a,b in indices]
    namespace = dict(vars(runtime), _xfeat_factory=lambda: XFeatAdapter(model, feature_scale=feature_scale),
                     _xfeat_mnn=mnn_matches, _xfeat_glue=glue,
                     _xfeat_mask_input=lambda gray, mask: mask_feature_input(gray, mask, input_mask, mask_region))
    try:
        for name, source in sources.items():
            filename = output / (name + ".py")
            filename.write_text(source, encoding="utf-8")
            exec(compile(source, str(filename), "exec"), namespace)
            setattr(runtime.GlobalMapLocalizer, name, namespace[name])
            metadata["methods"].append(identity(filename))
        init, localize = runtime.GlobalMapLocalizer.__init__, runtime.GlobalMapLocalizer.localize

        def measured_init(self, *args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            init(self, *args, **kwargs)
            self._xfeat_layer_shape = list(self.mosaic.shape[:2])
            metadata["setup"].append({"map_shape_hw": self._xfeat_layer_shape,
                "features": len(self.map_points), "wall_ms": (time.perf_counter()-wall)*1000,
                "process_cpu_ms": (time.process_time()-cpu)*1000})

        def measured_localize(self, observation, mask, *args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            fix = localize(self, observation, mask, *args, **kwargs)
            row = {"map_shape_hw": self._xfeat_layer_shape,
                   "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                   "wall_ms": (time.perf_counter()-wall)*1000,
                   "process_cpu_ms": (time.process_time()-cpu)*1000,
                   "valid": fix.valid, "xy": [fix.x, fix.y], "reasons": fix.rejection_reasons,
                   "matches": fix.ratio_match_count, "inliers": fix.inlier_count,
                   "inlier_ratio": fix.inlier_ratio,
                   "features": getattr(self.sift, "last_metrics", None)}
            calls.write(json.dumps(row)+"\n")
            calls.flush()
            return fix

        runtime.GlobalMapLocalizer.__init__ = measured_init
        runtime.GlobalMapLocalizer.localize = measured_localize
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
        yield metadata
    finally:
        for name, method in originals.items():
            setattr(runtime.GlobalMapLocalizer, name, method)
        calls.close()
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2))


def probe(args):
    from rig_runtime.adapters.filesystem.profiles import ProfileCatalog
    from rig_runtime.adapters.filesystem.session import SessionReader
    from replay.session_tools import sample_frames, decode_frames
    artifacts = Path("artifacts/workbench")
    calibration = artifacts / "minimap_calibrations/genshin-impact-pc" / CALIBRATION / "calibration.json"
    config = ProfileCatalog().game("genshin-impact-pc")["minimap_calibration"]
    extractor = runtime.MinimapExtractor(config["crop_xywh"], json.loads(calibration.read_text()))
    localizer = LayeredGlobalLocalizer(artifacts / "map_atlases/genshin-impact-pc" / ATLAS)
    manifest = {"calibration": identity(calibration), "inputs": [], "summaries": []}
    try:
        for number in args.runs:
            session = Path("sessions/workbench/recordings-genshin-impact-pc") / f"run_{number:02d}"
            reader = SessionReader(session)
            frames = reader.frames_by_stream["main"]
            end = min(frames[-1]["session_time_ns"], int(args.max_seconds*1e9)) if args.max_seconds else frames[-1]["session_time_ns"]
            selected = sample_frames(frames, frames[0]["session_time_ns"], end, args.rate)
            images = decode_frames(reader, "main", selected)
            manifest["inputs"].extend(identity(session/p) for p in ("frames.jsonl", "video_main.mkv"))
            rows = []
            with (args.output / f"probe{number:02d}.jsonl").open("x") as file:
                for frame, image in zip(selected, images):
                    observation, mask = extractor.extract(image)
                    cv2.setRNGSeed(0)
                    wall, cpu = time.perf_counter(), time.process_time()
                    # Every frame is an independent unrestricted cold query.
                    fix = localizer.localize(observation, mask)
                    row = {"session": number, "frame_index": frame["frame_index"], "session_time_ns": frame["session_time_ns"],
                        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                        "wall_ms": (time.perf_counter()-wall)*1000, "process_cpu_ms": (time.process_time()-cpu)*1000,
                        "valid": fix.valid, "xy": [fix.x, fix.y], "reasons": fix.rejection_reasons,
                        "mode": localizer.last_selected_mode_id, "matches": fix.ratio_match_count,
                        "inliers": fix.inlier_count, "inlier_ratio": fix.inlier_ratio,
                        "score": fix.score, "margin": fix.margin}
                    file.write(json.dumps(row)+"\n")
                    file.flush()
                    rows.append(row)
            summary = {"session": number, "samples": len(rows), "accepted": sum(r["valid"] for r in rows),
                "first_accepted_s": next((r["session_time_ns"]/1e9 for r in rows if r["valid"]), None),
                "wall_ms": distribution([r["wall_ms"] for r in rows]),
                "process_cpu_ms": distribution([r["process_cpu_ms"] for r in rows])}
            print("PROBE", json.dumps(summary), flush=True)
            manifest["summaries"].append(summary)
    finally:
        localizer.close()
        (args.output/"probe.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["sift", "xfeat-ratio", "xfeat-mnn", "xfeat-lighterglue"], required=True)
    parser.add_argument("--action", choices=["probe", "replay"], required=True)
    parser.add_argument("--runs", nargs="+", type=int, required=True)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--feature-scale", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input-mask", choices=["none", "zero", "mean"], default="none")
    parser.add_argument("--mask-region", choices=["both", "cursor", "outside"], default="both")
    parser.add_argument("--mode", choices=["free-roam", "route-assisted"], default="free-roam")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".tools/xfeat-upstream"))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Use a new output directory")
    args.atlas, args.calibration = ATLAS, CALIBRATION
    args.scene_yaw = "01dbaa74-8e00-4763-a215-9ea37e18b1b2"
    args.cache = Path("artifacts/benchmark_cache/atlas_references")
    args.references = Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json")
    args.references_only = args.record_video = False
    args.loss_error_limit_px = None
    with installed(args.variant, args.output/"candidate-source", args.upstream, args.threads,
                   args.feature_scale, args.input_mask, args.mask_region) as metadata:
        args.experiment = metadata
        if args.action == "probe":
            probe(args)
        else:
            run(args)


if __name__ == "__main__":
    main()
