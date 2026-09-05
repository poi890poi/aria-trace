"""Content-addressed slow atlas references, for post-run scoring only."""

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from aria_trace.workflows.route import compile_route_session


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def ensure_reference(root, session, atlas, calibration, minimap_config, cache_root, rate=2.0):
    """Invalidate on source, video, timestamps, atlas pixels, calibration or settings.

    Rejected samples stay in the package manifest. A cache hit also verifies all
    output hashes; interrupted or damaged entries are never silently reused.
    These inferred references are not external ground truth.
    """
    root, session, atlas, calibration, cache_root = map(Path, (root, session, atlas, calibration, cache_root))
    inputs = [session / name for name in ("manifest.json", "frames.jsonl", "video_main.mkv")]
    inputs += [calibration / "calibration.json"]
    inputs += sorted(p for p in atlas.rglob("*") if p.is_file())
    source = [Path(__file__).resolve()]
    for directory in ("aria_trace/services", "aria_trace/workflows", "rig_runtime/services/calibration", "rig_runtime/adapters/filesystem", "replay"):
        source += sorted((root / directory).rglob("*.py"))
    protocol = {
        "schema": 1, "role": "slow-inferred-atlas-reference-not-external-truth",
        "inputs": [identity(p) for p in inputs], "source": [identity(p) for p in source],
        "minimap_config": minimap_config, "reference_rate_hz": rate,
        "max_step_px": 80.0, "python": platform.python_version(),
        "opencv": cv2.__version__, "numpy": np.__version__,
    }
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    entry = cache_root / key
    marker = entry / "cache.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved["protocol"] != protocol:
            raise RuntimeError("Reference cache identity mismatch")
        for item in saved["outputs"]:
            if identity(entry / item["name"])["sha256"] != item["sha256"]:
                raise RuntimeError("Reference cache damaged: " + item["name"])
        return entry, True
    if entry.exists():
        raise RuntimeError("Incomplete reference cache; inspect before rebuilding: " + str(entry))
    started = time.perf_counter()
    manifest = compile_route_session(
        session, entry, stream_id="main", route_id=session.name,
        atlas_path=atlas, minimap_config=minimap_config,
        minimap_calibration=json.loads((calibration / "calibration.json").read_text()),
        reference_rate_hz=rate,
        progress=lambda message: print(session.name, message, flush=True),
    )
    manifest["reference_role"] = protocol["role"]
    (entry / "manifest.json").write_text(json.dumps(manifest, indent=2))
    outputs = [{"name": p.name, "sha256": identity(p)["sha256"]} for p in sorted(entry.iterdir()) if p.is_file()]
    marker.write_text(json.dumps({"key": key, "protocol": protocol, "outputs": outputs, "build_seconds": time.perf_counter() - started}, indent=2))
    return entry, False
