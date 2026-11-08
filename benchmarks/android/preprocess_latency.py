"""Compare exact 8-bit percentile evaluation on immutable saved frames."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from steady_latency import summary
from examples.genshin.profile import brightness_percentile as histogram_level


def prepare(image, level):
    return cv2.convertScaleAbs(image, alpha=min(16., 230. / level)) if 3 < level < 65 else image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, inputs = [], []
    for case in json.loads(args.manifest.read_text()):
        path = Path(case["image"])
        original = cv2.imread(str(path))
        assert original is not None, path
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for dim in (False, True):
            image = original // 8 if dim else original
            for i in range(args.rounds + 1):
                outputs = []
                for method in (("percentile", "histogram") if i % 2 else ("histogram", "percentile")):
                    start = time.perf_counter()
                    level = float(np.percentile(image, 99.5)) if method == "percentile" else histogram_level(image)
                    output = prepare(image, level)
                    elapsed = (time.perf_counter()-start)*1000
                    outputs.append(output)
                    if i:
                        rows.append(dict(name=case["name"], dim=dim, method=method, ms=elapsed, level=level))
                np.testing.assert_array_equal(outputs[0], outputs[1])
    result = {method: summary([r["ms"] for r in rows if r["method"] == method])
              for method in ("percentile", "histogram")}
    (args.output/"result.json").write_text(json.dumps(dict(inputs=inputs, rows=rows, summary=result,
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), ROOT/"examples/genshin/profile.py"]}), indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
