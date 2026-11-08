"""Offline comparison against a frozen source copy; no device access or input."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path('.'))
    parser.add_argument('--package', choices=('swiftest', 'androidtestkit'), default='swiftest',
                        help='Use androidtestkit only for a frozen pre-rename source baseline.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--real-ocr', action='store_true')
    parser.add_argument('--disable-cache', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import cv2
    import numpy as np
    package = importlib.import_module(args.package)
    source = (args.source_root / args.package).resolve()
    if Path(package.__file__).resolve().parent != source:
        parser.error('Imported package does not belong to --source-root.')
    Session, Template, Text = package.Session, package.Template, package.Text
    Observation = importlib.import_module(args.package + '.session').Observation
    LocalOCR = importlib.import_module(args.package + '.ocr').LocalOCR
    from unittest.mock import patch

    class Capture:
        repeat = False
        def read(self):
            return self.image
        def close(self):
            pass

    def summary(values):
        return dict(n=len(values), median_ms=statistics.median(values),
                    mean_ms=statistics.mean(values), max_ms=max(values),
                    p95_ms=float(np.percentile(values, 95)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tile = rng.integers(20, 235, (24, 32, 3), dtype=np.uint8)
    capture = Capture()
    report = dict(seed=args.seed, package=args.package, cache_disabled=args.disable_cache, numpy=np.__version__, opencv=cv2.__version__,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source.glob('*.py')})
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        asset = Path(temp)/'button.png'
        asset.write_bytes(cv2.imencode('.png', tile)[1].tobytes())
        session = Session(device=SimpleNamespace(close=lambda: None), capture=capture)
        if args.disable_cache:
            session._visual_cache.max_entries = 0
        target = Template(str(asset.resolve()), region=(.25, .25, .75, .75), threshold=.99)
        times, decisions = [], []
        with patch(args.package + '.targets.cv2.matchTemplate', wraps=cv2.matchTemplate) as matcher:
            for index in range(24):
                image = np.zeros((720, 1280, 3), np.uint8)
                image[:30, :30] = rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)
                points = [(400, 250)] if index < 20 else (
                    [(700, 350)] if index in (20, 23) else
                    [(400, 250), (700, 350)] if index == 21 else [])
                for x, y in points:
                    image[y:y+24, x:x+32] = tile
                capture.image = image
                started = time.perf_counter()
                boxes = target.matches(Observation(session))
                times.append((time.perf_counter()-started)*1000)
                actual = sorted((b.x, b.y, b.w, b.h) for b in boxes)
                expected = sorted((x, y, 32, 24) for x, y in points)
                assert actual == expected, (index, actual, expected)
                decisions.append(actual)
            report['template'] = dict(summary(times), matching_calls=matcher.call_count,
                                      per_frame_ms=times, expected_decisions_passed=len(decisions))
        if args.real_ocr:
            session.ocr = LocalOCR(threads=2, backend='cpu')
            target = Text('READY 42', region=(.25, .25, .75, .75))
            times, shapes = [], []
            real = session.recognize
            def recognize(image):
                shapes.append(list(image.shape))
                return real(image)
            session.recognize = recognize
            for index in range(4):
                image = np.full((720, 1280, 3), 255, np.uint8)
                image[:30, :30] = rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)
                cv2.putText(image, 'READY 42' if index < 3 else 'WAIT 42', (400, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
                capture.image = image
                started = time.perf_counter()
                found = target.matches(Observation(session))
                times.append((time.perf_counter()-started)*1000)
                assert bool(found) == (index < 3), (index, found)
            report['ocr'] = dict(summary(times), input_shapes=shapes, per_frame_ms=times,
                                 first_call_includes_model_initialization=True,
                                 expected_decisions_passed=4)
        report['metrics'] = session.report()
        session.close()
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('seed','template','ocr')}))


if __name__ == '__main__':
    main()
