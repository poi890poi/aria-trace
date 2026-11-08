"""Build private template crops from inspected, full-resolution setup captures."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("artifacts/airtest-benchmark"))
    args = parser.parse_args()
    root = args.directory
    out = root / "assets"
    out.mkdir(parents=True, exist_ok=True)
    # Crop rectangles and broader search rectangles in normalized landscape pixels.
    specs = {
        "entry": ("entry-setup2.png", (.441, .431, .560, .496), (.38, .38, .62, .58), .75),
        "door": ("preflight-progress.png", (.463, .918, .535, .946), (.38, .86, .62, .98), .8),
        "uid": ("awake.png", (.857, .976, .880, .999), (.83, .95, .99, 1), .8),
        "level": ("awake.png", (.397, .909, .414, .928), (.36, .87, .45, .97), .8),
        "map": ("map-setup.png", (.843, .932, .890, .958), (.80, .87, .95, .98), .8),
        "waypoint_title": ("selected-setup.png", (.740, .024, .799, .057), (.69, 0, .94, .15), .8),
        "teleport": ("selected-setup.png", (.827, .897, .859, .933), (.72, .84, .95, .97), .8),
    }
    manifest = {"resolution": [2400, 1080], "waypoint": [0.49887453105460605, 0.4978683966635774],
                "templates": {}}
    for name, (source, box, region, threshold) in specs.items():
        file = root / source
        image = cv2.imread(str(file))
        if image is None or image.shape[:2] != (1080, 2400):
            raise ValueError("Expected 2400x1080 capture: " + str(file))
        l, t, r, b = [round(v*s) for v, s in zip(box, (2400, 1080, 2400, 1080))]
        target = out / (name + ".png")
        cv2.imwrite(str(target), image[t:b, l:r])
        manifest["templates"][name] = dict(file=target.name, region=region, threshold=threshold,
                                           source=source, crop=box,
                                           source_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    intent = {"open": "com.miHoYo.GenshinImpact", "touch": [
        {"element": "minimap", "position": [.085, .145]},
        {"element": "unlocked_waypoint", "position": manifest["waypoint"]},
        {"element": "teleport"}]}
    (out / "teleport.json").write_text(json.dumps(intent, indent=2))
    print(out.resolve())


if __name__ == "__main__":
    main()
