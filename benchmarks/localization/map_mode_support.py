"""Build experimental atlas-owned observed mode coverage; unseen space stays unknown."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import read_rows, validate_reference_inputs


def build(root, include_run20=False):
    refs = {int(k): Path(v) for k, v in json.loads((root / "references.json").read_text()).items()}
    source = Path("artifacts/workbench/map_atlases/genshin-impact-pc/08b6f2d6-820a-4bfd-875a-6a55d1986a4e")
    target = root / ("map-feature-atlas-expanded" if include_run20 else "map-feature-atlas")
    if target.exists():
        raise RuntimeError("Use a new atlas candidate directory")
    atlas = json.loads((source / "map_atlas.json").read_text())
    width, height = atlas["canonical_size_wh"]
    layers = {l["mode_id"]: l for l in atlas["layers"]}
    # Inherited diagnostic uncertainty, frozen before holdout evaluation.
    envelope = json.loads((root / "unknown-start/run20/report.json").read_text())["tracking_loss"]["calibration"]
    radii = {mode: value["error_limit_px"] for mode, value in envelope["modes"].items()}
    masks = {mode: np.zeros((height, width), np.uint8) for mode in layers}
    samples, crossings, inputs = [], [], []
    for number in ((11, 14, 20) if include_run20 else (11, 14)):
        reference = refs[number]
        validate_reference_inputs(reference)
        marker = json.loads((reference / "cache.json").read_text())
        for item in marker["outputs"]:
            if identity(reference / item["name"])["sha256"] != item["sha256"]:
                raise RuntimeError("Reference output changed")
        inputs.append({"session": number, "reference": identity(reference / "cache.json")})
        rows = read_rows(reference / "route_states.jsonl")
        for row in rows:
            mode = row["mode_id"]
            scale = layers[mode]["map_pixels_per_minimap_pixel"]
            if abs(row["map_scale"] / scale - 1) > .12:
                continue  # Existing scale-consistency band, not tuned on run 20.
            point = tuple(int(round(v)) for v in row["canonical_xy"])
            cv2.circle(masks[mode], point, int(np.ceil(radii[mode])), 255, -1)
            samples.append({"source_session": number, "source_frame_index": row["source_frame_index"],
                            "session_time_ns": row["session_time_ns"], "mode_id": mode,
                            "canonical_xy": row["canonical_xy"], "scale": row["map_scale"]})
        for before, after in zip(rows, rows[1:]):
            if before["mode_id"] != after["mode_id"]:
                crossings.append({"source_session": number, "from": before["mode_id"], "to": after["mode_id"],
                                  "time_bracket_ns": [before["session_time_ns"], after["session_time_ns"]],
                                  "xy_bracket": [before["canonical_xy"], after["canonical_xy"]]})
    shutil.copytree(source, target)
    feature_dir = target / "map_features"
    feature_dir.mkdir()
    for mode, mask in masks.items():
        cv2.imwrite(str(feature_dir / (mode + "_observed_support.png")), mask)
    both = (masks["world"] > 0) & (masks["town"] > 0)
    codes = np.zeros((height, width), np.uint8)
    codes[masks["world"] > 0] = 64; codes[masks["town"] > 0] = 128; codes[both] = 192
    cv2.imwrite(str(feature_dir / "observed_mode_support.png"), codes)
    labels = {0: "unknown", 64: "world", 128: "town", 192: "ambiguous"}
    counts = Counter()
    for row in ([] if include_run20 else read_rows(refs[20] / "route_states.jsonl")):
        mode = row["mode_id"]
        if abs(row["map_scale"] / layers[mode]["map_pixels_per_minimap_pixel"] - 1) > .12:
            continue
        x, y = (int(round(v)) for v in row["canonical_xy"])
        predicted = labels[int(codes[y, x])] if 0 <= x < width and 0 <= y < height else "unknown"
        counts[(mode, predicted)] += 1
    feature = {"schema_version": 1, "owner": "map-atlas", "role": "benchmark-only-observed-location-support",
               "source_atlas": identity(source / "map_atlas.json"), "training_references": inputs,
               "evaluation_only_session": None if include_run20 else 20, "support_radius_px": radii,
               "expansion_policy": "Run 20 added after frozen holdout scoring; expanded map has no new holdout" if include_run20 else "Run 20 is evaluation only",
               "support_radius_role": "inferred-position distinguishability envelope, not region extent",
               "stable_scale_relative_band": .12,
               "pixel_codes": {str(k): v for k, v in labels.items()},
               "observed_samples": samples, "directed_crossing_brackets": crossings,
               "covered_pixels": {mode: int(np.count_nonzero(mask)) for mode, mask in masks.items()},
               "overlapping_support_pixels": int(np.count_nonzero(both)),
               "holdout_mode_lookup": [{"reference_mode": k[0], "support_label": k[1], "samples": v} for k, v in sorted(counts.items())],
               "limitations": ["Sparse support is not complete town segmentation.",
                               "Mini-map footprint is not a town membership label.",
                               "No route order, lap progress, or target pose is stored as transition authority.",
                               "Unknown support must not veto current-image mode evidence.",
                               "Position and scale labels share the original atlas localizer; not external truth.",
                               "New artwork and runtime transition policy are unchanged."]}
    (feature_dir / "observed_modes.json").write_text(json.dumps(feature, indent=2))
    atlas["atlas_id"] += "-observed-mode-support" + ("-expanded" if include_run20 else "")
    atlas["status"] = "benchmark-candidate"
    atlas["experimental_map_features"] = {"observed_modes": "map_features/observed_modes.json"}
    (target / "map_atlas.json").write_text(json.dumps(atlas, indent=2))
    picture = cv2.imread(str(target / "canonical_mosaic.png"))
    tint = picture.copy()
    tint[codes == 64] = (255, 160, 30); tint[codes == 128] = (40, 230, 70); tint[codes == 192] = (30, 60, 255)
    picture = cv2.addWeighted(picture, .35, tint, .65, 0)
    xy = np.array([s["canonical_xy"] for s in samples]); low = np.floor(xy.min(0)-35).astype(int); high = np.ceil(xy.max(0)+35).astype(int)
    picture = cv2.resize(picture[low[1]:high[1], low[0]:high[0]], None, fx=3, fy=3)
    picture = cv2.copyMakeBorder(picture, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))
    cv2.putText(picture, "Town green | World blue | Overlap red | Untinted unknown", (12,26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(feature_dir / "observed_mode_overlay.png"), picture)
    print(json.dumps({k: v for k, v in feature.items() if k not in ("observed_samples", "source_atlas", "training_references")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("root", type=Path)
    parser.add_argument("--include-run20", action="store_true")
    args = parser.parse_args()
    build(args.root, args.include_run20)
