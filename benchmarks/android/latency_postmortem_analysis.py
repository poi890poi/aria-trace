"""Offline waste accounting for latency_postmortem_probe; never controls a device."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from examples.genshin.profile import is_loading_screen, ocr_image
    summary, all_rows = [], []
    for manifest in sorted(args.directory.glob("*/diagnostic.json")):
        rows = json.loads(manifest.read_text())
        result = json.loads((manifest.parent / "result.json").read_text())
        timed_start = result["inputs"][-3]["started"] + result["map_to_arrival_seconds"] - result["task_seconds"]
        previous = None
        for row in rows:
            image = cv2.imread(row["path"])
            assert image is not None
            prepared = ocr_image(image)
            began = time.perf_counter()
            row["loading_pixels"] = is_loading_screen(prepared)
            row["loading_gate_ms"] = (time.perf_counter()-began)*1000
            row["timed"] = row["started"] >= timed_start
            row["relative_start"] = row["started"]-timed_start
            row["case"] = manifest.parent.name
            mask = np.zeros(image.shape[:2], np.uint8)
            for line in (row["observation"] or {}).get("lines", []):
                for b in line["words"]:
                    x, y, w, h = (b[k] for k in ("x", "y", "width", "height"))
                    mask[max(0,y):y+h, max(0,x):x+w] = 1
            row["recognized_box_area_fraction"] = float(mask.mean())
            if previous is not None:
                changed = np.any(image != previous, axis=2)
                row["decoded_changed_fraction"] = float(changed.mean())
                tiles = [changed[y:y+64,x:x+64] for y in range(0,image.shape[0],64)
                         for x in range(0,image.shape[1],64)]
                row["decoded_changed_tile_area_fraction"] = sum(t.size for t in tiles if t.any())/changed.size
            previous = image
        timed = [r for r in rows if r["timed"]]
        calls = [r for r in timed if r["ocr_calls"]]
        empty = [r for r in calls if next((s["count"] for s in r["stages"] if s["stage"]=="text_det"), -1)==0]
        changed = [r for r in timed if not r["repeated"] and "decoded_changed_fraction" in r]
        by_stage = {name: sum(s["seconds"] for r in timed for s in r["stages"] if s["stage"]==name)
                    for name in ("text_det", "text_cls", "text_rec")}
        summary.append(dict(case=manifest.parent.name,task_seconds=result["task_seconds"],
            captured=len(timed), states=dict(Counter(r["state"] for r in timed)),
            repeats=sum(r["repeated"] for r in timed), bytes=sum(r["bytes"] for r in timed),
            ocr_calls=len(calls), no_detections=len(empty),
            no_detection_seconds=sum(s["seconds"] for r in empty for s in r["stages"]),
            loading_gate_ocr_calls=sum(r["loading_pixels"] for r in calls),
            loading_gate_ms_median=statistics.median(r["loading_gate_ms"] for r in timed),
            stages=by_stage,
            recognized_area_median=statistics.median(r["recognized_box_area_fraction"] for r in calls),
            changed_pixel_median=statistics.median(r["decoded_changed_fraction"] for r in changed),
            changed_tile_area_median=statistics.median(r["decoded_changed_tile_area_fraction"] for r in changed)))
        all_rows.extend(rows)
        tiles = []
        for row in rows:
            if not row["timed"]:
                continue
            frame = cv2.resize(cv2.imread(row["path"]), (480,216))
            frame = cv2.copyMakeBorder(frame,28,0,0,0,cv2.BORDER_CONSTANT)
            label = "%s %.2fs %s%s" % (Path(row["path"]).stem[-4:],row["relative_start"],row["state"]," repeat" if row["repeated"] else "")
            cv2.putText(frame,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            tiles.append(frame)
        for offset in range(0,len(tiles),24):
            page=tiles[offset:offset+24]
            page += [np.zeros_like(page[0]) for _ in range((-len(page))%3)]
            contact=np.vstack([np.hstack(page[i:i+3]) for i in range(0,len(page),3)])
            assert cv2.imwrite(str(manifest.parent/("contact-%02d.jpg" % (offset//24))),contact)
    (args.directory / "analysis.json").write_text(json.dumps(dict(summary=summary, rows=all_rows),indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
