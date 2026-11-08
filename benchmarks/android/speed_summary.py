"""Aggregate speed trials, retaining every success/failure and resource cost."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from summarize import metrics


def distribution(values):
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values),
                p95=statistics.quantiles(values, n=20, method="inclusive")[-1]
                if len(values) > 1 else values[0])


def report(root):
    rows = [metrics(root, r) for r in json.loads((root/"results.json").read_text())]
    metadata = json.loads((root/"metadata.json").read_text())
    repository = Path(__file__).resolve().parents[2]
    unchanged = all(hashlib.sha256((repository/p).read_bytes()).hexdigest() == value
                    for p, value in metadata["hashes"].items())
    result = dict(source_hashes_unchanged=unchanged, trials=[], groups={})
    for row in rows:
        telemetry = json.loads((root/f'{row["index"]:02d}-testkit'/"temperatures.json").read_text())
        result["trials"].append(dict(case=row["case"], success=row["success"],
            process_seconds=row["process_seconds"], map_seconds=row["map_seconds"],
            charging_statuses=sorted(set(s.get("status", "missing") for s in telemetry)),
            thermal_statuses=sorted(set(s.get("thermal_status", -1) for s in telemetry)),
            sampler_errors=sum("error" in s for s in telemetry),
            rest_gate_met=row["rest_gate_met"], temperature=row["temperature"]))
    for case in dict.fromkeys(r["case"] for r in rows):
        selected = [r for r in rows if r["case"] == case]
        inputs = [i for r in selected for i in r["task"]["inputs"] if i["sent"]]
        measures = {
            "process_seconds": [r["process_seconds"] for r in selected],
            "map_seconds": [r["map_seconds"] for r in selected],
            "ocr_seconds": [r["task"]["ocr_seconds"] for r in selected],
            "capture_seconds": [r["task"]["capture_seconds"] for r in selected],
            "captures": [r["task"]["metrics"]["captures"] for r in selected],
            "capture_bytes": [r["task"]["metrics"]["capture_bytes"] for r in selected],
            "ready_frame_to_input_seconds": [i["ready_frame_to_input_seconds"] for i in inputs],
            "input_seconds": [i["seconds"] for i in inputs],
        }
        result["groups"][case] = {key: distribution(values) for key, values in measures.items()}
        result["groups"][case]["passed"] = sum(r["success"] for r in selected)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.directory), indent=2))
