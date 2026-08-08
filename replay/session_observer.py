"""Run the incremental observer on a recorded session and retain its timeline."""

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from acquisition.session import SessionReader

from .incremental import IncrementalReplayObserver
from .package import ReplayPackage
from .session_tools import make_stages, route_annotations, route_bounds, sample_frames, stage_for_time


def _selected_images(reader, stream_id, frames):
    requested = {int(frame["frame_index"]): frame for frame in frames}
    capture = cv2.VideoCapture(str(reader.video_path(stream_id)))
    if not capture.isOpened():
        raise RuntimeError("Could not open video: {}".format(reader.video_path(stream_id)))
    emitted = 0
    try:
        frame_index = 0
        maximum = max(requested) if requested else -1
        while frame_index <= maximum:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index in requested:
                emitted += 1
                yield requested[frame_index], image
            frame_index += 1
    finally:
        capture.release()
    if emitted != len(frames):
        raise RuntimeError("Decoded {} of {} selected frames".format(emitted, len(frames)))


def _write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_outputs(output_path, records, summary):
    with (output_path / "timeline.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    fields = list(records[0].keys())
    with (output_path / "timeline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    _write_json_atomic(output_path / "summary.json", summary)
    _write_timeline_html(output_path / "timeline.html", records, summary)


def _write_timeline_html(path, records, summary):
    payload = json.dumps(records, separators=(",", ":")).replace("<", "\\u003c")
    summary_payload = json.dumps(summary, indent=2).replace("<", "\\u003c")
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>AriaTrace Incremental Timeline</title>
<style>body{font:14px system-ui;margin:24px;background:#111;color:#ddd}canvas{background:#191f27;width:100%;height:320px}table{border-collapse:collapse;width:100%}th,td{padding:5px;border-bottom:1px solid #333;text-align:right}th:nth-child(2),td:nth-child(2){text-align:left}.bad{color:#ff6b6b}.good{color:#6ee7a8}pre{white-space:pre-wrap}</style></head>
<body><h1>Incremental replay timeline</h1><pre id="summary"></pre><canvas id="plot" width="1200" height="320"></canvas>
<p><span class="good">green: progress</span> · blue: confidence · <span class="bad">red: rejected observation</span></p>
<table><thead><tr><th>#</th><th>stage</th><th>progress</th><th>confidence</th><th>distance</th><th>latency ms</th><th>accepted</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=PAYLOAD;const summary=SUMMARY;document.getElementById('summary').textContent=JSON.stringify(summary,null,2);
const c=document.getElementById('plot'),x=c.getContext('2d'),w=c.width,h=c.height,p=24,n=Math.max(1,data.length-1);x.strokeStyle='#333';for(let i=0;i<=4;i++){let y=p+(h-2*p)*i/4;x.beginPath();x.moveTo(p,y);x.lineTo(w-p,y);x.stroke()}
function line(key,color){x.strokeStyle=color;x.lineWidth=2;x.beginPath();data.forEach((r,i)=>{let px=p+(w-2*p)*i/n,py=h-p-(h-2*p)*r[key];i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}line('progress','#6ee7a8');line('confidence','#55aaff');
data.forEach((r,i)=>{if(!r.accepted){x.fillStyle='#ff5555';x.beginPath();x.arc(p+(w-2*p)*i/n,h-p-(h-2*p)*r.progress,4,0,7);x.fill()}});
const body=document.getElementById('rows');data.forEach(r=>{let tr=document.createElement('tr');tr.className=r.accepted?'good':'bad';[r.observation_index,r.stage_label,r.progress.toFixed(3),r.confidence.toFixed(3),r.visual_distance.toFixed(3),r.processing_latency_ms.toFixed(3),r.accepted].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});body.appendChild(tr)});</script></body></html>"""
    path.write_text(
        html.replace("PAYLOAD", payload).replace("SUMMARY", summary_payload),
        encoding="utf-8",
    )


def observe_session(
    package_path: Path,
    query_session_path: Path,
    output_path: Path,
    stream_id: Optional[str] = None,
    route_id: Optional[str] = None,
    query_rate_hz: float = 5.0,
    max_advance: int = 4,
    distance_threshold: float = 0.45,
    min_margin: float = 0.0,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError("Observer output directory is not empty: {}".format(output_path))
    package = ReplayPackage(package_path)
    reader = SessionReader(query_session_path)
    stream_id = stream_id or package.manifest["stream_id"]
    route_id = route_id or package.manifest["route_id"]
    if stream_id not in reader.frames_by_stream:
        raise KeyError("Unknown query stream: {}".format(stream_id))
    annotations = route_annotations(query_session_path, stream_id, route_id)
    start, complete = route_bounds(annotations, require_complete=False)
    stream_frames = reader.frames_by_stream[stream_id]
    start_ns = start["session_time_ns"] if start else stream_frames[0]["session_time_ns"]
    end_ns = complete["session_time_ns"] if complete else stream_frames[-1]["session_time_ns"]
    frames = sample_frames(stream_frames, start_ns, end_ns, query_rate_hz)
    expected_stages = make_stages(start, complete, annotations) if start and complete else None
    observer = IncrementalReplayObserver(
        package,
        max_advance=max_advance,
        distance_threshold=distance_threshold,
        min_margin=min_margin,
    )
    records = []
    for frame, image in _selected_images(reader, stream_id, frames):
        before_ns = time.perf_counter_ns()
        record = observer.observe_image(image, timestamp_ns=frame["session_time_ns"])
        record["processing_latency_ms"] = (time.perf_counter_ns() - before_ns) / 1.0e6
        record["query_frame_index"] = frame["frame_index"]
        expected_label = None
        if expected_stages:
            expected_label = stage_for_time(expected_stages, frame["session_time_ns"])["label"]
        record["expected_stage_label"] = expected_label
        records.append(record)

    accepted = [record for record in records if record["accepted"]]
    latencies = [record["processing_latency_ms"] for record in records]
    evaluated = [record for record in records if record["expected_stage_label"] is not None]
    correct = [record for record in evaluated if record["expected_stage_label"] == record["stage_label"]]
    summary = {
        "schema_version": "1.0",
        "observer": "incremental_monotonic_viterbi_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": package.manifest["package_id"],
        "query_session_id": reader.manifest.get("session_id"),
        "route_id": route_id,
        "stream_id": stream_id,
        "query_bounds_source": "route_annotations" if start else "full_session",
        "completion_observed": complete is not None,
        "observation_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(records) - len(accepted),
        "accepted_fraction": len(accepted) / float(len(records)),
        "final_progress": records[-1]["progress"],
        "monotonic": all(records[i]["reference_index"] <= records[i + 1]["reference_index"] for i in range(len(records) - 1)),
        "stage_label_accuracy": len(correct) / float(len(evaluated)) if evaluated else None,
        "processing_latency_ms": {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "maximum": float(np.max(latencies)),
        },
        "visual_source_quality": "decoded_primary_video",
        "files": {
            "timeline_jsonl": "timeline.jsonl",
            "timeline_csv": "timeline.csv",
            "timeline_html": "timeline.html",
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _write_outputs(output_path, records, summary)
    return summary
