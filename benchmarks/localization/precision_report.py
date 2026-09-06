"""Post-run precision/transition comparison, with original-source timing."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.localization.build_template_report import build
from benchmarks.localization.run_workbench_replay import distribution, read_rows


def analyze(root):
    build(root)
    result = []
    transitions = json.loads(Path('artifacts/poc/narrow-lap-20260906/lap-analysis.json').read_text())['reference_transition_brackets']
    for path in sorted(root.glob('*/run*/report.json')):
        report = json.loads(path.read_text())
        rows = read_rows(path.parent/'scored_telemetry.jsonl')
        events = []
        previous = None
        for row in rows:
            mode = row.get('active_map_mode_id')
            if mode and mode != previous:
                events.append({'time_s': row['session_time_ns']/1e9, 'from': previous, 'to': mode})
                previous = mode
        windows = []
        if report['session'] == 20:
            for change in transitions:
                a, b = change['last_old_reference_s']-1, change['first_new_reference_s']+1
                selected = [r for r in rows if a <= r['session_time_ns']/1e9 <= b]
                if not selected:
                    continue
                windows.append({'from': change['from'], 'to': change['to'], 'time_window_s': [a, b],
                                'error_px': distribution([r['reference_error_px'] for r in selected if r.get('reference_error_px') is not None]),
                                'fresh_rate': float(np.mean([bool(r.get('xy_measurement_fresh_accepted')) for r in selected]))})
        splits = []
        for a, b in [(0, 80), (80, 160), (160, 240)] if report['session'] == 20 else [(0, report['duration_s']+1)]:
            selected = [r for r in rows if a <= r['session_time_ns']/1e9 < b]
            splits.append({'source_window_s': [a, b], 'error_px': distribution([r['reference_error_px'] for r in selected if r.get('reference_error_px') is not None])})
        scale_rows = [(r.get('route_tracking') or {}).get('precision_scale') for r in rows]
        scale_rows = [r for r in scale_rows if r]
        result.append({'cohort': path.parent.parent.name, 'session': report['session'], 'events': events, 'crossing_windows': windows,
                       'lap_windows': splits, 'bad_fresh_over_20px': sum(bool(r.get('xy_measurement_fresh_accepted')) and (r.get('reference_error_px') or 0)>20 for r in rows),
                       'wrong_layer_fresh': sum(bool(r.get('xy_measurement_fresh_accepted')) and r.get('reference_mode') is not None and r.get('active_map_mode_id')!=r['reference_mode'] for r in rows),
                       'scale_sweep_calls': sum(r.get('hypothesis_count', 0)>1 for r in scale_rows), 'scale_diagnostic_calls': len(scale_rows)})
        if report['session'] != 20 or report['duration_s'] < 200:
            continue
        canvas = np.full((650, 1350, 3), 250, np.uint8)
        cv2.putText(canvas, path.parent.parent.name+' | proxy disagreement (px) / source time (s)', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (25,25,25), 2)
        for panel, ymax, field in [(0,40.,'reference_error_px'), (1,120.,'capture_to_control_publish_ms')]:
            top = 65 + panel*285; bottom = top+210
            for tick in range(5):
                y = bottom-round(tick*210/4)
                cv2.line(canvas,(65,y),(1310,y),(215,215,215),1)
                cv2.putText(canvas,f'{ymax*tick/4:g}',(8,y+5),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,30,30),1)
            for t in range(0,241,30):
                x = 65+round(t/240*1245)
                cv2.putText(canvas,str(t),(x-8,bottom+20),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,30,30),1)
            for change in transitions:
                x = 65+round(change['first_new_reference_s']/240*1245)
                cv2.line(canvas,(x,top),(x,bottom),(170,170,230),1)
            for r in rows:
                value = r.get(field)
                if value is None:
                    continue
                x = 65+round(r['session_time_ns']/1e9/240*1245)
                y = bottom-round(min(ymax,value)/ymax*210)
                cv2.circle(canvas,(x,y),1,(190,90,20) if r.get('xy_measurement_fresh_accepted') else (40,40,220),-1)
            label = 'Reference error; missing reference intervals omitted' if panel == 0 else 'Publication latency (ms); clipped at 120 ms; red = held/unavailable'
            cv2.putText(canvas,label,(65,top-10),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        cv2.imwrite(str(root/(path.parent.parent.name+'-timeline.png')),canvas)
    (root/'precision-analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
