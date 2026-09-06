"""Aggregate cost experiments with unchanged-source and source-only effects explicit."""

import argparse
import json
from pathlib import Path

from benchmarks.localization.build_template_report import build
from benchmarks.localization.run_workbench_replay import distribution, read_rows


def analyze(root):
    build(root)
    result = json.loads((root/'comparison.json').read_text())
    summaries = []
    for report in result['replays']:
        path = Path(report['report']['path'])
        rows = read_rows(path.parent/'scored_telemetry.jsonl')
        source = read_rows(path.parent/'source_telemetry.jsonl')
        settings = report['experiment'].get('scale_cost') or {}
        selected = [r for r in rows if r['session_time_ns'] <= 25e9]
        early_source = [r for r in source if r['session_time_ns'] <= 25e9]
        joint = lambda r: (r.get('xy_measurement_fresh_accepted') and r.get('cursor_pose_measurement_fresh_accepted')
                           and r['capture_to_control_publish_ms'] <= 1000/30)
        summaries.append({'cohort':report['cohort'],'settings':settings,'full_duration_s':report['duration_s'],
            'source_frames':report['source_frames'],'processed_frames':report['processed_frames'],
            'reference_error_px':report['reference_error_px'],'output_counts':report['output_counts'],
            'longest_lost_s':report['tracking_loss']['longest_lost_s'],
            'episodes':report['tracking_loss']['episode_count'],'unknown_s':report['tracking_loss']['unknown_s'],
            'publication_ms':report['publication_ms'],'engine_ms':report['engine_ms'],
            'release_lateness_ms':report['release_lateness_ms'],
            'joint_33ms_all_source_rate':report['all_source_joint_xy_heading_33ms_rate'],
            'fresh_xy_all_source_rate':report['all_source_fresh_xy_rate'],
            'largest_step_px':report['output_step_px']['worst'],'steps_over_8px':report['steps_over_8px'],
            'wrong_mode_fresh':sum(bool(r.get('xy_measurement_fresh_accepted')) and r.get('reference_mode') is not None
                                    and r.get('active_map_mode_id')!=r['reference_mode'] for r in rows),
            'bad_fresh_over_20px':sum(bool(r.get('xy_measurement_fresh_accepted')) and (r.get('reference_error_px') or 0)>20 for r in rows),
            'source_released_early':sum(r['release_lateness_ms'] < 0 for r in source),
            'first25':{'processed':len(selected),'source':len(early_source),
                       'publication_ms':distribution([r['capture_to_control_publish_ms'] for r in selected]),
                       'engine_ms':distribution([r['update_elapsed_ms'] for r in selected]),
                       'local_template_ms':distribution([(r.get('route_tracking') or {})['elapsed_ms'] for r in selected
                                                         if 'elapsed_ms' in (r.get('route_tracking') or {})]),
                       'engine_beyond_template_ms':distribution([r['update_elapsed_ms']-(r.get('route_tracking') or {})['elapsed_ms'] for r in selected
                                                         if 'elapsed_ms' in (r.get('route_tracking') or {})]),
                       'joint_rate':sum(bool(joint(r)) for r in selected)/max(1,len(early_source))}})
    (root/'cost-analysis.json').write_text(json.dumps(summaries,indent=2))
    lines = ['# Scale cost comparisons','',
             'All-source delivery rates include dropped frames. Source-wait changes are benchmark transport changes.','',
             '| Cohort | Error P95 / worst px | Longest lost s | Fresh / held / unavailable | Publish P95 ms | Joint <=33.3ms / all source |',
             '|---|---:|---:|---:|---:|---:|']
    def fmt(v): return '—' if v is None else f'{v:.3f}'
    for r in summaries:
        c = r['output_counts']
        lines.append(f"| {r['cohort']} | {fmt(r['reference_error_px']['p95'])} / {fmt(r['reference_error_px']['worst'])} | "
                     f"{r['longest_lost_s']:.3f} | {c.get('fresh',0)} / {c.get('held',0)} / {c.get('unavailable',0)} | "
                     f"{fmt(r['publication_ms']['p95'])} | {100*r['joint_33ms_all_source_rate']:.2f}% |")
    (root/'COST_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
