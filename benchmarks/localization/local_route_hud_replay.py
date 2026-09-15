"""Real Python 3.7 Workbench / Python 3.12 GPU worker freshness integration."""
import json
import threading
import time
from pathlib import Path
from aria_trace.apps.workbench.route_guidance import WorkbenchRouteGuidanceMixin, guidance_hud
from aria_trace.services.tracking.destination_monitor import runtime_configuration


def main():
    output=Path('artifacts/poc/local-route-control-20260915/paced-hud-158-174').resolve()
    output.mkdir(parents=True,exist_ok=True)
    repo=Path.cwd();python,root=runtime_configuration()
    config=dict(output=str(output),package=str(repo/'artifacts/workbench/route_guides/genshin-impact-pc/town-circuit-run21'),
                game_profile_id='genshin-impact-pc',runtime_root=str(root),record_video=False,
                replay=dict(session=str(repo/'sessions/workbench/recordings-genshin-impact-pc/run_21'),start_s=158,end_s=174))
    (output/'config.json').write_text(json.dumps(config))
    class Owner(WorkbenchRouteGuidanceMixin):
        _live_tracker_lock=threading.RLock()
    runtime=dict(status='starting',stop=threading.Event(),tracking_mode='guidance',frame_source={})
    owner=Owner();command=[str(python),'-u','-m','aria_trace.services.tracking.route_guidance_worker','--config',str(output/'config.json')]
    thread=threading.Thread(target=owner._run_route_guidance,args=(runtime,output,command));thread.start();rows=[]
    try:
        while thread.is_alive():
            if runtime.get('latest',{}).get('status')=='running':
                value=guidance_hud(runtime);rows.append(dict(hint=value['guidance'],age_ms=value['capture_age_ms'],worker_available=runtime['latest']['available']))
            time.sleep(.1)
    finally:
        runtime['stop'].set();thread.join(6)
    result=dict(status=runtime['status'],error=runtime.get('error'),hud_samples=len(rows),hud_advice=sum(bool(r['hint']) for r in rows),
                fresh_worker_advice_suppressed=sum(r['worker_available'] and 0<=r['age_ms']<=500 and not r['hint'] for r in rows),
                stale_advice_displayed=sum(bool(r['hint']) and r['age_ms']>500 for r in rows))
    (output/'hud-samples.json').write_text(json.dumps(rows));(output/'hud-summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    assert runtime['status']=='stopped' and result['hud_advice']>0 and not result['fresh_worker_advice_suppressed'] and not result['stale_advice_displayed']


if __name__=='__main__':main()
