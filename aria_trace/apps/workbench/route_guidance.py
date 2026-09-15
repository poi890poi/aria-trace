"""Optional human guidance lifecycle, isolated from atlas tracking and actuation."""
import io
import json
import os
import subprocess
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from aria_trace.services.tracking.destination import checked_id
from aria_trace.services.tracking.destination_monitor import runtime_configuration
from aria_trace.services.tracking.route_guidance import catalog, load_guide


def guidance_hud(tracker, now_ns=None):
    latest=tracker.get('latest') or {}
    transit=((time.time_ns() if now_ns is None else now_ns)-latest.get('published_wall_ns',0))/1e9
    age=transit+latest.get('capture_to_result_ms',0)/1000
    fresh=tracker.get('status')=='running' and latest.get('status')=='running' and 0<=transit and 0<=age<=.5
    hint=latest.get('hint') if fresh and latest.get('available') else None
    reason=latest.get('reason') or 'Finding route context'
    if not fresh:reason=tracker.get('error') or ('Waiting for a fresh frame' if tracker.get('status')=='running' else 'Guidance '+tracker.get('status','stopped'))
    short={'Finding route context':'Finding route','Waiting for a fresh frame':'Waiting for frame',
           'Forward travel needed for this trial':'Hold W to guide','Measuring forward travel':'Measuring travel',
           'Finding the next supported route window':'Finding next window'}.get(reason,reason)
    return dict(visible=True,title='ROUTE GUIDANCE · YOU DRIVE',state='route_guidance',
                window_title=tracker.get('frame_source',{}).get('window_title'),
                status=hint['label'] if hint else short,
                detail='Shaded range: suggested turn · dot: your input' if hint else 'Advice unavailable · you remain in control',
                color='#72dfa3' if hint else '#ffd166',guidance=hint,
                capture_age_ms=age*1000 if latest.get('capture_ns') else None)


class WorkbenchRouteGuidanceMixin:
    def _guide_packages(self):
        return catalog(self.artifact_root/'route_guides')

    def guidance_status(self):
        tracker=self._live_tracker_descriptor() or {}
        return guidance_hud(tracker) if tracker.get('tracking_mode')=='guidance' else dict(visible=False,guidance=None)

    def guidance_telemetry(self, game_id, tracking_id):
        checked_id(game_id);checked_id(tracking_id)
        root=self._live_tracking_root(game_id)/tracking_id
        if not (root/'guidance.jsonl').is_file():raise ValueError('No guidance has been recorded for this run')
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in ['guidance.jsonl','inputs.jsonl','summary.json','live_tracking.json']:
                if (root/name).is_file():archive.write(root/name,name)
        return stream.getvalue()

    def guidance_video(self, game_id, tracking_id):
        checked_id(game_id);checked_id(tracking_id)
        root=self._live_tracking_root(game_id)/tracking_id
        if not (root/'guidance.jsonl').is_file():raise ValueError('This is not a guidance recording')
        video=root/'video'/'route_trace_game_overlay.mkv'
        if not video.is_file():raise ValueError('No video was recorded for this guidance run')
        return video

    def guidance_video(self, game_id, tracking_id):
        checked_id(game_id);checked_id(tracking_id)
        root=self._live_tracking_root(game_id)/tracking_id
        if not (root/'guidance.jsonl').is_file():raise ValueError('This is not a guidance recording')
        video=root/'video'/'route_trace_game_overlay.mkv'
        if not video.is_file():raise ValueError('No video was recorded for this guidance run')
        return video

    def start_route_guidance(self, value):
        with self._lock:
            if self._active is not None or self._tracker_running():raise RuntimeError('Stop the current recording or tracker first')
            if self._analysis_jobs.has_active_job():raise RuntimeError('Wait for background analysis to finish')
            game=checked_id(value.get('game_profile_id'));guide_id=checked_id(value.get('guide_id'))
            self.profiles.game(game)
            frame=value.get('frame_source') or {}
            if frame.get('adapter')!='windows_window' or not frame.get('window_title'):raise ValueError('Choose the Windows game window for this trial')
            if not isinstance(value.get('record_video',False),bool):raise ValueError('Recording option must be on or off')
            package=self.artifact_root/'route_guides'/game/guide_id
            try:
                meta,_,_=load_guide(package,game);python,root=runtime_configuration()
            except (OSError,KeyError) as exc:raise ValueError('Prepared guidance route is unavailable: {}'.format(exc))
            tracking_id='guide-'+uuid.uuid4().hex[:12]
            output=(self._live_tracking_root(game)/tracking_id).resolve();output.mkdir(parents=True)
            frame=dict(adapter='windows_window',window_title=str(frame['window_title']),exact_title=True,fps=30.)
            config=dict(package=str(package.resolve()),game_profile_id=game,output=str(output),runtime_root=str(root),frame_source=frame,record_video=value.get('record_video',False))
            (output/'config.json').write_text(json.dumps(config),encoding='utf-8')
            runtime=dict(status='starting',tracking_mode='guidance',tracking_id=tracking_id,game_profile_id=game,
                         guide_id=guide_id,label=meta['label'],frame_source=frame,record_video=config['record_video'],
                         started_utc=datetime.now(timezone.utc).isoformat(),latest=None,error=None,
                         stop=threading.Event())
            command=[str(python),'-u','-m','aria_trace.services.tracking.route_guidance_worker','--config',str(output/'config.json')]
            thread=threading.Thread(target=self._run_route_guidance,args=(runtime,output,command),name='manual-route-guidance',daemon=False)
            runtime['thread']=thread
            with self._live_tracker_lock:
                self._live_tracker=runtime;self._live_tracker_mosaic=None;self._live_tracker_route_points=None
            thread.start()
        return self.descriptor()

    def _run_route_guidance(self, runtime, output, command):
        process=None;start=time.monotonic();stopping=None;error=None
        try:
            env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[3])
            with (output/'worker.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=env['PYTHONPATH'],env=env,stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while True:
                    try:
                        latest=json.loads((output/'latest.json').read_text(encoding='utf-8'))
                        with self._live_tracker_lock:
                            runtime['latest']=latest
                            if latest['status']=='running':runtime['status']='running'
                    except (OSError,ValueError,KeyError):pass
                    if process.poll() is not None:break
                    now=time.monotonic()
                    if runtime['stop'].is_set() and stopping is None:
                        (output/'stop').touch();stopping=now
                    if stopping is not None:
                        # Sensors close before optional video finalization. Hung capture gets a short deadline.
                        allowance=70 if (runtime.get('latest') or {}).get('status')=='finalizing' else 4
                        if now-stopping>allowance:raise RuntimeError('Guidance worker did not stop in time')
                    elif runtime['status']=='starting' and now-start>90:raise RuntimeError('Guidance feature runtime did not become ready')
                    time.sleep(.05)
                latest=runtime.get('latest') or {}
                if process.returncode or latest.get('error'):raise RuntimeError(latest.get('error') or 'Guidance worker exited; see worker.log')
        except Exception as exc:error='{}: {}'.format(type(exc).__name__,exc)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
            with self._live_tracker_lock:
                runtime.update(status='failed' if error else 'stopped',error=error)
                runtime['latest']=dict(runtime.get('latest') or {},available=False,hint=None)
                summary={k:v for k,v in runtime.items() if k not in ('stop','thread')}
            (output/'live_tracking.json').write_text(json.dumps(summary),encoding='utf-8')
