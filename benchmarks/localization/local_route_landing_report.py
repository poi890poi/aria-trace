"""Readable paced-replay evidence for the optional human guidance trial."""
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('artifacts/poc/local-route-control-20260915')


def metrics(rows):
    t=[r['capture_s'] for r in rows]; latency=[r['capture_to_result_ms'] for r in rows]
    begin=None;lost=[];fresh=0;held=0;previous=None;reasons={};flips=0;last_sign=None
    for r in rows:
        if r['available']:
            if begin is not None:lost.append(r['capture_s']-begin);begin=None
            key=r['context']['capture_s'];fresh+=key!=previous;held+=key==previous;previous=key
            correction=r['hint']['input_adjustment'];sign=0 if abs(correction)<1 else np.sign(correction)
            if last_sign is not None and sign and last_sign and sign!=last_sign:flips+=1
            last_sign=sign
        else:
            if begin is None:begin=r['capture_s']
            reasons[r['reason']]=reasons.get(r['reason'],0)+1;last_sign=None
    if begin is not None:lost.append(t[-1]-begin)
    return dict(samples=len(rows),available=sum(r['available'] for r in rows),fresh_context=fresh,reused_context_fresh_map=held,
                duration_s=t[-1]-t[0],source_fps=(len(rows)-1)/(t[-1]-t[0]),longest_lost_s=max(lost or [0]),loss_episodes=len(lost),
                correction_sign_reversals=flips,reasons=reasons,
                capture_to_result_ms=dict(mean=float(np.mean(latency)),p50=float(np.median(latency)),p95=float(np.percentile(latency,95)),maximum=max(latency)))


def main():
    frames=[json.loads(s) for s in Path('sessions/workbench/recordings-genshin-impact-pc/run_21/frames.jsonl').read_text().splitlines()]
    allrows=[];report={}
    for name,lo,hi in [('paced-124-166',124,166),('paced-166-240',166,240),('paced-video-158-166',158,166),('paced-video-fixed-158-166',158,166)]:
        rows=[json.loads(s) for s in (ROOT/name/'guidance.jsonl').read_text().splitlines()]
        selected=[r for r in frames if lo<=r['session_time_ns']/1e9<hi]
        assert len(rows)==len(selected),(name,len(rows),len(selected))
        for r,f in zip(rows,selected):r['time_s']=f['session_time_ns']/1e9
        m=metrics(rows);report[name]=m;(ROOT/name/'metrics.json').write_text(json.dumps(m,indent=2))
        if not name.startswith('paced-video'):allrows.extend(rows)
    (ROOT/'paced-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    fig,axes=plt.subplots(3,1,figsize=(14,8),gridspec_kw={'height_ratios':[3,1,1]},sharex=True)
    times=np.array([r['time_s'] for r in allrows]);good=np.array([r['available'] for r in allrows]);actual=np.array([(r.get('hint') or {}).get('actual_mouse_rate',np.nan) for r in allrows]);lo=np.array([(r.get('hint') or {}).get('desired_mouse_rate',[np.nan,np.nan])[0] for r in allrows]);hi=np.array([(r.get('hint') or {}).get('desired_mouse_rate',[np.nan,np.nan])[1] for r in allrows])
    axes[0].fill_between(times,lo,hi,color='#329b86',alpha=.6,label='Suggested input window');axes[0].plot(times,actual,color='#28394e',lw=1,label='Actual human input');axes[0].axhline(0,color='gray',lw=.5);axes[0].set_ylabel('Mouse counts / second\nleft < 0 < right');axes[0].legend(loc='upper left');axes[0].set_title('Paced replay: input advice and actual input are separate measurements')
    axes[1].fill_between(times,0,good.astype(float),step='post',color='#329b86');axes[1].set_yticks([0,1]);axes[1].set_yticklabels(['Unavailable','Available']);axes[1].set_ylim(0,1.2)
    axes[2].plot(times,[r['capture_to_result_ms'] for r in allrows],color='#ba653a',lw=1);axes[2].set_ylabel('Result age (ms)');axes[2].set_xlabel('Session 21 time (s)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'guidance-timeline.png',dpi=150);plt.close(fig)
    cap=cv2.VideoCapture('sessions/workbench/recordings-genshin-impact-pc/run_21/video_main.mkv')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    frame_times=np.array([f['session_time_ns']/1e9 for f in frames])
    examples=[]
    for ax,t in zip(axes.flat,[135,160.44,162.98,218]):
        row=min(allrows,key=lambda r:abs(r['time_s']-t));i=int(np.argmin(abs(frame_times-row['time_s'])));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read();assert ok
        ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));ax.axis('off');hint=row.get('hint') or {}
        label=hint.get('label') or row.get('reason');ax.set_title('{:.2f} s | {}'.format(row['time_s'],label),fontsize=12)
        if hint:
            text='Suggested {:.0f} to {:.0f} | actual {:.0f} counts/s'.format(*hint['desired_mouse_rate'],hint['actual_mouse_rate'])
            ax.text(.02,.03,text,transform=ax.transAxes,color='white',fontsize=9,bbox=dict(facecolor='#182435',alpha=.9,edgecolor='none'))
        examples.append(dict(time_s=row['time_s'],hint=hint,reason=row.get('reason')))
    cap.release();fig.suptitle('Native frames with causal replay advice - not a successful driving demonstration',fontsize=14);fig.tight_layout();fig.savefig(ROOT/'guidance-examples.png',dpi=140);plt.close(fig)
    (ROOT/'examples.json').write_text(json.dumps(examples,indent=2))
    with zipfile.ZipFile(ROOT/'guidance-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in ['guidance-examples.png','guidance-timeline.png','paced-summary.json','production-cached-summary.json','examples.json']:
            z.write(ROOT/file,file)
        for name in ['paced-124-166','paced-166-240']:
            z.write(ROOT/name/'guidance.jsonl',name+'/guidance.jsonl');z.write(ROOT/name/'inputs.jsonl',name+'/inputs.jsonl')
        z.write('docs/local-route-guidance-trial.md','readme.md')
        z.write(ROOT/'paced-hud-158-174/hud-summary.json','hud-summary.json')
        z.write(ROOT/'paced-video-fixed-158-166/video/route_trace_game_overlay.mkv','corner-guidance.mkv')


if __name__=='__main__':main()
