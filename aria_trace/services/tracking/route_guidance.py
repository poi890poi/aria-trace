"""Local map course guidance. Observes human input; has no input executor."""
import hashlib
import json
import math
import threading
import zipfile
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from .destination import checked_id, MODEL, MODEL_SHA256

SCHEMA = 'aria-route-guide-1'
MAX_AGE_S = .5


def load_guide(path, game_id=None):
    path = Path(path)
    if (path/'guide.json').stat().st_size > 4*1024*1024:
        raise ValueError('Route guide metadata is too large')
    meta = json.loads((path/'guide.json').read_text(encoding='utf-8'))
    if meta.get('schema') != SCHEMA or meta.get('model') != MODEL or meta.get('model_sha256') != MODEL_SHA256:
        raise ValueError('Unsupported route guide or feature model')
    checked_id(meta['guide_id']); checked_id(meta['game_profile_id'])
    if game_id is not None and meta['game_profile_id'] != game_id:
        raise ValueError('This guide belongs to another game profile')
    if meta.get('image_size_wh') != [1280,720] or not 1 <= len(meta['views']) <= 512:
        raise ValueError('Unsupported route guide geometry or size')
    if not .001 <= float(meta['motor_gain_deg_per_count']) <= 1:
        raise ValueError('Invalid input calibration')
    scale=float(meta.get('turn_display_scale',1500))
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid turning display scale')
    boxes=[meta.get('scene_mask',{}).get('include',[])]+meta.get('scene_mask',{}).get('exclude',[])
    if len(boxes)>16:raise ValueError('Invalid scene mask')
    for box in boxes:
        if len(box)!=4 or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=360):raise ValueError('Invalid scene mask')
    files = {}
    for name in ['arrays.npz','maps.npy']:
        file = path/name
        if file.stat().st_size > 128*1024*1024 or hashlib.sha256(file.read_bytes()).hexdigest() != meta['sha256'][name]:
            raise ValueError('Route guide data failed its integrity check')
        files[name] = file
    with zipfile.ZipFile(files['arrays.npz']) as z:
        if sum(v.file_size for v in z.infolist()) > 192*1024*1024:
            raise ValueError('Expanded route guide is too large')
    with np.load(files['arrays.npz'], allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    maps = np.load(files['maps.npy'], allow_pickle=False)
    n = len(meta['views'])
    if maps.shape != (n,180,220) or maps.dtype != np.uint8:
        raise ValueError('Invalid prepared minimaps')
    if arrays['centers'].shape != (32,128) or arrays['signatures'].shape != (n,4096) or arrays['mask'].shape != (180,220):
        raise ValueError('Invalid route descriptor geometry')
    if arrays['mask'].dtype!=np.uint8 or np.count_nonzero(arrays['mask'])<100:raise ValueError('Invalid minimap mask')
    for a in arrays.values():
        if a.dtype.kind not in 'fubi' or not np.isfinite(a).all():
            raise ValueError('Non-finite or unsupported route data')
    keys = set()
    for v in meta['views']:
        key = v['key']
        if key in keys:
            raise ValueError('Repeated route view')
        keys.add(key); p = arrays[key+'p']; d = arrays[key+'d']
        if p.ndim != 2 or p.shape[1] != 2 or not 0 <= len(p) <= 900 or d.shape != (len(p),128):
            raise ValueError('Invalid scene features')
        if not math.isfinite(v['phase']) or not 0 <= v['phase'] <= 1:
            raise ValueError('Invalid route ordering')
        if len(v['paths'])>3:raise ValueError('Too many local route paths')
        for curve in v['paths']:
            a = np.asarray(curve['points'], dtype=float)
            if a.ndim != 2 or a.shape[1] != 2 or not 2 <= len(a) <= 128 or not np.isfinite(a).all() or abs(a).max() > 1000:
                raise ValueError('Invalid local route path')
    return meta, arrays, maps


def catalog(root):
    out = {}
    for p in sorted(Path(root).glob('*/*/guide.json')):
        try:
            m = json.loads(p.read_text(encoding='utf-8'))
            if m.get('schema') != SCHEMA or m.get('guide_id') != p.parent.name or m.get('game_profile_id') != p.parent.parent.name:
                continue
            checked_id(m['guide_id']); checked_id(m['game_profile_id'])
            out.setdefault(m['game_profile_id'],[]).append(dict(guide_id=m['guide_id'], label=m['label']))
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return out


def map_shift(a, b, mask):
    weight = mask.astype(np.float32)/255
    aa = a.astype(np.float32); bb = b.astype(np.float32)
    aa = (aa-cv2.GaussianBlur(aa,(0,0),3))*weight
    bb = (bb-cv2.GaussianBlur(bb,(0,0),3))*weight
    h,w = a.shape; hh=cv2.getOptimalDFTSize(h); ww=cv2.getOptimalDFTSize(w)
    while hh%2: hh=cv2.getOptimalDFTSize(hh+1)
    while ww%2: ww=cv2.getOptimalDFTSize(ww+1)
    window=cv2.createHanningWindow((w,h),cv2.CV_32F)*weight
    aa=cv2.copyMakeBorder(aa*window,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT)
    bb=cv2.copyMakeBorder(bb*window,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT)
    z,response=cv2.phaseCorrelate(aa,bb)
    return np.array(z),float(response)


def path_target(path, position, lookahead):
    path=np.asarray(path); vectors=np.diff(path,axis=0); length=np.linalg.norm(vectors,axis=1)
    valid=length>1e-6
    if not valid.any(): return None
    f=np.clip(np.sum((position-path[:-1])*vectors,axis=1)/np.maximum(length**2,1e-9),0,1)
    near=path[:-1]+f[:,None]*vectors; dist=np.linalg.norm(near-position,axis=1); dist[~valid]=np.inf
    k=int(np.argmin(dist)); arc=np.r_[0,np.cumsum(length)]; wanted=arc[k]+f[k]*length[k]+lookahead
    if wanted>arc[-1]: return None
    j=min(len(length)-1,int(np.searchsorted(arc,wanted,side='right')-1))
    return path[j]+(wanted-arc[j])/max(length[j],1e-9)*vectors[j]


class HumanInput:
    def __init__(self):
        self.lock=threading.Lock(); self.events=deque(); self.keys=set(); self.error=None

    def feed(self, kind, payload, timestamp_s):
        if kind=='pc_raw_input_error':
            with self.lock:self.error=payload.get('error','Raw input capture failed');self.keys.clear()
            return
        if not payload.get('foreground',True): return
        with self.lock:
            if kind=='pc_raw_mouse':
                value=float(payload.get('delta_x',0))
                if math.isfinite(value): self.events.append((timestamp_s,value))
            elif kind=='pc_raw_keyboard':
                key=payload.get('key_name')
                self.keys.add(key) if payload.get('pressed') else self.keys.discard(key)
            while self.events and self.events[0][0] < timestamp_s-2: self.events.popleft()

    def snapshot(self, timestamp_s):
        with self.lock:
            if self.error:return dict(keys=[],events=[],error=self.error)
            return dict(keys=list(self.keys), events=[(t,x) for t,x in self.events if timestamp_s-1 <= t <= timestamp_s])


def input_hint(desired, actual):
    lo,hi=desired
    if lo <= actual <= hi:
        return 'Maintain right turn' if lo>0 else 'Maintain left turn' if hi<0 else 'Keep current input'
    target=lo if actual<lo else hi
    if actual>0 and target>=0 and target<actual: return 'Ease right turn'
    if actual<0 and target<=0 and target>actual: return 'Ease left turn'
    return 'Turn right' if target>actual else 'Turn left'


class LocalCourseGuide:
    def __init__(self, meta, arrays, maps):
        self.meta=meta; self.maps=maps; self.mask=arrays['mask']; self.history=deque()
        yy,xx=np.indices(self.mask.shape)
        self.submasks=[self.mask*(xx<110),self.mask*(xx>=110),self.mask*(yy>85)]
        self.context=None; self.gain=float(meta['motor_gain_deg_per_count'])

    def _measure(self, a, b):
        z,response=map_shift(a,b,self.mask)
        if response<.35 or np.linalg.norm(z)>20: return None
        differences=[]
        for m in self.submasks:
            q,r=map_shift(a,b,m)
            if r>=.35 and np.linalg.norm(q)<=20: differences.append(float(np.linalg.norm(q-z)))
        if len(differences)<2 or max(differences)>1: return None
        return z,max(differences),response

    def step(self, minimap, capture_s, scene, human):
        if self.history and capture_s<=self.history[-1][0]:
            raise ValueError('Guidance frames must have increasing timestamps')
        self.history.append((capture_s,minimap.copy()))
        while self.history and self.history[0][0]<capture_s-1: self.history.popleft()
        if scene and scene.get('view') is not None and scene['capture_s']<=capture_s:
            if self.context is None or scene['capture_s']>self.context['capture_s']: self.context=dict(scene)
        result=dict(capture_s=capture_s, available=False, hint=None, reason='Finding route context', context=self.context)
        if human.get('error'):result['reason']='Input capture unavailable';return result
        if self.context is None or capture_s-self.context['capture_s']>1: return result
        past=min(self.history,key=lambda r:abs(r[0]-capture_s+.5)); dt=capture_s-past[0]
        if not .25<=dt<=.75: result['reason']='Measuring travel'; return result
        keys=set(human.get('keys',[]))
        if 'W' not in keys or keys&set('ASD'):
            result['reason']='Forward travel needed for this trial'; return result
        index=self.context['view']; relation=self._measure(self.maps[index],minimap); motion=self._measure(past[1],minimap)
        if relation is None or motion is None: result['reason']='Map evidence unavailable'; return result
        position=-relation[0]; velocity=-motion[0]/dt; speed=float(np.linalg.norm(velocity))
        if speed<2: result['reason']='Measuring forward travel'; return result
        events=human.get('events',[])
        turn=sum(x for t,x in events if capture_s-dt/2<t<=capture_s)*self.gain
        c=math.cos(math.radians(turn)); s=math.sin(math.radians(turn))
        velocity=np.array([c*velocity[0]-s*velocity[1],s*velocity[0]+c*velocity[1]])
        rates=[]; goals=[]
        for path in self.meta['views'][index]['paths']:
            goal=path_target(path['points'],position,speed)
            if goal is None: continue
            delta=goal-position; distance=float(np.linalg.norm(delta))
            if distance<.2: continue
            angle=math.atan2(velocity[0]*delta[1]-velocity[1]*delta[0],np.dot(velocity,delta))
            if abs(angle)>math.pi/2: continue  # Do not recommend an unprepared reversal.
            # Spatial disagreement is an empirical support spread, not a confidence interval.
            spread=math.atan2(max(.06,relation[1]),distance)+math.atan2(max(.06,motion[1]),speed*dt)
            for a in [max(-math.pi/2,angle-spread),min(math.pi/2,angle+spread)]:
                rates.append(math.degrees(2*speed*math.sin(a)/distance)/self.gain)
            goals.append(goal.tolist())
        if not rates: result['reason']='Finding the next supported route window'; return result
        desired=[min(rates),max(rates)]
        actual=sum(x for t,x in events if capture_s-.2<t<=capture_s)/.2
        target=float(np.clip(actual,*desired)); scale=float(self.meta.get('turn_display_scale',1500))
        hint=dict(label=input_hint(desired,actual),desired_mouse_rate=desired,actual_mouse_rate=actual,
                  input_adjustment=target-actual,range=[float(np.clip(x/scale,-1,1)) for x in desired],
                  current=float(np.clip(actual/scale,-1,1)),demonstrations=len(goals))
        result.update(available=True,hint=hint,reason=None,position_relative_to_view=position.tolist(),
                      goal_points=goals,course_velocity=velocity.tolist(),predicted_rotation_deg=turn,
                      map_support_spread_px=relation[1],motion_support_spread_px=motion[1],
                      scene_age_s=capture_s-self.context['capture_s'])
        return result
