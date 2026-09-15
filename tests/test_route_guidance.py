import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from aria_trace.services.tracking.route_guidance import HumanInput, LocalCourseGuide, map_shift, path_target, load_guide, input_hint, SCHEMA
from aria_trace.services.tracking.destination import MODEL, MODEL_SHA256, DEFAULT_MASK
from aria_trace.apps.workbench.route_guidance import WorkbenchRouteGuidanceMixin, guidance_hud
from aria_trace.apps.workbench.hud import _UrlStatusProvider


def package(folder):
    folder.mkdir(parents=True)
    mask=np.zeros((180,220),np.uint8);cv2.circle(mask,(110,85),65,255,-1)
    maps=np.random.RandomState(71).randint(0,255,(1,180,220),dtype=np.uint8)
    arrays=dict(mask=mask,centers=np.zeros((32,128),np.float32),signatures=np.zeros((1,4096),np.float32),vp=np.empty((0,2),np.float32),vd=np.empty((0,128),np.float32))
    np.save(str(folder/'maps.npy'),maps);np.savez_compressed(str(folder/'arrays.npz'),**arrays)
    meta=dict(schema=SCHEMA,model=MODEL,model_sha256=MODEL_SHA256,guide_id='route',game_profile_id='game',label='Route',image_size_wh=[1280,720],motor_gain_deg_per_count=.064,scene_mask=DEFAULT_MASK,views=[dict(key='v',phase=0,paths=[dict(points=[[0,0],[0,-20]])])],sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['maps.npy','arrays.npz']})
    (folder/'guide.json').write_text(json.dumps(meta))
    return meta,arrays,maps


class RouteGuidanceTests(unittest.TestCase):
    def test_shift_has_no_half_pixel_identity_bias_and_has_correct_sign(self):
        rng=np.random.RandomState(21);image=rng.randint(0,255,(180,220),dtype=np.uint8);mask=np.full_like(image,255)
        shift,response=map_shift(image,image,mask)
        np.testing.assert_allclose(shift,[0,0],atol=.00001)
        moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,-2]]),(220,180))
        shift,response=map_shift(image,moved,mask)
        np.testing.assert_allclose(shift,[3,-2],atol=.1)

    def test_local_progress_moves_target_forward_and_stops_at_end(self):
        path=[[0,0],[0,-5],[5,-5]]
        np.testing.assert_allclose(path_target(path,np.array([0,-4]),3),[2,-5])
        self.assertIsNone(path_target(path,np.array([5,-5]),3))

    def test_actual_input_changes_advice_and_keys_gate_it(self):
        self.assertEqual(input_hint([100,200],150),'Maintain right turn')
        self.assertEqual(input_hint([100,200],300),'Ease right turn')
        self.assertEqual(input_hint([-200,-100],-300),'Ease left turn')
        self.assertEqual(input_hint([-10,10],100),'Ease right turn')
        h=HumanInput();h.feed('pc_raw_keyboard',dict(key_name='W',pressed=True),1)
        h.feed('pc_raw_mouse',dict(delta_x=8),1.1)
        h.feed('pc_raw_mouse',dict(delta_x=9,foreground=False),1.2)
        self.assertEqual(h.snapshot(1.15),dict(keys=['W'],events=[(1.1,8)]))
        h.feed('pc_raw_keyboard',dict(key_name='W',pressed=False),1.3)
        self.assertFalse(h.snapshot(3)['keys']);self.assertFalse(h.snapshot(3)['events'])
        h.feed('pc_raw_input_error',dict(error='device failed'),3)
        self.assertEqual(h.snapshot(3)['error'],'device failed')

    def test_causal_guidance_expires_and_rejects_bad_regions(self):
        with tempfile.TemporaryDirectory() as d:
            meta,arrays,maps=package(Path(d)/'guide');guide=LocalCourseGuide(meta,arrays,maps)
            # Known local forward displacement: map moves down while player moves up.
            def frame(t):return cv2.warpAffine(maps[0],np.float32([[1,0,0],[0,1,t*5]]),(220,180))
            human=dict(keys=['W'],events=[])
            guide.step(frame(0),0,dict(view=0,capture_s=0),human)
            r=guide.step(frame(.5),.5,dict(view=0,capture_s=.5),human)
            self.assertTrue(r['available'],r)
            self.assertLess(abs(r['hint']['input_adjustment']),50)
            r=guide.step(frame(.6),.6,None,dict(keys=[],events=[]));self.assertFalse(r['available']);self.assertIsNone(r['hint'])
            r=guide.step(frame(1.6),1.6,dict(view=0,capture_s=2),human);self.assertFalse(r['available'])
            self.assertEqual(r['context']['capture_s'],.5)
            with self.assertRaises(ValueError):guide.step(frame(1),1,None,human)
            self.assertIsNone(guide._measure(maps[0],np.zeros_like(maps[0])))

    def test_package_integrity_and_wrong_game(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'guide';package(root);load_guide(root,'game')
            with self.assertRaises(ValueError):load_guide(root,'other')
            with (root/'maps.npy').open('ab') as stream:stream.write(b'broken')
            with self.assertRaises(ValueError):load_guide(root,'game')

    def test_hud_expiry_stop_and_network_failure_never_hold_advice(self):
        tracker=dict(status='running',frame_source={},latest=dict(status='running',capture_ns=123456789000000,published_wall_ns=1000000000,capture_to_result_ms=50,available=True,hint=dict(label='Turn left',range=[-.5,-.2],current=0)))
        self.assertIsNotNone(guidance_hud(tracker,1400000000)['guidance'])
        self.assertIsNone(guidance_hud(tracker,1600000000)['guidance'])
        tracker['status']='stopped';self.assertIsNone(guidance_hud(tracker,1000000000)['guidance'])
        provider=_UrlStatusProvider('http://unused');tracker['status']='running';provider._last_value=guidance_hud(tracker,1000000000)
        with patch('urllib.request.urlopen',side_effect=OSError('offline')):
            self.assertIsNone(provider()['guidance'])

    def test_owned_worker_stop_and_failure_clear_advice(self):
        class Owner(WorkbenchRouteGuidanceMixin):
            _live_tracker_lock=threading.RLock()
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);script=output/'worker.py'
            script.write_text("import json,time\nfrom pathlib import Path\np=Path(__file__).parent\n(p/'latest.json').write_text(json.dumps(dict(status='running',available=True,hint=dict(label='Turn right'))))\nwhile not (p/'stop').exists():time.sleep(.01)\n(p/'latest.json').write_text(json.dumps(dict(status='stopped',available=False,hint=None)))\n")
            runtime=dict(status='starting',stop=threading.Event());owner=Owner()
            thread=threading.Thread(target=owner._run_route_guidance,args=(runtime,output,[sys.executable,str(script)]));thread.start()
            deadline=time.monotonic()+8
            while runtime['status']=='starting' and time.monotonic()<deadline:time.sleep(.02)
            self.assertEqual(runtime['status'],'running');runtime['stop'].set();thread.join(5)
            self.assertFalse(thread.is_alive());self.assertEqual(runtime['status'],'stopped');self.assertIsNone(runtime['latest']['hint'])
            runtime=dict(status='starting',stop=threading.Event());owner._run_route_guidance(runtime,output,[sys.executable,'-c','raise RuntimeError("fixture")'])
            self.assertEqual(runtime['status'],'failed');self.assertIsNone(runtime['latest']['hint'])


if __name__=='__main__':unittest.main()
