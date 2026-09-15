import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from acquisition.workbench import AcquisitionWorkbench
from tests.test_route_guidance import package


class RouteGuidanceWorkbenchTests(unittest.TestCase):
    def test_route_only_start_observe_stop_and_download_without_atlas(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifacts=root/'artifacts';guide=artifacts/'route_guides/genshin-impact-pc/route'
            meta,_,_=package(guide);meta['game_profile_id']='genshin-impact-pc';(guide/'guide.json').write_text(json.dumps(meta))
            worker=root/'fixture.py'
            worker.write_text("import json,time,sys\nfrom pathlib import Path\np=Path(sys.argv[1])\n(p/'guidance.jsonl').write_text('{}\\n')\n(p/'inputs.jsonl').write_text('{}\\n')\nwhile not (p/'stop').exists():\n (p/'latest.json').write_text(json.dumps(dict(status='running',available=True,capture_ns=time.perf_counter_ns(),processed_frames=1,hint=dict(label='Keep current input',range=[-.2,.2],current=0))))\n time.sleep(.02)\n(p/'latest.json').write_text(json.dumps(dict(status='stopped',available=False,hint=None,processed_frames=1)))\n")
            original=AcquisitionWorkbench._run_route_guidance
            # Deliberately unrelated perf-counter origins; only epoch publication
            # time and within-worker latency may cross the process boundary.
            worker.write_text(worker.read_text().replace('capture_ns=time.perf_counter_ns(),processed_frames=1', 'capture_ns=time.perf_counter_ns(),published_wall_ns=time.time_ns(),capture_to_result_ms=0,processed_frames=1'))
            def use_fixture(state,runtime,output,command):return original(state,runtime,output,[sys.executable,str(worker),str(output)])
            state=AcquisitionWorkbench(root/'sessions',artifacts)
            try:
                payload=dict(game_profile_id='genshin-impact-pc',guide_id='route',frame_source=dict(adapter='windows_window',window_title='Fixture game'),record_video=False)
                with patch('aria_trace.apps.workbench.route_guidance.runtime_configuration',return_value=(Path(sys.executable),root)),patch.object(AcquisitionWorkbench,'_run_route_guidance',use_fixture):
                    with self.assertRaises(ValueError):state.start_route_guidance(dict(payload,frame_source=dict(adapter='android_scrcpy')))
                    state.start_route_guidance(payload)
                    deadline=time.monotonic()+8
                    while state._live_tracker['status']=='starting' and time.monotonic()<deadline:time.sleep(.02)
                    self.assertEqual(state._live_tracker['status'],'running')
                    with self.assertRaises(RuntimeError):state.start_route_guidance(payload)
                    hud=state.hud_descriptor();self.assertEqual(hud['state'],'route_guidance');self.assertIsNotNone(hud['guidance']);self.assertNotIn('map_overlay_url',hud)
                    descriptor=state.descriptor();json.dumps(descriptor);self.assertEqual(len(descriptor['guide_packages']['genshin-impact-pc']),1)
                    identity=state._live_tracker['tracking_id'];state.stop_live_tracker();state._live_tracker['thread'].join(5)
                    self.assertEqual(state._live_tracker['status'],'stopped');self.assertIsNone(state._live_tracker['latest']['hint'])
                    self.assertTrue(state.guidance_telemetry('genshin-impact-pc',identity).startswith(b'PK'))
                    with self.assertRaises(ValueError):state.guidance_video('genshin-impact-pc',identity)
                    saved=state._live_tracking_runs()['genshin-impact-pc'][0];self.assertEqual(saved['tracking_mode'],'guidance')
                    with self.assertRaises(ValueError):state.guidance_telemetry('genshin-impact-pc','../outside')
            finally:state.close()


if __name__=='__main__':unittest.main()
