import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from acquisition.workbench import AcquisitionWorkbench
from tests.test_destination_observer import package
from tests import test_live_tracker as live_tests


class FailedObserver:
    def __init__(self, *args):
        self.submissions = 0
        self.closed = False

    def submit(self, *args):
        self.submissions += 1

    def snapshot(self):
        return dict(worker_status="failed", state="unavailable", aim=None, error="Simulated feature worker failure", commands_issued=0)

    def close(self):
        self.closed = True
        return self.snapshot()


class DestinationWorkbenchTests(unittest.TestCase):
    def test_optional_worker_failure_does_not_break_route_tracker_or_restart(self):
        original = AcquisitionWorkbench.start_live_tracker
        observers = []

        def start_with_destination(state, value):
            root, _, _ = package(state.artifact_root / "destination_targets" / value["game_profile_id"] / "gate")
            meta = json.loads((root / "destination.json").read_text())
            meta["game_profile_id"] = value["game_profile_id"]
            (root / "destination.json").write_text(json.dumps(meta))
            return original(state, dict(value, destination_id="gate", tracking_mode="route-assisted"))

        def factory(*args):
            observer = FailedObserver(*args)
            observers.append(observer)
            return observer

        with patch.object(AcquisitionWorkbench, "start_live_tracker", start_with_destination), patch("aria_trace.apps.workbench.live_tracking.runtime_configuration", return_value=(Path("unused"), Path("unused"))), patch("aria_trace.apps.workbench.live_tracking.DestinationMonitor", side_effect=factory):
            for _ in range(2):
                # Existing lifecycle assertions include live pose, serializable descriptor,
                # clean stop, video telemetry and persisted evidence.
                live_tests.WorkbenchLiveTrackerTests()._exercise_route_start("demonstrated-start")
        self.assertEqual(len(observers), 2)
        self.assertTrue(all(o.closed and o.submissions > 0 for o in observers))

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser request check')
    def test_destination_is_optional_and_sent_only_for_route_tracking(self):
        source = Path('aria_trace/apps/workbench/static/recorder.html').read_text(encoding='utf-8')
        handler = re.search(r'^async function startLiveTracker\(mode\).*$', source, re.M)[0]
        script = r'''
const vm=require('vm'),assert=require('assert');
const ui={game:{value:'game'},routeAtlas:{},mapAtlas:{},routeTrackingProfile:{value:'real-time'},trackingProfile:{value:'real-time'},routeStartPolicy:{value:'demonstrated-start'},routeRecordVideo:{checked:false},trackingRecordVideo:{checked:false},routeAutoStop:{checked:false},routeDestination:{value:''}};
let sent;const context={ui,selectedCalibration:()=>({}),selectedSceneYaw:()=>({}),selectedStitch:()=>null,selectedAtlas:()=>({}),selectedRoutePackage:()=>({}),selectedFrameSource:()=>({}),act:f=>f(),post:async(u,b)=>{sent=b}};
vm.createContext(context);vm.runInContext(process.argv[1],context);
(async()=>{for(const destination of ['', 'gate'])for(const mode of ['route-assisted','free-roam']){ui.routeDestination.value=destination;await context.startLiveTracker(mode);assert.strictEqual(sent.destination_id,mode==='route-assisted'?destination:'');}})().catch(e=>{console.error(e);process.exitCode=1});
'''
        result = subprocess.run([shutil.which('node'), '-e', script, handler], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
