"""Measure recorded-source deadline waits without a tracker or video decoder."""

import argparse
import json
from pathlib import Path
import threading
import time

from benchmarks.localization.precise_release import wait_until
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import distribution


def screen(output):
    stop = threading.Event()
    rows = []
    for repetition in range(5):
        for kind in (('event','sleep') if repetition%2 == 0 else ('sleep','event')):
            cpu = time.process_time()
            block = []
            for delay_ms in (1,3,7,13,20)*4:
                deadline = time.perf_counter_ns()+delay_ms*1_000_000
                if kind == 'event':
                    stop.wait(max(0.,(deadline-time.perf_counter_ns())/1e9))
                else:
                    if not wait_until(deadline,stop):
                        raise AssertionError('Unexpected cancellation')
                lateness = (time.perf_counter_ns()-deadline)/1e6
                block.append(lateness)
                rows.append({'repeat':repetition,'kind':kind,'delay_ms':delay_ms,'lateness_ms':lateness})
            print(kind,repetition,distribution(block),'CPU ms',(time.process_time()-cpu)*1000,flush=True)
    summary = {kind:distribution([r['lateness_ms'] for r in rows if r['kind']==kind]) for kind in ('event','sleep')}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'source':identity(__file__),'summary':summary,'rows':rows},indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    screen(parser.parse_args().output)
