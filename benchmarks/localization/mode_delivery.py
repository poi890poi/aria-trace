"""Isolate consuming an already-ready mode observation after XY processing.

No new image observations, score changes or relaxed confirmation are introduced.
Freeze and save the executed update method alongside each replay.
"""

from contextlib import contextmanager
import inspect
import textwrap
from unittest.mock import patch

from aria_trace.services.tracking import runtime


@contextmanager
def consume_after_xy():
    source = textwrap.dedent(inspect.getsource(runtime.TwoRateRealtimeTracker.update))
    anchor = '    # Overlap cursor processing with XY work, then wait only within the\n'
    if source.count(anchor) != 1:
        raise RuntimeError('Mode-delivery experiment does not match the current tracker')
    source = source.replace(anchor,
        '    # Apply a completed representation result before publishing this frame.\n'
        '    representation_observation_fresh = (\n'
        '        self._consume_representation_observation(timestamp_ns)\n'
        '        or representation_observation_fresh\n'
        '    )\n'+anchor)
    namespace = dict(vars(runtime))
    exec(compile(source, '<mode-delivery-experiment>', 'exec'), namespace)
    with patch.object(runtime.TwoRateRealtimeTracker,'update',namespace['update']):
        yield source
