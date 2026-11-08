"""Reports preserve confirmed progress and identify the intent that failed."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from swiftest import Session, Position, TargetNotFoundError
from swiftest.cli import run_script, run_suite
from swiftest.intent import execute


class Missing:
    def matches(self, obs):
        return []


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.taps = []
        self.session = Session(device=SimpleNamespace(
            display_size=lambda: (100, 100), close=lambda: None,
            tap=lambda x, y, duration=0: self.taps.append((x, y)),
            text=lambda value: None), timeout=0, base=self.base)
        self.addCleanup(self.session.close)
        self.session.register('missing', Missing())

    def test_failed_intent_preserves_previous_checkpoint_and_exception(self):
        self.session.checkpoint({'level': 3}, 'level loaded')
        with self.assertRaises(TargetNotFoundError):
            with self.session.intent('Open rewards'):
                self.session.touch('missing')
        outcome = self.session.outcome()
        self.assertEqual(outcome['status'], 'failed')
        self.assertEqual(outcome['last_valid_result']['result'], {'level': 3})
        self.assertEqual(outcome['failed_intent']['intent'], 'Open rewards')
        self.assertEqual(self.taps, [])
        json.dumps(outcome, allow_nan=False)

    def test_dispatch_is_not_reported_as_verified_postcondition(self):
        self.session.touch(Position(.5, .5))
        self.assertEqual(self.session.last_valid_result['confirmation'], 'dispatch')
        previous = self.session.last_valid_result.copy()
        with self.assertRaises(TargetNotFoundError):
            self.session.touch(Position(.5, .5), expect='missing')
        self.assertEqual(self.session.last_valid_result, previous)
        self.assertEqual(len(self.taps), 2)

    def test_checkpoint_inside_failed_group_retains_partial_progress(self):
        with self.assertRaises(TargetNotFoundError):
            with self.session.intent('Finish level'):
                self.session.checkpoint({'score': 42}, 'Score observed')
                self.session.wait('missing')
        self.assertEqual(self.session.last_valid_result['intent'], 'Score observed')
        self.assertEqual(self.session.last_valid_result['result'], {'score': 42})
        self.assertEqual(self.session.failed_intent['intent'], 'Finish level')

    def test_system_exit_is_reported_and_later_cases_continue(self):
        path = self.base / 'test_exit.py'
        path.write_text("def test_exit(session):\n    session.checkpoint(42)\n    raise SystemExit(3)\n"
                        "def test_ok(session):\n    raise SystemExit(0)\n")
        cases = run_script(path, self.session)
        self.assertEqual([c['status'] for c in cases], ['failed', 'passed'])
        self.assertEqual(cases[0]['outcome']['last_valid_result']['result'], 42)
        self.assertEqual(cases[0]['outcome']['failed_intent']['error_type'], 'SystemExit')

    def test_top_level_error_preserves_checkpoint(self):
        path = self.base / 'test_top.py'
        path.write_text("from swiftest import checkpoint\ncheckpoint({'ready': True})\n"
                        "raise RuntimeError('top-level failure')\n")
        with self.session:
            with self.assertRaises(RuntimeError):
                run_script(path, self.session)
        self.assertEqual(self.session.last_valid_result['result'], {'ready': True})
        self.assertEqual(self.session.failed_intent['intent'], path.name)

    def test_reraised_error_after_recovery_gets_a_failed_outcome(self):
        path = self.base / 'test_recovered.py'
        path.write_text("def test_recovered(session):\n"
                        "    try:\n        session.wait('missing')\n"
                        "    except AssertionError as exc:\n        saved = exc\n"
                        "    session.checkpoint({'recovered': True})\n"
                        "    raise saved\n")
        case = run_script(path, self.session)[0]
        self.assertEqual(case['status'], 'failed')
        self.assertEqual(case['outcome']['status'], 'failed')
        self.assertEqual(case['outcome']['failed_intent']['intent'], 'test_recovered')
        self.assertEqual(case['outcome']['last_valid_result']['result'], {'recovered': True})

    def test_python_cases_reset_results_and_keep_first_failure(self):
        path = self.base / 'test_flow.py'
        path.write_text("""def test_first(session):
    session.checkpoint({'stage': 'loaded'})
    with session.intent('Claim reward'):
        session.wait('missing')
def test_second(session):
    return {'score': 10}
def test_third(session):
    raise AssertionError('unrelated failure')
""")
        cases = run_script(path, self.session)
        self.assertEqual([c['status'] for c in cases], ['failed', 'passed', 'failed'])
        self.assertEqual(cases[0]['outcome']['failed_intent']['intent'], 'Claim reward')
        self.assertEqual(cases[0]['outcome']['last_valid_result']['result'], {'stage': 'loaded'})
        self.assertEqual(cases[1]['outcome']['last_valid_result']['result'], {'score': 10})
        self.assertIsNone(cases[2]['outcome']['last_valid_result'])
        self.assertEqual(cases[2]['outcome']['failed_intent']['intent'], 'test_third')

    def test_labeled_json_failure_has_index_and_prior_result(self):
        plan = {'steps': [
            {'intent': 'Locate start', 'assert': {'position': [.5, .5]}},
            {'intent': 'Enter game', 'touch': 'missing'},
        ]}
        with self.assertRaises(TargetNotFoundError):
            execute(self.session, plan)
        out = self.session.outcome()
        self.assertEqual(out['last_valid_result']['intent'], 'Locate start')
        self.assertEqual(out['failed_intent']['intent'], 'Enter game')
        self.assertEqual(out['failed_intent']['details']['step'], 2)
        self.assertEqual(self.taps, [])

    def test_invalid_later_json_step_prevents_all_input_and_is_named(self):
        with self.assertRaises(ValueError):
            execute(self.session, {'steps': [{'touch': {'position': [.5, .5]}},
                                            {'intent': 'Bad step', 'unsupported': True}]})
        self.assertEqual(self.taps, [])
        self.assertEqual(self.session.failed_intent['intent'], 'Bad step')
        self.assertEqual(self.session.failed_intent['details']['phase'], 'validation')

    def test_json_wait_any_and_success_report(self):
        path = self.base / 'test_plan.json'
        path.write_text(json.dumps({'steps': [{'intent': 'Find state', 'wait_any': {
            'ready': {'position': [.5, .5]}, 'error': 'missing'}}]}))
        cases = run_script(path, self.session)
        self.assertEqual(cases[0]['outcome']['status'], 'succeeded')
        self.assertEqual(cases[0]['outcome']['last_valid_result']['result']['condition'], 'ready')

    def test_text_payload_is_omitted_from_automatic_details(self):
        self.session.text('secret')
        self.assertNotIn('secret', json.dumps(self.session.outcome()))

    def test_runner_serializes_failure_even_when_later_case_passes(self):
        path = self.base / 'test_suite.py'
        path.write_text("def test_first(session):\n    session.wait('missing')\n"
                        "def test_second(session):\n    return {'done': True}\n")
        args = argparse.Namespace(path=path, output=self.base/'results', serial=None,
                                  adb='adb', capture='auto', scrcpy_server=None,
                                  timeout=0, ocr_threads=2, ocr_max_side=None,
                                  ocr_backend='cpu', catalog=None, profile=None,
                                  failure_screenshot=False)
        with patch('swiftest.cli.Session', return_value=self.session):
            self.assertEqual(run_suite(args), 1)
        report = json.loads(next((self.base/'results').glob('*/report.json')).read_text())
        self.assertEqual(report[0]['outcome']['status'], 'failed')
        self.assertEqual(report[0]['outcome']['failed_intent']['intent'], 'wait')
        self.assertEqual(report[0]['cases'][1]['outcome']['status'], 'succeeded')
        self.assertNotIn('outcome', report[0]['metrics'])


if __name__ == '__main__':
    unittest.main()
