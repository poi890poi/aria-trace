"""Benchmark completion must reject missing evidence and unintended inputs."""
import unittest
from benchmarks.android.full_comparison_trial import validate


class ComparisonContractTests(unittest.TestCase):
    def inputs(self):
        return [dict(reason=r,sent=True) for r in ("minimap","waypoint","teleport")]

    def test_valid_warm_testkit(self):
        validate(dict(status="teleport_completed"),"testkit","warm",self.inputs(),
                 [dict(state=s) for s in ("map","loading","world","world")],[])

    def test_no_loading_cannot_be_success(self):
        with self.assertRaises(AssertionError):
            validate(dict(status="teleport_completed"),"testkit","warm",self.inputs(),
                     [dict(state="world")]*2,[])

    def test_recovery_or_duplicate_tap_is_rejected(self):
        for inputs in (self.inputs()+[dict(reason="teleport",sent=True)],
                       [dict(reason=r["reason"],sent=False) for r in self.inputs()]):
            with self.assertRaises(AssertionError):
                validate(dict(status="teleport_completed"),"testkit","warm",inputs,
                         [dict(state=s) for s in ("loading","world","world")],[])

    def test_airtest_requires_ordered_postconditions(self):
        names=["world_ready","map_ready","waypoint_selected","loading","arrival"]
        validate(dict(status="teleport_completed"),"airtest","warm",self.inputs(),[],
                 [dict(event=n) for n in names])
        with self.assertRaises(AssertionError):
            validate(dict(status="teleport_completed"),"airtest","warm",self.inputs(),[],
                     [dict(event=n) for n in reversed(names)])


if __name__=="__main__":
    unittest.main()
