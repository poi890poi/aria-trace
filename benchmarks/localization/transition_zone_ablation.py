"""Remove learned spatial transition gates, leaving visual confirmation intact.

Experimental only. Accepts template_cpu CLI arguments. Optional --reset-transition
combines with the already isolated confirmed-switch search reset.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from rig_runtime.services.calibration.minimap.transition import TransitionController
from benchmarks.localization import template_cpu


@contextmanager
def without_spatial_gates():
    original = TransitionController.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.transition_zones = ()
    with patch.object(TransitionController, "__init__", initialize):
        yield


def main():
    original_run = template_cpu.run
    def run(args):
        args.experiment["transition_zone_policy"] = "disabled-spatial-gates-visual-confirmation-unchanged"
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "transition_zone_source.py").write_text(Path(__file__).read_text(), encoding="utf-8")
        return original_run(args)
    with without_spatial_gates(), patch.object(template_cpu, "run", run):
        template_cpu.main()


if __name__ == "__main__":
    main()
