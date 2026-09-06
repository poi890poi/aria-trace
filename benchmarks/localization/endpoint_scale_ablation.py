"""Ablate intermediate layers while retaining image-verified endpoint XY handoff."""

from pathlib import Path
from unittest.mock import patch

from benchmarks.localization import precision_candidates, template_cpu
from benchmarks.localization.reference_cache import identity


def main():
    original_build = precision_candidates.build_pyramid
    original_run = template_cpu.run
    def build(layered):
        return original_build(layered, count=2)
    def run(args):
        args.experiment['endpoint_scale_ablation'] = {
            'layer_count': 2, 'role': 'remove-intermediate-scales-retain-endpoint-xy-handoff',
            'implementation': identity(__file__)}
        (args.output/'endpoint_scale_source.py').write_text(Path(__file__).read_text(),encoding='utf-8')
        return original_run(args)
    with patch.object(precision_candidates, 'build_pyramid', build), patch.object(template_cpu, 'run', run):
        precision_candidates.main()


if __name__ == '__main__':
    main()
