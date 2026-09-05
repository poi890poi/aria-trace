"""Reevaluate saved production telemetry without rerunning inference or tracking."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchmarks.localization.build_workbench_report import build
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import read_rows, score
from benchmarks.localization.tracking_loss import calibrate_loss_tolerances


def rescore(root, *, loss_error_limit_px=None):
    implementation = [identity(Path(__file__).parent/name) for name in
                      ("run_workbench_replay.py", "tracking_loss.py", "build_workbench_report.py", "rescore_workbench.py")]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    paths = [p for p in sorted(root.glob("*/run*/report.json")) if not p.parent.parent.name.startswith("smoke")]
    references = {Path(json.loads(p.read_text())["reference"]) for p in paths}
    for path in paths:
        report = json.loads(path.read_text())
        for name in ("report.json", "scored_telemetry.jsonl"):
            original = path.parent/name
            backup = path.parent/("pre_tracking_loss_v1_"+name)
            if not backup.exists():
                shutil.copy2(original, backup)
        source_rows = read_rows(path.parent/"source_telemetry.jsonl")
        source = SimpleNamespace(rows=source_rows, frames=source_rows,
                                 origin=source_rows[0]["host_time_ns"]-source_rows[0]["session_time_ns"])
        raw = read_rows(Path(report["evidence"])/"telemetry.jsonl")
        enriched, metrics = score(raw, source, Path(report["reference"]),
                                  loss_error_limit_px=loss_error_limit_px,
                                  loss_calibration=calibrate_loss_tolerances(references, exclude_reference=report["reference"]))
        # Old names incorrectly described nonfresh-output episodes as loss.
        for name in ("loss_episodes", "longest_loss_s"):
            report.pop(name, None)
        report.update(metrics)
        report.update(evaluation_implementation=implementation,
                      evaluation_git_revision=revision, evaluation_git_status=status,
                      evaluation_note="Post-run reference tracking loss v1; original runtime telemetry, implementation identities and pre-v1 scores preserved. References are reused without inference.")
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (path.parent/"scored_telemetry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in enriched), encoding="utf-8")
    build(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--loss-error-limit-px", type=float)
    args = parser.parse_args()
    rescore(args.root, loss_error_limit_px=args.loss_error_limit_px)
