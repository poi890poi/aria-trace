"""Print a machine-readable summary of a recorded session."""

import argparse
import json
from pathlib import Path

from rig_runtime.adapters.filesystem.session import SessionReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = SessionReader(args.session).summary()
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
