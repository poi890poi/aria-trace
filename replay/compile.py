"""Command-line ReplayPackage compiler."""

import argparse
import json
from pathlib import Path

from .package import compile_replay_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stream", default="main")
    parser.add_argument("--route", required=True)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    args = parser.parse_args()
    manifest = compile_replay_package(
        args.session,
        args.output,
        args.stream,
        args.route,
        args.rate_hz,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

