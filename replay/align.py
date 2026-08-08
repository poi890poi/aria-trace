"""Command-line two-session replay alignment."""

import argparse
import json
from pathlib import Path

from .alignment import align_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stream")
    parser.add_argument("--route")
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--max-advance", type=int, default=4)
    parser.add_argument("--distance-threshold", type=float, default=0.45)
    args = parser.parse_args()
    summary = align_session(
        args.package,
        args.session,
        args.output,
        stream_id=args.stream,
        route_id=args.route,
        query_rate_hz=args.rate_hz,
        max_advance=args.max_advance,
        distance_threshold=args.distance_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

