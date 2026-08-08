"""Run live-shaped incremental alignment on a recorded session."""

import argparse
import json
from pathlib import Path

from .session_observer import observe_session


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
    parser.add_argument("--min-margin", type=float, default=0.0)
    args = parser.parse_args()
    summary = observe_session(
        args.package,
        args.session,
        args.output,
        stream_id=args.stream,
        route_id=args.route,
        query_rate_hz=args.rate_hz,
        max_advance=args.max_advance,
        distance_threshold=args.distance_threshold,
        min_margin=args.min_margin,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
