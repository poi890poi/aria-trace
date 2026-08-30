"""Minimal profile-selected HIK adapter example.

Set ARIA_GAME_ID or replace the game_id value. The adapter performs no phone
operation while streaming.
"""

import argparse

import cv2
import hikcam


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id")
    parser.add_argument("--mode", choices=("full", "minimap", "dual"), default="full")
    parser.add_argument("--no-rectify", action="store_true")
    args = parser.parse_args()

    config = {
        "game_id": args.game_id,
        "mode": args.mode,
        "rectify": not args.no_rectify,
        "color_order": "BGR",
    }
    with hikcam.HikCamera(config=config) as camera:
        while True:
            frames = camera.get_frames() if args.mode == "dual" else {args.mode: camera.get_frame()}
            for name, image in frames.items():
                cv2.imshow("HIK {}".format(name), image)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
