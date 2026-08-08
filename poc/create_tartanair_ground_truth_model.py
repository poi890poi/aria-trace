"""Create a metric COLMAP text model from TartanAir camera poses."""

import argparse
from pathlib import Path

import numpy as np

try:
    from .evaluate_relocalization import matrix_to_quaternion, quaternion_to_matrix
    from .evaluate_tartanair_relocalization import OPENCV_TO_TARTAN_CAMERA
except ImportError:  # Direct script execution.
    from evaluate_relocalization import matrix_to_quaternion, quaternion_to_matrix
    from evaluate_tartanair_relocalization import OPENCV_TO_TARTAN_CAMERA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty output: {}".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)

    names = args.image_list.read_text(encoding="utf-8").splitlines()
    pose_cache = {}
    image_lines = []
    for image_id, name in enumerate(names, 1):
        trajectory = Path(name).parts[0]
        if trajectory not in pose_cache:
            pose_cache[trajectory] = np.loadtxt(
                str(args.data_root / trajectory / "pose_lcam_front.txt")
            )
        frame_index = int(Path(name).stem.split("_")[0])
        values = pose_cache[trajectory][frame_index]
        center = values[:3]
        qx, qy, qz, qw = values[3:]
        camera_to_world_tartan = quaternion_to_matrix(qw, qx, qy, qz)
        camera_to_world_opencv = camera_to_world_tartan @ OPENCV_TO_TARTAN_CAMERA
        world_to_camera = camera_to_world_opencv.T
        translation = -world_to_camera @ center
        quaternion = matrix_to_quaternion(world_to_camera)
        pose = [image_id] + quaternion.tolist() + translation.tolist() + [1, name]
        image_lines.append(" ".join(str(value) for value in pose))
        image_lines.append("")

    (args.output / "cameras.txt").write_text(
        "# Camera list\n# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        "1 SIMPLE_PINHOLE 640 640 320 320 320\n",
        encoding="utf-8",
    )
    (args.output / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (args.output / "points3D.txt").write_text("# Empty before triangulation\n", encoding="utf-8")
    print("wrote {} registered poses".format(len(names)))


if __name__ == "__main__":
    main()
