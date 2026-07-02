"""Isolated URDF splitting and timeline indexing for a completed rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.robocasa_4d import postprocess_rollout_urdf, stitch_chunk_pointcloud_timelines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_root", type=Path)
    parser.add_argument("--robot-urdf", type=Path)
    parser.add_argument("--robot-padding", type=float, default=0.008)
    args = parser.parse_args()
    if args.robot_urdf is not None:
        postprocess_rollout_urdf(args.rollout_root, args.robot_urdf, args.robot_padding)
        print(f"Saved offline URDF robot/environment split under {args.rollout_root}")
    manifest = stitch_chunk_pointcloud_timelines(args.rollout_root)
    print(f"Saved stitched 4D timeline to {manifest}")


if __name__ == "__main__":
    main()
