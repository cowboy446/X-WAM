"""Project one saved RGB-D frame's URDF mask in an isolated process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.robocasa_4d import load_urdf_visual_triangles, project_urdf_robot_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("qpos", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--depth-tolerance", type=float, default=0.03)
    parser.add_argument("--dilation-pixels", type=int, default=2)
    args = parser.parse_args()

    with args.qpos.open("r", encoding="utf-8") as handle:
        qpos = json.load(handle)
    with np.load(args.input) as archive:
        depth = archive["depth"]
        K = archive["K"]
        poses = archive["poses"]
    triangles = load_urdf_visual_triangles(args.robot_urdf, [qpos])
    mask = project_urdf_robot_masks(
        triangles,
        depth,
        K,
        poses,
        args.depth_tolerance,
        args.dilation_pixels,
    )
    np.savez_compressed(args.output, mask=mask)


if __name__ == "__main__":
    main()
