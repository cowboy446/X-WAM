"""Project a URDF into saved RGB-D views and reconstruct one 4D chunk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.robocasa_4d import postprocess_chunk_urdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_root", type=Path)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--depth-tolerance", type=float, default=0.03)
    parser.add_argument("--dilation-pixels", type=int, default=2)
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.0,
        help="Minimum generated uint8 depth value used for predicted reconstruction",
    )
    args = parser.parse_args()
    postprocess_chunk_urdf(
        args.chunk_root,
        args.robot_urdf,
        args.depth_tolerance,
        args.dilation_pixels,
        args.depth_threshold,
    )


if __name__ == "__main__":
    main()
