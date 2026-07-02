"""Isolated URDF splitting and timeline indexing for a completed rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.robocasa_4d import stitch_chunk_pointcloud_timelines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_root", type=Path)
    args = parser.parse_args()
    manifest = stitch_chunk_pointcloud_timelines(args.rollout_root)
    print(f"Saved stitched 4D timeline to {manifest}")


if __name__ == "__main__":
    main()
