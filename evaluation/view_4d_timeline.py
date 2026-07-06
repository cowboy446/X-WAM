"""Compatibility entry point for the stable GLFW/Open3D timeline viewer."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.view_4d_timeline_legacy import main


if __name__ == "__main__":
    main()
