"""Package a RoboCasa task's continuous 4D visualization files."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.robocasa_4d import stitch_chunk_pointcloud_timelines


def _rollout_directories(task_dir: Path) -> list[Path]:
    rollouts = sorted(path for path in task_dir.glob("*_4d") if path.is_dir())
    if not rollouts:
        raise FileNotFoundError(f"No *_4d rollout directories found under {task_dir}")
    return rollouts


def collect_visualization_files(task_dir: str | Path) -> list[Path]:
    """Collect all JSON and timeline-referenced PLY files under one task."""
    task_dir = Path(task_dir).expanduser().resolve()
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Task directory does not exist: {task_dir}")

    files = {path.resolve() for path in task_dir.rglob("*.json") if path.is_file()}
    for rollout in _rollout_directories(task_dir):
        manifest_path = rollout / "timeline" / "manifest.json"
        if not manifest_path.exists():
            manifest_path = stitch_chunk_pointcloud_timelines(rollout)
        manifest_path = manifest_path.resolve()
        files.add(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for frames in manifest.get("sources", {}).values():
            for frame in frames:
                for relative in frame.get("files", {}).values():
                    if relative is None:
                        continue
                    ply = (manifest_path.parent / relative).resolve()
                    try:
                        ply.relative_to(task_dir)
                    except ValueError as exc:
                        raise ValueError(
                            f"Timeline references a file outside the task directory: {ply}"
                        ) from exc
                    if not ply.is_file():
                        raise FileNotFoundError(f"Timeline-referenced PLY is missing: {ply}")
                    files.add(ply)
    return sorted(files)


def package_4d_task(task_dir: str | Path, output: str | Path | None = None) -> Path:
    """Create a .tar.gz while preserving paths below the task's parent."""
    task_dir = Path(task_dir).expanduser().resolve()
    if output is None:
        output = task_dir.parent / f"{task_dir.name}_4d_visualization.tar.gz"
    output = Path(output).expanduser().resolve()
    if not output.name.endswith((".tar.gz", ".tgz")):
        raise ValueError("Output filename must end in .tar.gz or .tgz")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_visualization_files(task_dir)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(task_dir.parent), recursive=False)
    print(f"Packaged {len(files)} files from {task_dir} to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package all continuous 4D visualization PLY/JSON files for one task."
    )
    parser.add_argument("task_dir", type=Path, help="Task directory containing *_4d rollouts")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output .tar.gz (default: beside the task directory)"
    )
    args = parser.parse_args()
    package_4d_task(args.task_dir, args.output)


if __name__ == "__main__":
    main()
