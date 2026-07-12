"""Package all predicted and ground-truth RGB files for one RoboCasa task."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


RGB_SOURCES = ("predicted", "groundtruth", "ground_truth")


def collect_rgb_files(task_dir: str | Path) -> list[Path]:
    """Collect files below every chunk's predicted/ground-truth ``rgb`` directory."""
    task_dir = Path(task_dir).expanduser().resolve()
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Task directory does not exist: {task_dir}")

    rollouts = sorted(path for path in task_dir.glob("*_4d") if path.is_dir())
    if not rollouts:
        raise FileNotFoundError(f"No *_4d rollout directories found under {task_dir}")

    files: set[Path] = set()
    for rollout in rollouts:
        for source in RGB_SOURCES:
            for rgb_dir in rollout.glob(f"chunks/*/{source}/rgb"):
                files.update(path.resolve() for path in rgb_dir.rglob("*") if path.is_file())

    if not files:
        sources = ", ".join(RGB_SOURCES)
        raise FileNotFoundError(
            f"No RGB files found under chunks/*/{{{sources}}}/rgb in {task_dir}"
        )
    return sorted(files)


def package_rgb_task(task_dir: str | Path, output: str | Path | None = None) -> Path:
    """Create a gzip-compressed tar archive rooted at the task directory."""
    task_dir = Path(task_dir).expanduser().resolve()
    if output is None:
        output = task_dir.parent / f"{task_dir.name}_rgb.tar.gz"
    output = Path(output).expanduser().resolve()
    if not output.name.endswith((".tar.gz", ".tgz")):
        raise ValueError("Output filename must end in .tar.gz or .tgz")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_rgb_files(task_dir)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(task_dir.parent), recursive=False)
    print(f"Packaged {len(files)} RGB files from {task_dir} to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package all predicted and ground-truth RGB files for one task."
    )
    parser.add_argument("task_dir", type=Path, help="Task directory containing *_4d rollouts")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output .tar.gz (default: beside the task directory)"
    )
    args = parser.parse_args()
    package_rgb_task(args.task_dir, args.output)


if __name__ == "__main__":
    main()
