#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 TASK_DIR [OUTPUT.tar.gz]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_dir="$1"

if [[ $# -eq 2 ]]; then
    exec python "$repo_root/evaluation/package_rgb_task.py" "$task_dir" --output "$2"
fi
exec python "$repo_root/evaluation/package_rgb_task.py" "$task_dir"
