"""Import an exported PLY sequence into Blender as an animated point cloud.

Run from a shell, for example:
  blender --python evaluation/blender_import_4d.py -- \
    /path/to/pointclouds /path/to/scene.blend

Each PLY is imported as one object. Visibility keyframes show exactly one frame
at a time. This is intentionally simple and works without Blender add-ons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("Usage: blender --python blender_import_4d.py -- POINTCLOUD_DIR [OUTPUT.blend]")
    pointcloud_dir = Path(argv[0]).resolve()
    output = Path(argv[1]).resolve() if len(argv) > 1 else pointcloud_dir / "pointcloud_sequence.blend"
    with (pointcloud_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        files = json.load(handle)["files"]

    objects = []
    for frame_index, filename in enumerate(files):
        bpy.ops.wm.ply_import(filepath=str(pointcloud_dir / filename))
        obj = bpy.context.active_object
        obj.name = f"pointcloud_{frame_index:04d}"
        objects.append(obj)

    for frame_index, obj in enumerate(objects):
        for keyframe in range(len(objects)):
            visible = keyframe == frame_index
            obj.hide_viewport = not visible
            obj.hide_render = not visible
            obj.keyframe_insert(data_path="hide_viewport", frame=keyframe + 1)
            obj.keyframe_insert(data_path="hide_render", frame=keyframe + 1)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(objects)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))


if __name__ == "__main__":
    main()
