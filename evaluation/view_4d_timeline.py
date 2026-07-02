"""Open an interactive Open3D slider for a stitched X-WAM 4D rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import open3d as o3d
from open3d.visualization import gui, rendering

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.robocasa_4d import stitch_chunk_pointcloud_timelines


class TimelineViewer:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.sources = self.manifest["sources"]
        self.source = next((name for name, frames in self.sources.items() if frames), None)
        if self.source is None:
            raise ValueError("Timeline contains no point-cloud frames")
        self.subset = "environment"
        self.frame = 0

        app = gui.Application.instance
        self.window = app.create_window("X-WAM 4D timeline", 1280, 800)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.04, 0.04, 0.04, 1.0])

        self.source_box = gui.Combobox()
        source_names = list(self.sources)
        for name in source_names:
            self.source_box.add_item(name)
        self.source_box.selected_index = source_names.index(self.source)
        self.source_box.set_on_selection_changed(self._on_source)

        self.subset_box = gui.Combobox()
        for name in ("full", "robot", "environment"):
            self.subset_box.add_item(name)
        self.subset_box.selected_index = 2
        self.subset_box.set_on_selection_changed(self._on_subset)

        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_on_value_changed(self._on_frame)
        self.label = gui.Label("")

        controls = gui.Horiz(8)
        controls.add_child(gui.Label("source"))
        controls.add_child(self.source_box)
        controls.add_child(gui.Label("points"))
        controls.add_child(self.subset_box)
        controls.add_stretch()
        controls.add_child(self.label)
        layout = gui.Vert(8, gui.Margins(8, 8, 8, 8))
        layout.add_child(self.scene)
        layout.add_child(self.slider)
        layout.add_child(controls)
        self.window.add_child(layout)
        self._reset_slider()
        self._show_frame(reset_camera=True)

    def _frames(self):
        return self.sources[self.source]

    def _reset_slider(self):
        count = len(self._frames())
        self.frame = min(self.frame, max(count - 1, 0))
        self.slider.set_limits(0, max(count - 1, 1))
        self.slider.int_value = self.frame

    def _on_source(self, text, _index):
        old_frames = self._frames()
        old_time = old_frames[self.frame]["time_s"] if old_frames else 0.0
        self.source = text
        frames = self._frames()
        self.frame = min(range(len(frames)), key=lambda i: abs(frames[i]["time_s"] - old_time)) if frames else 0
        self._reset_slider()
        self._show_frame(reset_camera=False)

    def _on_subset(self, text, _index):
        self.subset = text
        self._show_frame(reset_camera=False)

    def _on_frame(self, value):
        self.frame = int(round(value))
        self._show_frame(reset_camera=False)

    def _show_frame(self, reset_camera: bool):
        frames = self._frames()
        if not frames:
            self.label.text = f"{self.source}: no frames"
            return
        entry = frames[self.frame]
        relative = entry["files"].get(self.subset) or entry["files"].get("full")
        if relative is None:
            self.label.text = f"frame {self.frame}: no {self.subset} file"
            return
        path = (self.manifest_path.parent / relative).resolve()
        cloud = o3d.io.read_point_cloud(str(path))
        self.scene.scene.remove_geometry("pointcloud")
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2.0
        self.scene.scene.add_geometry("pointcloud", cloud, material)
        if reset_camera and cloud.has_points():
            bounds = cloud.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60.0, bounds, bounds.get_center())
        self.label.text = (
            f"frame {self.frame + 1}/{len(frames)}   t={entry['time_s']:.3f}s   "
            f"points={len(cloud.points)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="timeline manifest or rollout *_4d directory")
    args = parser.parse_args()
    if args.path.is_dir():
        manifest = args.path / "timeline" / "manifest.json"
        if not manifest.exists():
            manifest = stitch_chunk_pointcloud_timelines(args.path)
    else:
        manifest = args.path
    gui.Application.instance.initialize()
    TimelineViewer(manifest)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
