"""Stable GLFW/Open3D legacy viewer for an X-WAM 4D timeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import open3d as o3d


class LegacyTimelineViewer:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sources = manifest["sources"]
        self.source_names = [name for name, frames in self.sources.items() if frames]
        self.source_index = 0
        self.subsets = ("full", "robot", "environment")
        self.subset_index = 2
        self.frame = 0
        self.playing = False
        self.last_advance = time.monotonic()

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window("X-WAM 4D timeline (legacy Open3D)", 1280, 800):
            raise RuntimeError("Open3D could not create a GLFW window")
        self.cloud = self._read_cloud()
        self.vis.add_geometry(self.cloud)
        render = self.vis.get_render_option()
        render.background_color = [0.04, 0.04, 0.04]
        render.point_size = 2.0

        self._bind(" ", self.toggle_playback)
        self._bind("J", self.step(-1))
        self._bind("L", self.step(1))
        self._bind("I", self.toggle_source)
        self._bind("P", self.cycle_subset)
        self._bind("1", self.select_subset(0))
        self._bind("2", self.select_subset(1))
        self._bind("3", self.select_subset(2))
        self._bind("A", lambda v: self.rotate(v, -30, 0))
        self._bind("D", lambda v: self.rotate(v, 30, 0))
        self._bind("W", lambda v: self.rotate(v, 0, -30))
        self._bind("S", lambda v: self.rotate(v, 0, 30))
        self._bind("Z", lambda v: self.zoom(v, 1.12))
        self._bind("X", lambda v: self.zoom(v, 0.88))
        self._bind("R", self.reset_view)
        self.vis.register_animation_callback(self.animate)
        self.print_status()

    @property
    def source(self):
        return self.source_names[self.source_index]

    @property
    def subset(self):
        return self.subsets[self.subset_index]

    @property
    def frames(self):
        return self.sources[self.source]

    def _bind(self, key, callback):
        self.vis.register_key_callback(ord(key), callback)

    def _read_cloud(self):
        entry = self.frames[self.frame]
        relative = entry["files"].get(self.subset) or entry["files"].get("full")
        cloud = o3d.io.read_point_cloud(str((self.manifest_path.parent / relative).resolve()))
        if not cloud.has_points():
            raise ValueError(f"Empty point cloud: {relative}")
        return cloud

    def update_cloud(self, vis, reset_view=False):
        new_cloud = self._read_cloud()
        self.cloud.points = new_cloud.points
        self.cloud.colors = new_cloud.colors
        self.cloud.normals = new_cloud.normals
        vis.update_geometry(self.cloud)
        if reset_view:
            vis.reset_view_point(True)
        self.print_status()
        return False

    def print_status(self):
        entry = self.frames[self.frame]
        state = "PLAY" if self.playing else "PAUSED"
        print(
            f"\r{state} | {self.source} | {self.subset} | "
            f"frame {self.frame + 1}/{len(self.frames)} | "
            f"t={entry['time_s']:.3f}s | points={len(self.cloud.points):,}    ",
            end="",
            flush=True,
        )

    def toggle_playback(self, _vis):
        self.playing = not self.playing
        self.last_advance = time.monotonic()
        self.print_status()
        return False

    def step(self, delta):
        def callback(vis):
            self.playing = False
            self.frame = (self.frame + delta) % len(self.frames)
            return self.update_cloud(vis)
        return callback

    def toggle_source(self, vis):
        old_time = self.frames[self.frame]["time_s"]
        self.playing = False
        self.source_index = (self.source_index + 1) % len(self.source_names)
        self.frame = min(range(len(self.frames)), key=lambda i: abs(self.frames[i]["time_s"] - old_time))
        return self.update_cloud(vis)

    def cycle_subset(self, vis):
        return self.select_subset((self.subset_index + 1) % len(self.subsets))(vis)

    def select_subset(self, index):
        def callback(vis):
            self.playing = False
            self.subset_index = index
            return self.update_cloud(vis)
        return callback

    @staticmethod
    def rotate(vis, x, y):
        vis.get_view_control().rotate(x, y)
        return False

    @staticmethod
    def zoom(vis, factor):
        vis.get_view_control().scale(factor)
        return False

    @staticmethod
    def reset_view(vis):
        vis.reset_view_point(True)
        return False

    def animate(self, vis):
        now = time.monotonic()
        if self.playing and now - self.last_advance >= 0.2:
            self.last_advance = now
            self.frame = (self.frame + 1) % len(self.frames)
            self.update_cloud(vis)
        return False

    def run(self):
        print("\nKeys: Space play | J/L frame | I source | P or 1/2/3 points | "
              "A/D/W/S rotate | Z/X zoom | R reset | Q close")
        self.vis.run()
        self.vis.destroy_window()
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="timeline manifest or rollout *_4d directory")
    args = parser.parse_args()
    manifest = args.path / "timeline" / "manifest.json" if args.path.is_dir() else args.path
    LegacyTimelineViewer(manifest).run()


if __name__ == "__main__":
    main()
