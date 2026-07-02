import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.robocasa_4d import (
    backproject_rgbd,
    fit_predicted_depth_to_metric,
    points_inside_robot,
    save_pointcloud_sequence,
    stitch_chunk_pointcloud_timelines,
    transform_intrinsics_for_resize_crop,
    validate_4d_shapes,
    write_binary_ply,
)


class RoboCasa4DTest(unittest.TestCase):
    def test_multiview_4d_shape_validation(self):
        T, V, H, W = 33, 3, 8, 10
        rgb = np.zeros((T, V, H, W, 3), dtype=np.uint8)
        depth = np.ones((T, V, H, W), dtype=np.float32)
        K = np.repeat(np.eye(3)[None, None], T * V, axis=0).reshape(T, V, 3, 3)
        poses = np.repeat(np.eye(4)[None, None], T * V, axis=0).reshape(T, V, 4, 4)
        validate_4d_shapes(
            rgb, depth, K, poses, ["left", "right", "wrist"], "test",
            action_offsets=np.arange(T), executed_action_count=32,
        )
        with self.assertRaisesRegex(ValueError, "depth must match"):
            validate_4d_shapes(rgb, depth[:-1], K, poses, ["left", "right", "wrist"], "test")

    def test_dense_action_timeline_has_initial_plus_post_action_states(self):
        action_count = 32
        offsets = np.arange(action_count + 1)
        timestamps = offsets / 20.0
        self.assertEqual(len(offsets), 33)
        self.assertEqual(len(np.diff(offsets)), action_count)
        np.testing.assert_allclose(np.diff(timestamps), 0.05)

    def test_backprojection_identity_camera(self):
        rgb = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        depth = np.ones((1, 2, 2), dtype=np.float32)
        K = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        poses = np.eye(4)[None]
        xyz, colors, view_ids = backproject_rgbd(rgb, depth, K, poses, stride=1)
        np.testing.assert_allclose(
            xyz,
            np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32),
        )
        self.assertTrue(np.all(colors[:, 0] == 255))
        self.assertTrue(np.all(view_ids == 0))

    def test_inverse_depth_calibration(self):
        depth = np.linspace(0.5, 2.0, 400, dtype=np.float32).reshape(1, 20, 20)
        raw0 = (1.0 / depth - 0.25) / 0.01
        raw = np.stack([raw0, raw0 * 0.9], axis=0)
        metric, metadata = fit_predicted_depth_to_metric(raw, depth, representation="inverse")
        np.testing.assert_allclose(metric[0, 0], depth[0], rtol=1e-4, atol=1e-4)
        self.assertEqual(metadata["representation"], "inverse_affine")

    def test_intrinsics_resize_crop(self):
        K = np.array([[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]])
        transformed = transform_intrinsics_for_resize_crop(K, (100, 100), (200, 200), 1.0)
        np.testing.assert_allclose(transformed[0, 0, 0], 200.0)
        np.testing.assert_allclose(transformed[0, 1, 1], 200.0)
        np.testing.assert_allclose(transformed[0, :2, 2], [100.0, 100.0])

    def test_ply_and_sequence_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_binary_ply(root / "one.ply", np.zeros((2, 3), np.float32), np.zeros((2, 3), np.uint8))
            self.assertTrue((root / "one.ply").read_bytes().startswith(b"ply\nformat binary_little_endian"))

            rgb = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
            depth = np.ones((1, 1, 2, 2), dtype=np.float32)
            K = np.eye(3)[None]
            poses = np.eye(4)[None]
            save_pointcloud_sequence(
                root / "sequence", rgb, depth, K, poses, stride=1,
                timestamps_s=np.array([0.0]), action_offsets=np.array([0]),
            )
            manifest = json.loads((root / "sequence" / "manifest.json").read_text())
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["timestamps_s"], [0.0])
            self.assertEqual(manifest["action_offsets"], [0])
            self.assertTrue((root / "sequence" / manifest["files"][0]).exists())

    def test_robot_point_split_box(self):
        geom = {
            "type": 6,
            "size": np.array([0.5, 0.5, 0.5]),
            "T_base_from_geom": np.eye(4),
        }
        xyz = np.array([[0.0, 0.0, 0.0], [0.51, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_array_equal(
            points_inside_robot(xyz, [geom], padding=0.02), [True, True, False]
        )

    def test_sequence_writes_robot_and_environment_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.zeros((1, 1, 1, 2, 3), dtype=np.uint8)
            depth = np.ones((1, 1, 1, 2), dtype=np.float32)
            K = np.eye(3)[None]
            poses = np.eye(4)[None]
            geom = {"type": 6, "size": np.array([0.1, 0.1, 1.1]), "T_base_from_geom": np.eye(4)}
            save_pointcloud_sequence(
                root, rgb, depth, K, poses, stride=1, robot_geoms_t=[[geom]], robot_padding=0.0
            )
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertTrue((root / manifest["robot_files"][0]).exists())
            self.assertTrue((root / manifest["environment_files"][0]).exists())
            self.assertEqual(len(np.load(root / "robot/frame_0000.npz")["xyz"]), 1)
            self.assertEqual(len(np.load(root / "environment/frame_0000.npz")["xyz"]), 1)

    def test_stitch_chunk_timelines_orders_and_deduplicates_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "0_0_4d"
            for start in (0, 4):
                chunk = rollout / "chunks" / f"step_{start:06d}"
                (chunk / "predicted/pointclouds/environment").mkdir(parents=True)
                (chunk / "ground_truth/pointclouds/environment").mkdir(parents=True)
                (chunk / "metadata.json").write_text(json.dumps({
                    "chunk_start_step": start, "action_fps": 2.0,
                }))
                pc_manifest = {
                    "frame_count": 3,
                    "timestamps_s": [0.0, 1.0, 2.0],
                    "files": [f"frame_{i:04d}.ply" for i in range(3)],
                    "robot_files": [],
                    "environment_files": [f"environment/frame_{i:04d}.ply" for i in range(3)],
                }
                for source in ("predicted", "ground_truth"):
                    (chunk / source / "pointclouds/manifest.json").write_text(json.dumps(pc_manifest))
            manifest_path = stitch_chunk_pointcloud_timelines(rollout)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["frame_counts"], {"imagined": 5, "simulation": 5})
            self.assertEqual(
                [frame["time_s"] for frame in manifest["sources"]["imagined"]],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            )


if __name__ == "__main__":
    unittest.main()
