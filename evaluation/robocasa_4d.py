"""RGB-D capture and 4D point-cloud export helpers for RoboCasa evaluation.

The module deliberately depends only on NumPy and imageio. RoboSuite imports are
kept inside :func:`capture_rgbd` so saved captures can be reconstructed on a
lightweight machine without installing MuJoCo or RoboCasa.

Coordinate convention
---------------------
``T_base_from_camera`` maps homogeneous camera coordinates to the robot base
frame. Depth is measured along the camera optical z axis in metres. The camera
pose returned by RoboSuite's camera utilities already includes its OpenGL to
computer-vision convention conversion; do not add another axis flip here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def validate_4d_shapes(
    rgb,
    depth,
    K,
    poses,
    camera_names,
    label,
    action_offsets=None,
    executed_action_count=None,
) -> None:
    """Fail early when time/view/camera calibration axes are not aligned."""
    rgb, depth, K, poses = map(np.asarray, (rgb, depth, K, poses))
    if rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError(f"{label} rgb must be [T,V,H,W,3], got {rgb.shape}")
    if depth.shape != rgb.shape[:-1]:
        raise ValueError(f"{label} depth must match rgb [T,V,H,W], got {depth.shape} vs {rgb.shape}")
    time_count, view_count = rgb.shape[:2]
    if view_count != len(camera_names):
        raise ValueError(f"{label} has {view_count} views but {len(camera_names)} camera names")
    if K.shape != (time_count, view_count, 3, 3):
        raise ValueError(f"{label} K must be [T,V,3,3], got {K.shape}")
    if poses.shape != (time_count, view_count, 4, 4):
        raise ValueError(f"{label} poses must be [T,V,4,4], got {poses.shape}")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(poses)):
        raise ValueError(f"{label} K/poses contain non-finite values")
    if action_offsets is not None:
        offsets = np.asarray(action_offsets)
        if offsets.shape != (time_count,):
            raise ValueError(f"{label} action_offsets must have one entry per frame, got {offsets.shape}")
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError(f"{label} action_offsets must start at 0 and increase strictly: {offsets}")
        if executed_action_count is not None and offsets[-1] != executed_action_count:
            raise ValueError(
                f"{label} final action offset {offsets[-1]} does not match "
                f"executed action count {executed_action_count}"
            )


def transform_intrinsics_for_resize_crop(
    K_v33: np.ndarray,
    input_hw: tuple[int, int],
    output_hw: tuple[int, int],
    crop_ratio: float,
) -> np.ndarray:
    """Match ``policy_server.resize_and_center_crop_tensor`` geometrically."""
    K = np.asarray(K_v33, dtype=np.float64).copy()
    in_h, in_w = input_hw
    out_h, out_w = output_hw
    K[:, 0, :] *= out_w / in_w
    K[:, 1, :] *= out_h / in_h
    crop_h, crop_w = int(out_h * crop_ratio), int(out_w * crop_ratio)
    top, left = (out_h - crop_h) // 2, (out_w - crop_w) // 2
    K[:, 0, 2] -= left
    K[:, 1, 2] -= top
    K[:, 0, :] *= out_w / crop_w
    K[:, 1, :] *= out_h / crop_h
    return K


def resize_center_crop_nearest(
    array_vhw: np.ndarray,
    output_hw: tuple[int, int],
    crop_ratio: float,
) -> np.ndarray:
    """NumPy nearest-neighbour equivalent used for depth calibration only."""
    array = np.asarray(array_vhw)
    out_h, out_w = output_hw
    in_h, in_w = array.shape[-2:]
    yi = np.minimum((np.arange(out_h) * in_h / out_h).astype(int), in_h - 1)
    xi = np.minimum((np.arange(out_w) * in_w / out_w).astype(int), in_w - 1)
    resized = array[:, yi[:, None], xi[None, :]]
    crop_h, crop_w = int(out_h * crop_ratio), int(out_w * crop_ratio)
    top, left = (out_h - crop_h) // 2, (out_w - crop_w) // 2
    cropped = resized[:, top : top + crop_h, left : left + crop_w]
    yi2 = np.minimum((np.arange(out_h) * crop_h / out_h).astype(int), crop_h - 1)
    xi2 = np.minimum((np.arange(out_w) * crop_w / out_w).astype(int), crop_w - 1)
    return cropped[:, yi2[:, None], xi2[None, :]]


def capture_rgbd(env, camera_names: list[str], base2world: np.ndarray, height: int, width: int) -> dict[str, np.ndarray]:
    """Render RGB-D and record calibrated camera matrices for every view."""
    try:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
    except ImportError as exc:  # pragma: no cover - only available in server env
        raise ImportError("RoboSuite camera utilities are required while capturing simulation frames") from exc

    rgbs, depths, intrinsics, world_from_cameras, base_from_cameras = [], [], [], [], []
    world_from_base = np.asarray(base2world, dtype=np.float64)
    base_from_world = np.linalg.inv(world_from_base)
    for camera_name in camera_names:
        rendered = env.sim.render(
            height=height,
            width=width,
            camera_name=camera_name,
            depth=True,
            segmentation=False,
        )
        if not isinstance(rendered, tuple) or len(rendered) != 2:
            raise RuntimeError(f"Expected RGB/depth tuple from camera {camera_name}, got {type(rendered)!r}")
        rgb, depth_buffer = rendered
        rgb = np.asarray(rgb)[::-1].copy()
        depth_buffer = np.asarray(depth_buffer)[::-1].copy()
        depth_m = np.asarray(get_real_depth_map(env.sim, depth_buffer), dtype=np.float32)
        if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
            depth_m = depth_m[..., 0]

        K = np.asarray(get_camera_intrinsic_matrix(env.sim, camera_name, height, width), dtype=np.float64)
        T_world_camera = np.asarray(get_camera_extrinsic_matrix(env.sim, camera_name), dtype=np.float64)
        rgbs.append(rgb)
        depths.append(depth_m)
        intrinsics.append(K)
        world_from_cameras.append(T_world_camera)
        base_from_cameras.append(base_from_world @ T_world_camera)

    return {
        "rgb": np.stack(rgbs).astype(np.uint8),
        "depth_m": np.stack(depths).astype(np.float32),
        "K": np.stack(intrinsics),
        "T_world_from_camera": np.stack(world_from_cameras),
        "T_base_from_camera": np.stack(base_from_cameras),
        "T_world_from_base": world_from_base,
    }


def capture_robot_state(env) -> dict[str, Any]:
    """Capture only joint state; geometry construction is deferred offline."""
    sim = env.sim
    data = sim.data
    robot = env.robots[0]
    indexes = np.asarray(getattr(robot, "_ref_joint_pos_indexes", []), dtype=np.int64)
    robot_qpos = np.asarray(data.qpos[indexes], dtype=np.float64).copy() if indexes.size else np.empty(0)
    joint_names = list(getattr(robot.robot_model, "joints", []))
    # Match PointWorld's Franka URDF contract. RoboSuite exposes the seven arm
    # joints in robot_qpos; the Robotiq controller exposes its driving joint.
    urdf_qpos = {f"panda_joint{i + 1}": float(value) for i, value in enumerate(robot_qpos[:7])}
    controller = robot.composite_controller
    grippers = list(controller.grippers.keys())
    if grippers:
        gripper_controller = controller.part_controllers[grippers[0]]
        urdf_qpos["finger_joint"] = float(np.asarray(gripper_controller.joint_pos).reshape(-1)[0])
    return {
        "sim_qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
        "robot_qpos": robot_qpos,
        "robot_joint_names": joint_names,
        "urdf_qpos": urdf_qpos,
    }


def load_urdf_visual_triangles(
    urdf_path: str | Path,
    urdf_qpos_t: list[dict[str, float]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return one base-frame visual triangle soup per robot state."""
    # urdfpy 0.0.22 still references NumPy aliases removed in recent releases.
    for alias, value in (("float", float), ("int", int), ("bool", bool)):
        if alias not in np.__dict__:
            setattr(np, alias, value)
    try:
        import urdfpy
    except ImportError as exc:
        raise ImportError(
            "URDF mask projection requires urdfpy/trimesh; install the PointWorld "
            "URDF runtime requirements first"
        ) from exc

    path = Path(urdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    robot = urdfpy.URDF.load(str(path))
    actuated = {joint.name for joint in robot.actuated_joints}
    sequence: list[tuple[np.ndarray, np.ndarray]] = []
    for saved_cfg in urdf_qpos_t:
        cfg = {name: float(value) for name, value in saved_cfg.items() if name in actuated}
        vertices, triangles, vertex_offset = [], [], 0
        for link, base_from_link in robot.link_fk(cfg=cfg).items():
            for visual in link.visuals:
                mesh_spec = visual.geometry.mesh
                if mesh_spec is None:
                    continue
                transform = np.asarray(base_from_link) @ np.asarray(visual.origin)
                scale = mesh_spec.scale
                for mesh in getattr(mesh_spec, "_meshes", None) or []:
                    local_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
                    if scale is not None:
                        local_vertices *= np.asarray(scale, dtype=np.float64)
                    homogeneous = np.concatenate(
                        [local_vertices, np.ones((len(local_vertices), 1))], axis=1
                    )
                    world_vertices = (homogeneous @ transform.T)[:, :3]
                    faces = np.asarray(mesh.faces, dtype=np.int32)
                    vertices.append(world_vertices.astype(np.float32))
                    triangles.append(faces + vertex_offset)
                    vertex_offset += len(world_vertices)
        if not vertices:
            raise ValueError(f"URDF has no visual triangle meshes: {path}")
        sequence.append((np.concatenate(vertices), np.concatenate(triangles)))
    return sequence


def fit_predicted_depth_to_metric(
    predicted_raw_tvhws: np.ndarray,
    reference_depth_vhw: np.ndarray,
    representation: str = "inverse",
    min_depth: float = 0.05,
    max_depth: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate generated depth using its frame-0 overlap with measured depth.

    The released depth videos do not carry a metric scale sidecar. For the
    default ``inverse`` representation we robustly fit
    ``1 / z_metric = scale * raw + shift`` per view using frame 0. ``metric``
    treats raw values as metres and is mainly useful for future checkpoints.
    Raw predictions must always be saved alongside this calibrated result.
    """
    raw = np.asarray(predicted_raw_tvhws, dtype=np.float32)
    reference = np.asarray(reference_depth_vhw, dtype=np.float32)
    if raw.ndim != 4:
        raise ValueError(f"predicted_raw must be [T,V,H,W], got {raw.shape}")
    if reference.shape != raw.shape[1:]:
        raise ValueError(f"reference depth shape {reference.shape} does not match {raw.shape[1:]}")
    if representation == "metric":
        metric = np.clip(raw, min_depth, max_depth)
        return metric, {"representation": "metric", "per_view": []}
    if representation != "inverse":
        raise ValueError(f"Unknown predicted depth representation: {representation}")

    metric = np.empty_like(raw, dtype=np.float32)
    calibration = []
    for view in range(raw.shape[1]):
        x = raw[0, view].reshape(-1)
        z = reference[view].reshape(-1)
        valid = np.isfinite(x) & np.isfinite(z) & (z > min_depth) & (z < max_depth)
        if valid.sum() < 100:
            raise ValueError(f"Not enough valid depth pixels to calibrate view {view}: {valid.sum()}")
        xv = x[valid]
        yv = 1.0 / z[valid]

        # Trim both tails before least-squares so object edges and generated
        # outliers do not set the global metric scale.
        x_lo, x_hi = np.quantile(xv, [0.02, 0.98])
        y_lo, y_hi = np.quantile(yv, [0.02, 0.98])
        keep = (xv >= x_lo) & (xv <= x_hi) & (yv >= y_lo) & (yv <= y_hi)
        A = np.stack([xv[keep], np.ones(keep.sum(), dtype=np.float32)], axis=1)
        scale, shift = np.linalg.lstsq(A, yv[keep], rcond=None)[0]
        inv_depth = scale * raw[:, view] + shift
        depth = 1.0 / np.maximum(inv_depth, 1.0 / max_depth)
        metric[:, view] = np.clip(depth, min_depth, max_depth)
        calibration.append({
            "view": view,
            "scale": float(scale),
            "shift": float(shift),
            "valid_pixels": int(keep.sum()),
        })
    return metric, {"representation": "inverse_affine", "per_view": calibration}


def backproject_rgbd(
    rgb_vhwc: np.ndarray,
    depth_vhw: np.ndarray,
    K_v33: np.ndarray,
    T_base_camera_v44: np.ndarray,
    stride: int = 2,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
    pixel_mask_vhw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse calibrated multi-view RGB-D into a base-frame colored point cloud."""
    rgb = np.asarray(rgb_vhwc)
    depth = np.asarray(depth_vhw)
    K = np.asarray(K_v33)
    poses = np.asarray(T_base_camera_v44)
    if rgb.shape[:3] != depth.shape or rgb.shape[-1] != 3:
        raise ValueError(f"RGB/depth shapes are incompatible: {rgb.shape}, {depth.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    xyz_chunks, color_chunks, view_chunks = [], [], []
    for view in range(rgb.shape[0]):
        h, w = depth[view].shape
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[view, ::stride, ::stride]
        valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
        if pixel_mask_vhw is not None:
            valid &= np.asarray(pixel_mask_vhw[view], dtype=bool)[::stride, ::stride]
        fx, fy = K[view, 0, 0], K[view, 1, 1]
        cx, cy = K[view, 0, 2], K[view, 1, 2]
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        camera_points = np.stack([x, y, z, np.ones_like(z)], axis=-1)[valid]
        base_points = camera_points @ poses[view].T
        xyz_chunks.append(base_points[:, :3].astype(np.float32))
        color_chunks.append(rgb[view, ::stride, ::stride][valid].astype(np.uint8))
        view_chunks.append(np.full(valid.sum(), view, dtype=np.uint8))
    return np.concatenate(xyz_chunks), np.concatenate(color_chunks), np.concatenate(view_chunks)


def write_binary_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a Blender-importable binary little-endian colored PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError(f"Expected xyz/rgb [N,3], got {xyz.shape}/{rgb.shape}")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def save_rgb_video(path: str | Path, frames_thwc: np.ndarray, fps: float) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, np.asarray(frames_thwc, dtype=np.uint8), fps=fps, codec="libx264", quality=8)


def save_depth_sequence(directory: str | Path, depth_thw: np.ndarray, fps: float, raw: bool = False) -> None:
    """Save lossless frames plus a viewable MP4; metric frames use uint16 millimetres."""
    import imageio.v2 as imageio

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_thw)
    previews = []
    for index, frame in enumerate(depth):
        if raw:
            encoded = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            encoded = np.clip(frame * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        imageio.imwrite(directory / f"frame_{index:04d}.png", encoded)
        finite = frame[np.isfinite(frame)]
        if finite.size:
            lo, hi = np.quantile(finite, [0.02, 0.98])
            preview = np.clip((frame - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255).astype(np.uint8)
        else:
            preview = np.zeros(frame.shape, dtype=np.uint8)
        previews.append(np.repeat(preview[..., None], 3, axis=-1))
    save_rgb_video(directory / "preview.mp4", np.stack(previews), fps)


def save_pointcloud_sequence(
    directory: str | Path,
    rgb_t_vhwc: np.ndarray,
    depth_t_vhw: np.ndarray,
    K_t_v33: np.ndarray,
    poses_t_v44: np.ndarray,
    stride: int,
    timestamps_s: np.ndarray | None = None,
    action_offsets: np.ndarray | None = None,
    robot_masks_t_vhw: np.ndarray | None = None,
) -> None:
    """Save one fused PLY and compressed NPZ per time step."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_count = len(rgb_t_vhwc)
    if K_t_v33.ndim == 3:
        K_t_v33 = np.repeat(K_t_v33[None], frame_count, axis=0)
    if poses_t_v44.ndim == 3:
        poses_t_v44 = np.repeat(poses_t_v44[None], frame_count, axis=0)
    if timestamps_s is not None and len(timestamps_s) != frame_count:
        raise ValueError("timestamps_s length must match point-cloud frame count")
    if action_offsets is not None and len(action_offsets) != frame_count:
        raise ValueError("action_offsets length must match point-cloud frame count")
    if robot_masks_t_vhw is not None and len(robot_masks_t_vhw) != frame_count:
        raise ValueError("robot_masks_t_vhw length must match point-cloud frame count")
    manifest = {
        "frame_count": frame_count,
        "stride": stride,
        "timestamps_s": None if timestamps_s is None else np.asarray(timestamps_s).tolist(),
        "action_offsets": None if action_offsets is None else np.asarray(action_offsets).tolist(),
        "files": [],
        "robot_files": [],
        "environment_files": [],
        "robot_mask_method": "urdf_visual_mesh_depth_projection" if robot_masks_t_vhw is not None else None,
    }
    for time_index in range(frame_count):
        xyz, rgb, view_id = backproject_rgbd(
            rgb_t_vhwc[time_index], depth_t_vhw[time_index], K_t_v33[time_index], poses_t_v44[time_index], stride
        )
        stem = f"frame_{time_index:04d}"
        write_binary_ply(directory / f"{stem}.ply", xyz, rgb)
        np.savez_compressed(directory / f"{stem}.npz", xyz=xyz, rgb=rgb, view_id=view_id)
        manifest["files"].append(f"{stem}.ply")
        if robot_masks_t_vhw is not None:
            image_robot_mask = np.asarray(robot_masks_t_vhw[time_index], dtype=bool)
            for subset, image_mask, key in (
                ("robot", image_robot_mask, "robot_files"),
                ("environment", ~image_robot_mask, "environment_files"),
            ):
                subset_xyz, subset_rgb, subset_view_id = backproject_rgbd(
                    rgb_t_vhwc[time_index],
                    depth_t_vhw[time_index],
                    K_t_v33[time_index],
                    poses_t_v44[time_index],
                    stride,
                    pixel_mask_vhw=image_mask,
                )
                ply_rel = f"{subset}/{stem}.ply"
                npz_rel = f"{subset}/{stem}.npz"
                write_binary_ply(directory / ply_rel, subset_xyz, subset_rgb)
                np.savez_compressed(
                    directory / npz_rel,
                    xyz=subset_xyz,
                    rgb=subset_rgb,
                    view_id=subset_view_id,
                )
                manifest[key].append(ply_rel)
    with (directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def project_urdf_robot_masks(
    triangle_soups_t: list[tuple[np.ndarray, np.ndarray]],
    depth_t_vhw: np.ndarray,
    K_t_v33: np.ndarray,
    poses_t_v44: np.ndarray,
    depth_tolerance: float = 0.03,
    dilation_pixels: int = 2,
) -> np.ndarray:
    """Raycast URDF visual meshes and match their depth to observed RGB-D."""
    import open3d as o3d
    from scipy.ndimage import binary_dilation

    depth = np.asarray(depth_t_vhw)
    masks = np.zeros(depth.shape, dtype=bool)
    for time_index, (vertices, triangles) in enumerate(triangle_soups_t):
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
        )
        scene.add_triangles(mesh)
        for view_index in range(depth.shape[1]):
            h, w = depth.shape[-2:]
            vv, uu = np.mgrid[:h, :w]
            K = K_t_v33[time_index, view_index]
            directions_camera = np.stack(
                [(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))],
                axis=-1,
            )
            pose = poses_t_v44[time_index, view_index]
            directions_base = directions_camera @ pose[:3, :3].T
            origins = np.broadcast_to(pose[:3, 3], directions_base.shape)
            rays = np.concatenate([origins, directions_base], axis=-1).astype(np.float32)
            robot_depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            observed = depth[time_index, view_index]
            mask = (
                np.isfinite(robot_depth)
                & np.isfinite(observed)
                & (np.abs(observed - robot_depth) <= depth_tolerance)
            )
            if dilation_pixels > 0:
                mask = binary_dilation(mask, iterations=dilation_pixels)
            masks[time_index, view_index] = mask
    return masks


def postprocess_chunk_urdf(
    chunk_root: str | Path,
    robot_urdf: str | Path,
    depth_tolerance: float = 0.03,
    dilation_pixels: int = 2,
) -> None:
    """Project URDF masks and reconstruct one completed chunk."""
    chunk = Path(chunk_root).resolve()
    metadata_path = chunk / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    point_stride = int(metadata["point_stride"])
    sources = (
        ("predicted", "predicted_rgbd.npz", "predicted_urdf_qpos"),
        ("ground_truth", "ground_truth_rgbd.npz", "ground_truth_urdf_qpos"),
    )
    qpos_counts = [len(metadata[qpos_key]) for _, _, qpos_key in sources]
    all_qpos = [cfg for _, _, qpos_key in sources for cfg in metadata[qpos_key]]
    all_triangles = load_urdf_visual_triangles(robot_urdf, all_qpos)
    triangle_offset = 0
    for (source, archive_name, _), qpos_count in zip(sources, qpos_counts):
        archive = np.load(chunk / archive_name)
        rgb, depth = archive["rgb"], archive["depth_m"]
        K, poses = archive["K"], archive["T_base_from_camera"]
        triangles = all_triangles[triangle_offset : triangle_offset + qpos_count]
        triangle_offset += qpos_count
        masks = project_urdf_robot_masks(
            triangles, depth, K, poses, depth_tolerance, dilation_pixels
        )
        save_pointcloud_sequence(
            chunk / source / "pointclouds",
            rgb,
            depth,
            K,
            poses,
            point_stride,
            timestamps_s=archive["timestamps_s"],
            action_offsets=archive["action_offsets"] if "action_offsets" in archive else None,
            robot_masks_t_vhw=masks,
        )
        np.savez_compressed(chunk / source / "robot_masks.npz", mask=masks)
    metadata["robot_split_status"] = "complete"
    metadata["robot_mask_depth_tolerance_m"] = depth_tolerance
    metadata["robot_mask_dilation_pixels"] = dilation_pixels
    metadata["robot_geometry_source"] = str(Path(robot_urdf).expanduser().resolve())
    json_dump(metadata_path, metadata)


def stitch_chunk_pointcloud_timelines(rollout_root: str | Path) -> Path:
    """Index every chunk as continuous predicted/ground-truth timelines.

    Point clouds stay in their chunk directories; the global manifest uses
    relative paths, so stitching is fast and does not duplicate large PLYs.
    Duplicate timestamps at adjacent chunk boundaries are collapsed.
    """
    rollout_root = Path(rollout_root).resolve()
    timeline_dir = rollout_root / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "version": 1,
        "rollout_root": "..",
        "sources": {},
    }
    source_dirs = {"imagined": "predicted", "simulation": "ground_truth"}
    for source_name, disk_name in source_dirs.items():
        entries_by_time: dict[int, dict[str, Any]] = {}
        for metadata_path in sorted((rollout_root / "chunks").glob("step_*/metadata.json")):
            chunk_dir = metadata_path.parent
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            manifest_path = chunk_dir / disk_name / "pointclouds" / "manifest.json"
            if not manifest_path.exists():
                continue
            with manifest_path.open("r", encoding="utf-8") as handle:
                point_manifest = json.load(handle)
            local_times = point_manifest.get("timestamps_s")
            if local_times is None:
                local_times = list(range(int(point_manifest["frame_count"])))
            start_s = float(metadata["chunk_start_step"]) / float(metadata["action_fps"])
            for frame_index, local_s in enumerate(local_times):
                global_s = start_s + float(local_s)
                # Integer microseconds provide deterministic boundary de-duplication.
                time_key = int(round(global_s * 1_000_000.0))
                files = {}
                for subset, manifest_key in (
                    ("full", "files"),
                    ("robot", "robot_files"),
                    ("environment", "environment_files"),
                ):
                    names = point_manifest.get(manifest_key, [])
                    if frame_index < len(names):
                        path = manifest_path.parent / names[frame_index]
                        files[subset] = os.path.relpath(path, timeline_dir)
                entries_by_time[time_key] = {
                    "time_s": global_s,
                    "chunk_start_step": int(metadata["chunk_start_step"]),
                    "local_frame": frame_index,
                    "files": files,
                }
        output["sources"][source_name] = [entries_by_time[key] for key in sorted(entries_by_time)]
    output["frame_counts"] = {name: len(entries) for name, entries in output["sources"].items()}
    manifest_path = timeline_dir / "manifest.json"
    json_dump(manifest_path, output)
    return manifest_path
