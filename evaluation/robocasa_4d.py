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


def load_urdf_robot_geometries(
    urdf_path: str | Path,
    urdf_qpos_t: list[dict[str, float]],
) -> list[list[dict[str, Any]]]:
    """Build per-frame collision meshes using PointWorld's urdfpy FK pattern."""
    # urdfpy 0.0.22 still references NumPy aliases removed in recent releases.
    for alias, value in (("float", float), ("int", int), ("bool", bool)):
        if alias not in np.__dict__:
            setattr(np, alias, value)
    try:
        import urdfpy
    except ImportError as exc:
        raise ImportError(
            "URDF robot splitting requires urdfpy/trimesh; install the PointWorld "
            "URDF runtime requirements first"
        ) from exc

    path = Path(urdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    robot = urdfpy.URDF.load(str(path))
    actuated = {joint.name for joint in robot.actuated_joints}
    sequence = []
    for saved_cfg in urdf_qpos_t:
        # Match PointWorld: provide the seven Panda joints and driving finger
        # joint only. urdfpy derives the Robotiq mimic joints itself.
        cfg = {name: float(value) for name, value in saved_cfg.items() if name in actuated}
        sequence.append(_urdf_collision_geometries(robot, cfg))
    return sequence


def _urdf_collision_geometries(robot: Any, cfg: dict[str, float]) -> list[dict[str, Any]]:
    """Build collision geoms without urdfpy's broken primitive mesh properties."""
    frame_geoms = []
    for link, base_from_link in robot.link_fk(cfg=cfg).items():
        for collision in link.collisions:
            wrapper = collision.geometry
            transform = np.asarray(base_from_link, dtype=np.float64) @ np.asarray(
                collision.origin, dtype=np.float64
            )
            name = collision.name or link.name
            if wrapper.box is not None:
                frame_geoms.append({
                    "name": name,
                    "type": 6,
                    "size": np.asarray(wrapper.box.size, dtype=np.float64) / 2.0,
                    "T_base_from_geom": transform,
                })
            elif wrapper.cylinder is not None:
                frame_geoms.append({
                    "name": name,
                    "type": 5,
                    "size": np.array(
                        [wrapper.cylinder.radius, wrapper.cylinder.length / 2.0, 0.0],
                        dtype=np.float64,
                    ),
                    "T_base_from_geom": transform,
                })
            elif wrapper.sphere is not None:
                frame_geoms.append({
                    "name": name,
                    "type": 2,
                    "size": np.array([wrapper.sphere.radius, 0.0, 0.0], dtype=np.float64),
                    "T_base_from_geom": transform,
                })
            elif wrapper.mesh is not None:
                meshes = getattr(wrapper.mesh, "_meshes", None)
                if not meshes:
                    raise ValueError(f"URDF collision mesh failed to load: {wrapper.mesh.filename}")
                scale = wrapper.mesh.scale
                for mesh_index, mesh in enumerate(meshes):
                    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
                    if scale is not None:
                        vertices *= np.asarray(scale, dtype=np.float64)
                    frame_geoms.append({
                        "name": f"{name}_{mesh_index}",
                        "type": 7,
                        "size": np.zeros(3, dtype=np.float64),
                        "T_base_from_geom": transform,
                        "vertices": vertices,
                    })
    return frame_geoms


def points_inside_robot(
    xyz: np.ndarray,
    robot_geoms: list[dict[str, Any]],
    padding: float = 0.008,
) -> np.ndarray:
    """Return a mask for points inside (or just on) robot collision geometry.

    MuJoCo geom type values are stable: sphere=2, capsule=3, ellipsoid=4,
    cylinder=5, box=6, mesh=7. Mesh geoms use their convex hull equations;
    robot collision meshes are conventionally convex pieces.
    """
    points = np.asarray(xyz, dtype=np.float64)
    inside = np.zeros(len(points), dtype=bool)
    for geom in robot_geoms:
        if inside.all():
            break
        T = np.asarray(geom["T_base_from_geom"], dtype=np.float64)
        local = (points[~inside] - T[:3, 3]) @ T[:3, :3]
        size = np.asarray(geom["size"], dtype=np.float64)
        geom_type = int(geom["type"])
        if geom_type == 2:  # sphere
            hit = np.linalg.norm(local, axis=1) <= size[0] + padding
        elif geom_type == 3:  # capsule, axis is local z
            dz = np.maximum(np.abs(local[:, 2]) - size[1], 0.0)
            hit = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2 + dz ** 2) <= size[0] + padding
        elif geom_type == 4:  # ellipsoid
            hit = np.sum((local / (size + padding)) ** 2, axis=1) <= 1.0
        elif geom_type == 5:  # cylinder, radius + half-height
            hit = (np.linalg.norm(local[:, :2], axis=1) <= size[0] + padding) & (
                np.abs(local[:, 2]) <= size[1] + padding
            )
        elif geom_type == 6:  # box half extents
            hit = np.all(np.abs(local) <= size + padding, axis=1)
        elif geom_type == 7 and len(geom.get("vertices", [])) >= 4:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(np.asarray(geom["vertices"], dtype=np.float64))
            hit = np.all(local @ hull.equations[:, :3].T + hull.equations[:, 3] <= padding, axis=1)
        else:
            continue
        remaining = np.flatnonzero(~inside)
        inside[remaining[hit]] = True
    return inside


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
    robot_geoms_t: list[list[dict[str, Any]]] | None = None,
    robot_padding: float = 0.008,
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
    if robot_geoms_t is not None and len(robot_geoms_t) != frame_count:
        raise ValueError("robot_geoms_t length must match point-cloud frame count")
    manifest = {
        "frame_count": frame_count,
        "stride": stride,
        "timestamps_s": None if timestamps_s is None else np.asarray(timestamps_s).tolist(),
        "action_offsets": None if action_offsets is None else np.asarray(action_offsets).tolist(),
        "files": [],
        "robot_files": [],
        "environment_files": [],
        "robot_padding_m": robot_padding if robot_geoms_t is not None else None,
    }
    for time_index in range(frame_count):
        xyz, rgb, view_id = backproject_rgbd(
            rgb_t_vhwc[time_index], depth_t_vhw[time_index], K_t_v33[time_index], poses_t_v44[time_index], stride
        )
        stem = f"frame_{time_index:04d}"
        write_binary_ply(directory / f"{stem}.ply", xyz, rgb)
        np.savez_compressed(directory / f"{stem}.npz", xyz=xyz, rgb=rgb, view_id=view_id)
        manifest["files"].append(f"{stem}.ply")
        if robot_geoms_t is not None:
            robot_mask = points_inside_robot(xyz, robot_geoms_t[time_index], padding=robot_padding)
            for subset, mask, key in (
                ("robot", robot_mask, "robot_files"),
                ("environment", ~robot_mask, "environment_files"),
            ):
                ply_rel = f"{subset}/{stem}.ply"
                npz_rel = f"{subset}/{stem}.npz"
                write_binary_ply(directory / ply_rel, xyz[mask], rgb[mask])
                np.savez_compressed(
                    directory / npz_rel, xyz=xyz[mask], rgb=rgb[mask], view_id=view_id[mask]
                )
                manifest[key].append(ply_rel)
    with (directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def split_saved_pointcloud_sequence(
    directory: str | Path,
    robot_geoms_t: list[list[dict[str, Any]]],
    robot_padding: float,
) -> None:
    """Split an already-saved full point-cloud sequence without RGB-D rendering."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    full_files = manifest.get("files", [])
    if len(full_files) != len(robot_geoms_t):
        raise ValueError(
            f"Geometry/frame mismatch in {directory}: {len(robot_geoms_t)} vs {len(full_files)}"
        )
    manifest["robot_files"] = []
    manifest["environment_files"] = []
    manifest["robot_padding_m"] = robot_padding
    for frame_index, ply_name in enumerate(full_files):
        stem = Path(ply_name).stem
        payload = np.load(directory / f"{stem}.npz")
        xyz = payload["xyz"]
        rgb = payload["rgb"]
        view_id = payload["view_id"]
        robot_mask = points_inside_robot(xyz, robot_geoms_t[frame_index], padding=robot_padding)
        for subset, mask, key in (
            ("robot", robot_mask, "robot_files"),
            ("environment", ~robot_mask, "environment_files"),
        ):
            ply_rel = f"{subset}/{stem}.ply"
            npz_rel = f"{subset}/{stem}.npz"
            write_binary_ply(directory / ply_rel, xyz[mask], rgb[mask])
            (directory / subset).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / npz_rel, xyz=xyz[mask], rgb=rgb[mask], view_id=view_id[mask]
            )
            manifest[key].append(ply_rel)
    json_dump(manifest_path, manifest)


def postprocess_rollout_urdf(
    rollout_root: str | Path,
    robot_urdf: str | Path,
    robot_padding: float = 0.008,
) -> None:
    """Apply URDF point splitting after simulation has closed."""
    rollout_root = Path(rollout_root).resolve()
    for metadata_path in sorted((rollout_root / "chunks").glob("step_*/metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        chunk = metadata_path.parent
        for source, qpos_key in (
            ("predicted", "predicted_urdf_qpos"),
            ("ground_truth", "ground_truth_urdf_qpos"),
        ):
            qpos = metadata.get(qpos_key)
            if qpos is None:
                raise ValueError(f"Missing {qpos_key} in {metadata_path}")
            geoms = load_urdf_robot_geometries(robot_urdf, qpos)
            split_saved_pointcloud_sequence(chunk / source / "pointclouds", geoms, robot_padding)
        metadata["robot_split_status"] = "complete"
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
