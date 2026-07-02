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
from pathlib import Path
from typing import Any

import numpy as np


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
    manifest = {
        "frame_count": frame_count,
        "stride": stride,
        "timestamps_s": None if timestamps_s is None else np.asarray(timestamps_s).tolist(),
        "action_offsets": None if action_offsets is None else np.asarray(action_offsets).tolist(),
        "files": [],
    }
    for time_index in range(frame_count):
        xyz, rgb, view_id = backproject_rgbd(
            rgb_t_vhwc[time_index], depth_t_vhw[time_index], K_t_v33[time_index], poses_t_v44[time_index], stride
        )
        stem = f"frame_{time_index:04d}"
        write_binary_ply(directory / f"{stem}.ply", xyz, rgb)
        np.savez_compressed(directory / f"{stem}.npz", xyz=xyz, rgb=rgb, view_id=view_id)
        manifest["files"].append(f"{stem}.ply")
    with (directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
