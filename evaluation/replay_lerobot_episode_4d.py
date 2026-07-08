"""Replay one RoboCasa LeRobot episode and export RGB-D plus 4D point clouds.

This is a data-replay companion to ``evaluation/robocasa_client.py``. It does
not call X-WAM; it restores a recorded RoboCasa episode, replays either the
stored controller actions or simulator states, renders RGB-D from the standard
three cameras, and writes the same point-cloud timeline shape consumed by
``evaluation/view_4d_timeline_legacy.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.robocasa_4d import (
    capture_rgbd,
    capture_robot_state,
    json_dump,
    save_depth_sequence,
    save_pointcloud_sequence,
    save_rgb_video,
    stitch_chunk_pointcloud_timelines,
)


DEFAULT_CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def infer_env_name(dataset_root: Path) -> str:
    root = dataset_root.resolve()
    if root.name == "lerobot" and len(root.parents) >= 2:
        return root.parent.parent.name
    return root.name


def is_lerobot_root(path: Path) -> bool:
    return (path / "data").is_dir() and (path / "meta").is_dir() and (path / "extras").is_dir()


def resolve_lerobot_root(path: Path) -> Path:
    """Accept either the LeRobot root itself or a parent that contains one."""
    root = path.expanduser().resolve()
    if is_lerobot_root(root):
        return root
    if is_lerobot_root(root / "lerobot"):
        return root / "lerobot"
    matches = sorted(candidate for candidate in root.glob("**/lerobot") if is_lerobot_root(candidate))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple LeRobot roots under {root}; pass one explicitly:\n"
            + "\n".join(f"  {match}" for match in matches[:20])
        )
    raise FileNotFoundError(
        f"{root} is not a LeRobot root and no nested lerobot/ with data/, meta/, extras/ was found"
    )


def episode_name(index: int) -> str:
    return f"episode_{index:06d}"


def find_episode_parquet(dataset_root: Path, index: int) -> Path:
    pattern = f"{episode_name(index)}.parquet"
    matches = sorted((dataset_root / "data").glob(f"chunk-*/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {pattern} under {dataset_root / 'data'}")
    return matches[0]


def read_lerobot_episode_table(dataset_root: Path, index: int) -> dict[str, np.ndarray]:
    """Load action/timestamp/done columns from a LeRobot parquet episode."""
    parquet_path = find_episode_parquet(dataset_root, index)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading LeRobot parquet requires pyarrow in the RoboCasa replay "
            "environment. Install pyarrow or run with --replay-mode states."
        ) from exc

    columns = ["action", "timestamp", "next.done", "episode_index", "frame_index", "index"]
    available = set(pq.read_schema(parquet_path).names)
    table = pq.read_table(parquet_path, columns=[name for name in columns if name in available])
    result: dict[str, np.ndarray] = {}
    for name in table.column_names:
        values = table[name].to_pylist()
        if name == "action":
            result[name] = np.asarray(values, dtype=np.float64)
        else:
            result[name] = np.asarray(values)
    return result


def read_episode_task(dataset_root: Path, index: int) -> str | None:
    episodes = dataset_root / "meta" / "episodes.jsonl"
    if not episodes.exists():
        return None
    with episodes.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if int(entry.get("episode_index", -1)) == index:
                tasks = entry.get("tasks") or []
                return tasks[0] if tasks else None
    return None


def load_states(extra_dir: Path) -> np.ndarray:
    states_path = extra_dir / "states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"Missing simulator states: {states_path}")
    with np.load(states_path) as archive:
        if "states" not in archive:
            raise KeyError(f"{states_path} does not contain a 'states' array")
        return np.asarray(archive["states"], dtype=np.float64)


def reset_env_to_recorded_initial_state(env: Any, extra_dir: Path, states: np.ndarray) -> None:
    """Restore the exact recorded MuJoCo model and first flattened state."""
    env.reset()
    xml_gz = extra_dir / "model.xml.gz"
    if xml_gz.exists():
        with gzip.open(xml_gz, "rb") as handle:
            xml = handle.read().decode("utf-8")
        if hasattr(env, "reset_from_xml_string"):
            env.reset_from_xml_string(xml)
        elif hasattr(env.sim, "reset_from_xml_string"):
            env.sim.reset_from_xml_string(xml)
        else:
            raise AttributeError(
                "This RoboSuite env cannot reset from model.xml.gz; "
                "expected env.reset_from_xml_string or env.sim.reset_from_xml_string."
            )
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()


def capture_4d_frame(env: Any, camera_names: list[str], base2world: np.ndarray, height: int, width: int) -> dict[str, Any]:
    frame = capture_rgbd(env, camera_names, base2world, height=height, width=width)
    frame.update(capture_robot_state(env))
    return frame


def robot_base_pose(env: Any) -> np.ndarray:
    controller = env.robots[0].composite_controller
    arm = controller.arms[0]
    arm_controller = controller.part_controllers[arm]
    base2world = np.eye(4, dtype=np.float64)
    base2world[:3, :3] = arm_controller.origin_ori
    base2world[:3, 3] = arm_controller.origin_pos
    return base2world


def project_action(raw_action: np.ndarray, env_action_dim: int, mode: str) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if mode == "full":
        projected = raw
    elif mode == "eef7":
        if raw.size >= 12:
            projected = np.concatenate([raw[5:11], raw[11:12]])
        elif raw.size >= 7:
            projected = raw[:7]
        else:
            raise ValueError(f"Cannot make eef7 action from shape {raw.shape}")
    elif mode == "auto":
        if raw.size == env_action_dim:
            projected = raw
        elif env_action_dim == 7 and raw.size >= 12:
            projected = np.concatenate([raw[5:11], raw[11:12]])
        elif env_action_dim == 7 and raw.size >= 7:
            projected = raw[:7]
        else:
            projected = raw
    else:
        raise ValueError(f"Unknown action projection: {mode}")

    if projected.size != env_action_dim and mode == "full":
        raise ValueError(
            "Recorded action dimension does not match the simulator action spec: "
            f"recorded={projected.size}, env={env_action_dim}. This script defaults "
            "to exact simulator-action replay; use --action-projection eef7 only "
            "for debugging a 7D end-effector controller, not for faithful replay."
        )

    action = np.zeros(env_action_dim, dtype=np.float64)
    n = min(env_action_dim, projected.size)
    action[:n] = projected[:n]
    return action


def save_camera_streams(root: Path, prefix: str, rgb: np.ndarray, depth: np.ndarray, camera_names: list[str], fps: float) -> None:
    for view, camera_name in enumerate(camera_names):
        save_rgb_video(root / prefix / "rgb" / f"{camera_name}.mp4", rgb[:, view], fps=fps)
        save_depth_sequence(root / prefix / "depth" / camera_name, depth[:, view], fps=fps)


def write_replay_outputs(
    chunk_root: Path,
    captures: list[dict[str, Any]],
    camera_names: list[str],
    actions: np.ndarray | None,
    action_offsets: np.ndarray,
    action_fps: float,
    point_stride: int,
    metadata: dict[str, Any],
) -> Path:
    chunk_root.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([frame["rgb"] for frame in captures])
    depth = np.stack([frame["depth_m"] for frame in captures])
    K = np.stack([frame["K"] for frame in captures])
    poses = np.stack([frame["T_base_from_camera"] for frame in captures])
    world_poses = np.stack([frame["T_world_from_camera"] for frame in captures])
    timestamps = action_offsets.astype(np.float64) / float(action_fps)

    np.savez_compressed(
        chunk_root / "ground_truth_rgbd.npz",
        rgb=rgb,
        depth_m=depth,
        K=K,
        T_base_from_camera=poses,
        T_world_from_camera=world_poses,
        action_offsets=action_offsets,
        timestamps_s=timestamps,
        executed_controller_actions=np.empty((0,)) if actions is None else actions,
        robot_qpos=np.stack([frame["robot_qpos"] for frame in captures]),
        simulator_qpos=np.stack([frame["sim_qpos"] for frame in captures]),
    )
    json_dump(chunk_root / "ground_truth" / "cameras.json", {
        "camera_names": camera_names,
        "K": K.tolist(),
        "T_base_from_camera": poses.tolist(),
        "T_world_from_camera": world_poses.tolist(),
        "action_offsets": action_offsets.tolist(),
        "timestamps_s": timestamps.tolist(),
    })
    save_camera_streams(chunk_root, "ground_truth", rgb, depth, camera_names, fps=action_fps)
    save_pointcloud_sequence(
        chunk_root / "ground_truth" / "pointclouds",
        rgb,
        depth,
        K,
        poses,
        stride=point_stride,
        timestamps_s=timestamps,
        action_offsets=action_offsets,
        camera_names=camera_names,
    )
    json_dump(chunk_root / "metadata.json", {
        **metadata,
        "camera_names": camera_names,
        "ground_truth_frame_count": len(captures),
        "chunk_start_step": 0,
        "ground_truth_action_offsets": action_offsets.tolist(),
        "action_fps": action_fps,
        "ground_truth_effective_fps": action_fps,
        "ground_truth_is_dense_per_action": bool(len(action_offsets) > 1 and np.all(np.diff(action_offsets) == 1)),
        "point_stride": point_stride,
        "coordinate_frame": "robot_base",
        "length_unit": "metre",
        "robot_joint_names": captures[0]["robot_joint_names"],
        "robot_geometry_source": None,
        "robot_split_status": "disabled",
    })
    return chunk_root


def make_env(args: argparse.Namespace, camera_names: list[str]):
    from evaluation.robocasa_client import create_env

    layout_and_style_ids = args.layout_and_style_ids
    if layout_and_style_ids is None:
        extra_meta = args.dataset_root / "extras" / episode_name(args.episode_index) / "ep_meta.json"
        if extra_meta.exists():
            meta = json.loads(extra_meta.read_text(encoding="utf-8"))
            if "layout_id" in meta and "style_id" in meta:
                layout_and_style_ids = ((int(meta["layout_id"]), int(meta["style_id"])),)
    return create_env(
        env_name=args.env_name or infer_env_name(args.dataset_root),
        robots=args.robots,
        camera_names=camera_names,
        camera_widths=args.camera_width,
        camera_heights=args.camera_height,
        seed=args.seed,
        render_onscreen=args.render_onscreen,
        layout_and_style_ids=layout_and_style_ids,
    )


def parse_layout_and_style(value: str | None):
    if value is None:
        return None
    pairs = []
    for item in value.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*[:/]\s*(\d+)\s*", item)
        if not match:
            raise argparse.ArgumentTypeError("Use layout/style pairs like '21:35' or '21:35,22:36'")
        pairs.append((int(match.group(1)), int(match.group(2))))
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="LeRobot root containing data/, meta/, extras/, or a parent directory containing lerobot/",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--env-name", default=None, help="RoboCasa env name; defaults to the task directory name")
    parser.add_argument("--output-root", type=Path, default=Path("eval_results/robocasa_lerobot_replay"))
    parser.add_argument("--replay-mode", choices=["actions", "states"], default="actions")
    parser.add_argument("--action-projection", choices=["auto", "full", "eef7"], default="full")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--capture-stride", type=int, default=1)
    parser.add_argument("--action-fps", type=float, default=20.0)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-names", default=",".join(DEFAULT_CAMERA_NAMES))
    parser.add_argument("--robots", default="PandaOmron")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render-onscreen", action="store_true")
    parser.add_argument("--layout-and-style-ids", type=parse_layout_and_style, default=None)
    args = parser.parse_args()

    if args.capture_stride < 1:
        raise ValueError("--capture-stride must be >= 1")
    if args.action_fps <= 0:
        raise ValueError("--action-fps must be > 0")
    if args.point_stride < 1:
        raise ValueError("--point-stride must be >= 1")

    args.dataset_root = resolve_lerobot_root(args.dataset_root)
    camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]
    extra_dir = args.dataset_root / "extras" / episode_name(args.episode_index)
    states = load_states(extra_dir)
    table = read_lerobot_episode_table(args.dataset_root, args.episode_index) if args.replay_mode == "actions" else {}
    task = read_episode_task(args.dataset_root, args.episode_index)

    env = make_env(args, camera_names)
    executed_actions: list[np.ndarray] = []
    captures: list[dict[str, Any]] = []
    action_offsets: list[int] = []
    try:
        reset_env_to_recorded_initial_state(env, extra_dir, states)
        base2world = robot_base_pose(env)
        captures.append(capture_4d_frame(env, camera_names, base2world, args.camera_height, args.camera_width))
        action_offsets.append(0)

        if args.replay_mode == "states":
            state_count = len(states) if args.max_steps is None else min(len(states), args.max_steps + 1)
            for state_index in range(1, state_count):
                env.sim.set_state_from_flattened(states[state_index])
                env.sim.forward()
                if state_index % args.capture_stride == 0 or state_index == state_count - 1:
                    captures.append(capture_4d_frame(env, camera_names, base2world, args.camera_height, args.camera_width))
                    action_offsets.append(state_index)
        else:
            raw_actions = np.asarray(table["action"], dtype=np.float64)
            step_count = len(raw_actions) if args.max_steps is None else min(len(raw_actions), args.max_steps)
            env_action_dim = int(np.asarray(env.action_spec[0]).size)
            for step in range(step_count):
                action = project_action(raw_actions[step], env_action_dim, args.action_projection)
                env.step(action)
                executed_actions.append(action.copy())
                offset = step + 1
                if offset % args.capture_stride == 0 or offset == step_count:
                    captures.append(capture_4d_frame(env, camera_names, base2world, args.camera_height, args.camera_width))
                    action_offsets.append(offset)
    finally:
        env.close()

    rollout_root = (
        args.output_root.expanduser().resolve()
        / (args.env_name or infer_env_name(args.dataset_root))
        / f"{episode_name(args.episode_index)}_4d"
    )
    chunk_root = rollout_root / "chunks" / "step_000000"
    metadata = {
        "replay_source": "lerobot_robocasa",
        "replay_mode": args.replay_mode,
        "dataset_root": str(args.dataset_root),
        "episode_index": args.episode_index,
        "episode_task": task,
        "source_state_count": int(len(states)),
        "source_action_count": int(len(table.get("action", []))),
        "action_projection": args.action_projection,
    }
    write_replay_outputs(
        chunk_root,
        captures,
        camera_names,
        None if not executed_actions else np.stack(executed_actions),
        np.asarray(action_offsets, dtype=np.int32),
        args.action_fps,
        args.point_stride,
        metadata,
    )
    manifest = stitch_chunk_pointcloud_timelines(rollout_root)
    print(f"Saved replay chunk to {chunk_root}")
    print(f"Saved timeline manifest to {manifest}")
    print(f"View with: python evaluation/view_4d_timeline_legacy.py {rollout_root}")


if __name__ == "__main__":
    main()
