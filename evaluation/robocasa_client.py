import os
import sys
import time
import json
import pickle
import subprocess
import tempfile
import zmq
import tyro
import imageio
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.robocasa_4d import (
    capture_rgbd,
    capture_robot_state,
    fit_predicted_depth_to_metric,
    json_dump,
    resize_center_crop_nearest,
    robocasa_depth_calibration_mask,
    save_depth_sequence,
    save_rgb_video,
    save_urdf_projection_masks,
    transform_intrinsics_for_resize_crop,
    validate_4d_shapes,
)

import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config

# Robocasa -> pretrain: eef axes rotated +90 deg around z.
# R_pretrain = R_robocasa @ EEF_AXES_XFORM
# R_robocasa = R_pretrain @ EEF_AXES_XFORM.T
EEF_AXES_XFORM = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

TASK_MAX_STEPS = {
    # Pick and place tasks
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door tasks
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer tasks
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove tasks
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink tasks
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee tasks
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave tasks
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

camera_names = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_obs(env, camera_names, base2world, camera_height=256, camera_width=256):
    rgbs = []
    for cam_name in camera_names:
        rgb = env.sim.render(
            height=camera_height, width=camera_width, camera_name=cam_name, depth=False, segmentation=False
        )
        rgb = rgb[::-1].copy()
        rgbs.append(rgb)

    rgbs = np.stack(rgbs, axis=0)
    rgbs_view = rgbs.transpose(1, 0, 2, 3).reshape(camera_height, -1, 3)
    rgbs_norm = rgbs.astype(np.float32) / 127.5 - 1.0

    controller = env.robots[0].composite_controller
    eef_pos, eef_mat = (
        controller.part_controllers[controller.arms[0]].ref_pos,
        controller.part_controllers[controller.arms[0]].ref_ori_mat,
    )
    eef2world = np.eye(4)
    eef2world[:3, :3] = eef_mat
    eef2world[:3, 3] = eef_pos

    eef2base = np.linalg.inv(base2world) @ eef2world
    eef2base_pos = eef2base[:3, 3]
    rot_mat = eef2base[:3, :3] @ EEF_AXES_XFORM  # robocasa -> pretrain
    rot_quat = R.from_matrix(rot_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]  # xyzw -> wxyz

    gripper_openness = (
        controller.part_controllers[list(controller.grippers.keys())[0]].joint_pos[0:1]
        / controller.part_controllers[list(controller.grippers.keys())[0]].actuator_max[0]
    )

    zero_padding = np.zeros(8)
    eef_states = np.concatenate([eef2base_pos, rot_quat, gripper_openness, zero_padding])

    return rgbs_view, rgbs_norm, eef_states


def predicted_camera_poses(
    predicted_proprios: np.ndarray,
    initial_base_from_camera: np.ndarray,
    names: list[str],
) -> np.ndarray:
    """Build future base-from-camera poses from X-WAM predicted EEF states."""
    proprios = np.asarray(predicted_proprios, dtype=np.float64)
    poses = np.repeat(initial_base_from_camera[None], len(proprios), axis=0)
    wrist_indices = [i for i, name in enumerate(names) if "eye_in_hand" in name]
    eef_poses = []
    for time_index, state in enumerate(proprios):
        eef_base = np.eye(4, dtype=np.float64)
        eef_base[:3, 3] = state[:3]
        pretrain_rotation = R.from_quat(state[3:7][[1, 2, 3, 0]]).as_matrix()
        eef_base[:3, :3] = pretrain_rotation @ EEF_AXES_XFORM.T
        eef_poses.append(eef_base)
    # Derive hand-eye from the simulator's measured frame-0 camera pose instead
    # of relying on a hard-coded robot model transform. This also makes camera
    # randomization safe as long as the wrist camera remains rigid on the hand.
    eef_poses = np.stack(eef_poses)
    hand_eye_by_view = {
        view_index: np.linalg.inv(eef_poses[0]) @ initial_base_from_camera[view_index]
        for view_index in wrist_indices
    }
    for time_index, eef_base in enumerate(eef_poses):
        for view_index in wrist_indices:
            poses[time_index, view_index] = eef_base @ hand_eye_by_view[view_index]
    return poses


def _save_camera_streams(root, prefix, rgb_t_vhwc, depth_t_vhw, camera_names, fps, raw_depth=False):
    for view_index, camera_name in enumerate(camera_names):
        save_rgb_video(root / prefix / "rgb" / f"{camera_name}.mp4", rgb_t_vhwc[:, view_index], fps)
        save_depth_sequence(
            root / prefix / "depth" / camera_name,
            depth_t_vhw[:, view_index],
            fps,
            raw=raw_depth,
        )


def capture_4d_frame(env, camera_names, base2world):
    """Keep RGB-D rendering unchanged and attach only lightweight joint state."""
    capture = capture_rgbd(env, camera_names, base2world, height=256, width=256)
    capture.update(capture_robot_state(env))
    return capture


def project_frame0_urdf_mask_isolated(
    robot_urdf,
    urdf_qpos,
    reference_depth,
    K,
    poses,
    depth_tolerance,
    dilation_pixels,
):
    """Run Open3D raycasting outside the live MuJoCo renderer process."""
    worker = os.path.join(os.path.dirname(__file__), "project_urdf_mask.py")
    with tempfile.TemporaryDirectory(prefix="xwam_urdf_mask_") as tmp:
        input_path = os.path.join(tmp, "input.npz")
        qpos_path = os.path.join(tmp, "qpos.json")
        output_path = os.path.join(tmp, "mask.npz")
        np.savez_compressed(
            input_path,
            depth=np.asarray(reference_depth)[None],
            K=np.asarray(K)[None],
            poses=np.asarray(poses)[None],
        )
        with open(qpos_path, "w", encoding="utf-8") as handle:
            json.dump(urdf_qpos, handle)
        subprocess.run(
            [
                sys.executable,
                worker,
                input_path,
                qpos_path,
                output_path,
                "--robot-urdf",
                os.path.abspath(os.path.expanduser(robot_urdf)),
                "--depth-tolerance",
                str(depth_tolerance),
                "--dilation-pixels",
                str(dilation_pixels),
            ],
            check=True,
        )
        with np.load(output_path) as archive:
            return archive["mask"][0].copy()


def save_4d_chunk(
    chunk_root,
    result,
    initial_capture,
    gt_captures,
    camera_names,
    nominal_actions,
    executed_actions,
    gt_action_offsets,
    chunk_start_step,
    video_fps,
    action_fps,
    point_stride,
    model_crop_ratio,
    pred_depth_representation,
    robot_urdf,
    robot_mask_depth_tolerance,
    robot_mask_dilation_pixels,
):
    """Persist predicted/ground-truth RGB-D and reconstruct both 4D sequences."""
    chunk_root = os.fspath(chunk_root)
    from pathlib import Path

    root = Path(chunk_root)
    root.mkdir(parents=True, exist_ok=True)
    pred_rgb = np.asarray(result["predicted_rgb"], dtype=np.uint8)  # [V,T,H,W,C]
    pred_rgb = pred_rgb.transpose(1, 0, 2, 3, 4)
    pred_depth_rgb = np.asarray(result["predicted_depth_raw"], dtype=np.uint8).transpose(1, 0, 2, 3, 4)
    pred_depth_raw = pred_depth_rgb.astype(np.float32).mean(axis=-1)
    pred_h, pred_w = pred_depth_raw.shape[-2:]

    transformed_K = transform_intrinsics_for_resize_crop(
        initial_capture["K"],
        initial_capture["depth_m"].shape[-2:],
        (pred_h, pred_w),
        model_crop_ratio,
    )
    reference_depth = resize_center_crop_nearest(
        initial_capture["depth_m"], (pred_h, pred_w), model_crop_ratio
    )
    calibration_mask = None
    calibration_regions = ["full_frame"] * len(camera_names)
    if robot_urdf and pred_depth_representation == "inverse":
        frame0_robot_mask = project_frame0_urdf_mask_isolated(
            robot_urdf,
            initial_capture["urdf_qpos"],
            reference_depth,
            transformed_K,
            initial_capture["T_base_from_camera"],
            robot_mask_depth_tolerance,
            robot_mask_dilation_pixels,
        )
        calibration_mask, calibration_regions = robocasa_depth_calibration_mask(
            frame0_robot_mask, camera_names
        )
        save_urdf_projection_masks(
            root / "urdf_proj_mask", frame0_robot_mask[None], camera_names
        )
    pred_depth_m, depth_calibration = fit_predicted_depth_to_metric(
        pred_depth_raw,
        reference_depth,
        representation=pred_depth_representation,
        view_names=camera_names,
        calibration_mask_vhw=calibration_mask,
    )
    for entry, region in zip(depth_calibration.get("per_view", []), calibration_regions):
        entry["fit_region"] = region
    pred_poses = predicted_camera_poses(
        result["proprios"], initial_capture["T_base_from_camera"], camera_names
    )
    pred_frames = len(pred_rgb)
    if len(pred_depth_m) != pred_frames or len(pred_poses) != pred_frames:
        raise ValueError(
            "Predicted RGB/depth/pose frame counts differ: "
            f"{pred_frames}/{len(pred_depth_m)}/{len(pred_poses)}"
        )
    pred_K = np.repeat(transformed_K[None], pred_frames, axis=0)
    pred_world_poses = initial_capture["T_world_from_base"][None, None] @ pred_poses

    gt_rgb = np.stack([frame["rgb"] for frame in gt_captures])
    gt_depth = np.stack([frame["depth_m"] for frame in gt_captures])
    gt_K = np.stack([frame["K"] for frame in gt_captures])
    gt_poses = np.stack([frame["T_base_from_camera"] for frame in gt_captures])
    gt_action_offsets = np.asarray(gt_action_offsets, dtype=np.int32)
    gt_timestamps_s = gt_action_offsets.astype(np.float64) / float(action_fps)
    pred_timestamps_s = np.arange(pred_frames, dtype=np.float64) / float(video_fps)
    pred_action_offsets = pred_timestamps_s * float(action_fps)
    pred_geom_indices = np.abs(
        gt_action_offsets[:, None] - pred_action_offsets[None, :]
    ).argmin(axis=0)
    gt_urdf_qpos = [frame["urdf_qpos"] for frame in gt_captures]
    pred_urdf_qpos = [gt_urdf_qpos[index] for index in pred_geom_indices]

    validate_4d_shapes(pred_rgb, pred_depth_m, pred_K, pred_poses, camera_names, "predicted")
    validate_4d_shapes(
        gt_rgb,
        gt_depth,
        gt_K,
        gt_poses,
        camera_names,
        "ground truth",
        action_offsets=gt_action_offsets,
        executed_action_count=len(executed_actions),
    )

    np.savez_compressed(
        root / "predicted_rgbd.npz",
        rgb=pred_rgb,
        depth_raw=pred_depth_raw,
        depth_raw_rgb=pred_depth_rgb,
        depth_m=pred_depth_m,
        K=pred_K,
        T_base_from_camera=pred_poses,
        T_world_from_camera=pred_world_poses,
        proprios=result["proprios"],
        nominal_actions=nominal_actions,
        executed_controller_actions=executed_actions,
        timestamps_s=pred_timestamps_s,
        robot_qpos=np.stack([gt_captures[index]["robot_qpos"] for index in pred_geom_indices]),
        simulator_qpos=np.stack([gt_captures[index]["sim_qpos"] for index in pred_geom_indices]),
    )
    np.savez_compressed(
        root / "ground_truth_rgbd.npz",
        rgb=gt_rgb,
        depth_m=gt_depth,
        K=gt_K,
        T_base_from_camera=gt_poses,
        T_world_from_camera=np.stack([frame["T_world_from_camera"] for frame in gt_captures]),
        action_offsets=gt_action_offsets,
        timestamps_s=gt_timestamps_s,
        executed_controller_actions=executed_actions,
        robot_qpos=np.stack([frame["robot_qpos"] for frame in gt_captures]),
        simulator_qpos=np.stack([frame["sim_qpos"] for frame in gt_captures]),
    )
    json_dump(root / "metadata.json", {
        "camera_names": camera_names,
        "prediction": result.get("prediction_metadata", {}),
        "predicted_depth_calibration": depth_calibration,
        "predicted_depth_warning": (
            "Metric predicted depth is an inverse-affine calibration using selected frame-0 "
            "measured-depth pixels (fixed-view background; eye-in-hand robot). "
            "Use depth_raw for auditing; this calibration is not ground-truth future depth."
        ),
        "predicted_frame_count": pred_frames,
        "ground_truth_frame_count": len(gt_captures),
        "chunk_start_step": chunk_start_step,
        "ground_truth_action_offsets": gt_action_offsets.tolist(),
        "video_fps": video_fps,
        "action_fps": action_fps,
        "ground_truth_effective_fps": (
            float(action_fps / np.diff(gt_action_offsets).mean()) if len(gt_action_offsets) > 1 else None
        ),
        "ground_truth_is_dense_per_action": bool(
            len(gt_action_offsets) > 1 and np.all(np.diff(gt_action_offsets) == 1)
        ),
        "point_stride": point_stride,
        "coordinate_frame": "robot_base",
        "length_unit": "metre",
        "predicted_robot_state_source": "nearest synchronized ground-truth simulator qpos/geometry",
        "robot_geometry_source": os.path.abspath(os.path.expanduser(robot_urdf)) if robot_urdf else None,
        "robot_split_status": "pending_offline" if robot_urdf else "disabled",
        "robot_joint_names": gt_captures[0]["robot_joint_names"],
        "ground_truth_urdf_qpos": gt_urdf_qpos,
        "predicted_urdf_qpos": pred_urdf_qpos,
    })
    json_dump(root / "predicted" / "cameras.json", {
        "camera_names": camera_names,
        "K": pred_K.tolist(),
        "T_base_from_camera": pred_poses.tolist(),
        "T_world_from_camera": pred_world_poses.tolist(),
    })
    json_dump(root / "ground_truth" / "cameras.json", {
        "camera_names": camera_names,
        "K": gt_K.tolist(),
        "T_base_from_camera": gt_poses.tolist(),
        "T_world_from_camera": np.stack(
            [frame["T_world_from_camera"] for frame in gt_captures]
        ).tolist(),
        "action_offsets": gt_action_offsets.tolist(),
        "timestamps_s": gt_timestamps_s.tolist(),
    })

    _save_camera_streams(root, "predicted", pred_rgb, pred_depth_raw, camera_names, video_fps, raw_depth=True)
    for view_index, camera_name in enumerate(camera_names):
        save_depth_sequence(root / "predicted" / "depth_metric" / camera_name, pred_depth_m[:, view_index], video_fps)
    gt_effective_fps = action_fps / np.diff(gt_action_offsets).mean() if len(gt_action_offsets) > 1 else action_fps
    _save_camera_streams(root, "ground_truth", gt_rgb, gt_depth, camera_names, gt_effective_fps)

def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,  # (not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


@dataclass
class Args:
    action_length: int = 32
    save_root_dir: str = "./eval_results/robocasa/"
    env_global_rank: int = 0
    """Global rank of this client across all machines and environments"""
    world_size: int = 1
    """Total number of environment clients across all machines (WORLD_SIZE * num_envs_per_machine)"""
    num_evals_per_worker: int = 5
    server_addr: str = "localhost"
    server_port: int = 10086
    """Broker frontend port (must match policy_broker.py --frontend_port)"""
    cfg: float = 0.0
    capture_4d: bool = False
    """Finish X-WAM RGB-D generation and save predicted/ground-truth 4D captures."""
    capture_stride: int = 1
    """Simulator action steps between ground-truth RGB-D frames (1 exports dense per-action 4D)."""
    capture_fps: float = 5.0
    """X-WAM generated video frame rate."""
    action_fps: float = 20.0
    """Simulator controller/action rate used to timestamp dense ground-truth RGB-D."""
    point_stride: int = 2
    """Pixel stride used when exporting fused point clouds."""
    model_crop_ratio: float = 0.95
    pred_depth_representation: str = "inverse"
    """Metric conversion for generated depth: inverse (default) or metric."""
    robot_mask_depth_tolerance: float = 0.03
    """Depth tolerance in metres when matching RGB-D pixels to projected URDF mesh."""
    robot_mask_dilation_pixels: int = 2
    """Image-space dilation applied to the projected robot mask."""
    robot_urdf: str = "../PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf"
    """Franka URDF visual meshes projected into RGB-D views; empty disables reconstruction."""


def main(args: Args):
    if args.capture_stride < 1:
        raise ValueError("capture_stride must be >= 1")
    if args.capture_fps <= 0 or args.action_fps <= 0:
        raise ValueError("capture_fps and action_fps must be > 0")
    if args.point_stride < 1:
        raise ValueError("point_stride must be >= 1")
    if args.robot_mask_depth_tolerance < 0 or args.robot_mask_dilation_pixels < 0:
        raise ValueError("robot mask tolerance and dilation must be >= 0")
    if args.capture_4d and args.robot_urdf and not os.path.isfile(os.path.expanduser(args.robot_urdf)):
        raise FileNotFoundError(f"robot_urdf does not exist: {args.robot_urdf}")
    if not 0 < args.model_crop_ratio <= 1:
        raise ValueError("model_crop_ratio must be in (0, 1]")
    if args.pred_depth_representation not in {"inverse", "metric"}:
        raise ValueError("pred_depth_representation must be 'inverse' or 'metric'")

    env_name_list = list(TASK_MAX_STEPS.keys())
    env_name = env_name_list[args.env_global_rank % len(env_name_list)]

    global_rank = args.env_global_rank
    world_size = args.world_size

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{args.server_addr}:{args.server_port}")

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(args.num_evals_per_worker)):
        env = create_env(
            env_name=env_name,
            render_onscreen=False,
            seed=global_rank * args.num_evals_per_worker + rollout_i,  # set seed=None to run unseeded
        )
        env.reset()

        controller = env.robots[0].composite_controller
        base_pos, base_mat = (
            controller.part_controllers[controller.arms[0]].origin_pos,
            controller.part_controllers[controller.arms[0]].origin_ori,
        )
        base2world = np.eye(4)
        base2world[:3, :3] = base_mat
        base2world[:3, 3] = base_pos

        # run rollouts with random actions and save video
        num_steps = TASK_MAX_STEPS[env_name]

        video_array = []

        print(f"Rollout {rollout_i} / {args.num_evals_per_worker} started: {env_name} - {env.get_ep_meta()['lang']}")

        step_i = 0
        success = False
        while step_i < num_steps:
            chunk_start_step = step_i
            initial_capture = None
            if args.capture_4d:
                initial_capture = capture_4d_frame(env, camera_names, base2world)
                rgbs_uint8 = initial_capture["rgb"]
                rgbs = rgbs_uint8.astype(np.float32) / 127.5 - 1.0
                _, _, eef_states = render_obs(env, camera_names, base2world)
            else:
                _, rgbs, eef_states = render_obs(env, camera_names, base2world)

            data_batch = {
                "env_rank": global_rank,
                "rollout_id": rollout_i,
                "step_id": step_i,
                "video": rgbs.copy(),
                "proprios": eef_states.copy(),
                "prompt": [env.get_ep_meta()["lang"]],
                "cfg": args.cfg,
                "capture_4d": args.capture_4d,
            }

            socket.send(pickle.dumps(data_batch))
            result = pickle.loads(socket.recv())  # shape: [Ta, Da]
            action = result["actions"]

            action = action[: args.action_length]
            gt_captures = [initial_capture] if args.capture_4d else []
            gt_action_offsets = [0] if args.capture_4d else []
            executed_actions = []
            pad_action = np.zeros(env.action_spec[0].shape)
            for ai in range(action.shape[0]):
                pad_action[:7] = action[ai]

                if step_i % 4 == 0:
                    video_img, _, _ = render_obs(env, camera_names, base2world)
                    video_array.append(video_img)

                env.step(pad_action)
                executed_actions.append(pad_action.copy())
                step_i += 1

                if args.capture_4d and (ai + 1) % args.capture_stride == 0:
                    gt_captures.append(capture_4d_frame(env, camera_names, base2world))
                    gt_action_offsets.append(ai + 1)

                if env._check_success():
                    success = True
                    num_success_rollouts += 1
                    break

                if step_i >= num_steps:
                    success = False
                    break

            if args.capture_4d:
                # Preserve an early-success / max-step terminal state even when
                # it does not land exactly on the regular capture stride.
                if gt_action_offsets[-1] != len(executed_actions):
                    gt_captures.append(capture_4d_frame(env, camera_names, base2world))
                    gt_action_offsets.append(len(executed_actions))
                rollout_root = os.path.join(
                    args.save_root_dir,
                    env_name,
                    f"{global_rank}_{rollout_i}_4d",
                    "chunks",
                    f"step_{chunk_start_step:06d}",
                )
                save_4d_chunk(
                    rollout_root,
                    result,
                    initial_capture,
                    gt_captures,
                    camera_names,
                    action,
                    np.asarray(executed_actions),
                    gt_action_offsets,
                    chunk_start_step,
                    args.capture_fps,
                    args.action_fps,
                    args.point_stride,
                    args.model_crop_ratio,
                    args.pred_depth_representation,
                    args.robot_urdf,
                    args.robot_mask_depth_tolerance,
                    args.robot_mask_dilation_pixels,
                )
                print(f"Saved 4D capture to {rollout_root}")
                if args.robot_urdf:
                    postprocess_script = os.path.join(
                        os.path.dirname(__file__), "postprocess_4d_chunk.py"
                    )
                    subprocess.run([
                        sys.executable,
                        postprocess_script,
                        os.path.abspath(rollout_root),
                        "--robot-urdf",
                        os.path.abspath(os.path.expanduser(args.robot_urdf)),
                        "--depth-tolerance",
                        str(args.robot_mask_depth_tolerance),
                        "--dilation-pixels",
                        str(args.robot_mask_dilation_pixels),
                    ], check=True)
                    print(f"Saved projected-mask 4D point clouds to {rollout_root}")

            if success:
                break

        env.close()

        if args.capture_4d:
            rollout_4d_root = os.path.abspath(os.path.join(
                args.save_root_dir, env_name, f"{global_rank}_{rollout_i}_4d"
            ))
            postprocess_script = os.path.join(os.path.dirname(__file__), "postprocess_4d.py")
            command = [sys.executable, postprocess_script, rollout_4d_root]
            print("Indexing completed chunk point clouds after env.close()")
            subprocess.run(command, check=True)

        os.makedirs(f"{args.save_root_dir}/{env_name}", exist_ok=True)
        video_path = (
            f"{args.save_root_dir}/{env_name}/{global_rank}_{rollout_i}_{'success' if success else 'failure'}.mp4"
        )
        imageio.mimsave(video_path, video_array, fps=10)
        print(f"Saved video to {video_path}")

    info[env_name] = {
        "num_success_rollouts": num_success_rollouts,
        "num_rollouts": args.num_evals_per_worker,
        "success_rate": num_success_rollouts / args.num_evals_per_worker,
    }

    print(info)
    with open(os.path.join(args.save_root_dir, f"eval_results_{global_rank}.json"), "w") as f:
        json.dump(info, f, indent=4)

    while True:
        end_files = [e for e in os.listdir(args.save_root_dir) if e.endswith(".json")]
        if len(end_files) >= world_size:
            break
        print(
            f"[Rank {global_rank}] Waiting for all end files... ({len(list(end_files))}/{world_size}) files present. Sleeping for 30 seconds."
        )
        time.sleep(30)


if __name__ == "__main__":
    main(tyro.cli(Args))
