# X-WAM Evaluation Guidelines

## Installation

Please clone the whole repository with submodules:

```bash
git clone --recurse-submodules https://github.com/sharinka0715/X-WAM.git
cd X-WAM
```

If you have already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Base Environment

Follow the main [README](../README.md) to install the base environment (PyTorch, FlashAttention, etc.).

### RoboCasa

Refer to `third_party/robocasa/README.md` for installation.

### RoboTwin 2.0

Refer to `third_party/RoboTwin/README.md` for installation. You can ignore the `torch` / `huggingface_hub` version requirements in `third_party/RoboTwin/script/requirements.txt`.

### Fix NumPy versions

If you install all evaluation packages in one environment, you should make sure that NumPy version is compatible (we use `numpy==1.23.5` in our experiments).

## Download Checkpoints

Download the checkpoints from Hugging Face:

```bash
hf download sharinka0715/X-WAM-checkpoints --local-dir checkpoints
```

You also need the Wan2.2-TI2V-5B base weights. Specify the path via `--wan_checkpoint_dir` when launching the policy server.

## Evaluation

The evaluation system uses a broker-server-client architecture:
- **Policy Broker**: middleware that dispatches inference requests from clients to servers
- **Policy Server**: loads the model and performs inference
- **Client**: runs the simulation environment and sends observations to the broker

### Step 1: Start the Policy Broker

```bash
python evaluation/policy_broker.py \
    --frontend_port 10086 \
    --backend_port 10087
```

### Step 2: Start the Policy Server(s)

Launch one or more policy servers (each on a separate GPU):

```bash
# RoboCasa
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
    --exp_path checkpoints/robocasa_sft \
    --wan_checkpoint_dir /path/to/wan22_5b \
    --broker_port 10087 \
    --denoise_steps 50 \
    --action_denoise_steps 10

# RoboTwin 2.0
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
    --exp_path checkpoints/robotwin_sft \
    --wan_checkpoint_dir /path/to/wan22_5b \
    --broker_port 10087 \
    --denoise_steps 50 \
    --action_denoise_steps 10
```

You can launch multiple servers on different GPUs to parallelize inference:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir /path/to/wan22_5b --broker_port 10087 &
CUDA_VISIBLE_DEVICES=2 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir /path/to/wan22_5b --broker_port 10087 &
```

### Step 3: Start the Evaluation Client(s)

#### RoboCasa

Each client evaluates one task. There are 24 tasks in total, indexed 0–23. Launch one client per task:

```bash
python evaluation/robocasa_client.py \
    --env_global_rank 0 \
    --world_size 24 \
    --num_evals_per_worker 5 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robocasa/
```

To export calibrated 4D point clouds and split robot/environment points with
the same Franka URDF used by PointWorld:

```bash
pip install urdfpy==0.0.22 --no-deps
pip install -r environments/requirements_urdfpy_runtime.txt

python evaluation/robocasa_client.py \
    --capture_4d \
    --capture_stride 1 \
    --robot_urdf ../PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf \
    --robot_mask_depth_tolerance 0.03 \
    --robot_mask_dilation_pixels 2
```

Every captured frame stores simulator and robot `qpos`. After each chunk, an
isolated subprocess runs URDF FK, raycasts the visual mesh into every camera,
matches rendered and observed depth, and reconstructs `full`, `robot`, and
`environment` point clouds from the resulting pixel masks. Passing an empty
`--robot_urdf` disables point-cloud reconstruction and splitting.

Before inverse-depth calibration, the frame-0 URDF projection selects a stable
fit region independently for each view: the fixed left/right cameras use
background pixels outside the robot mask, while `eye_in_hand` uses robot pixels
inside the mask. The resulting per-view transform is applied to the whole
chunk. Lossless projected masks are saved under
`urdf_proj_mask/<camera>/` for the exact frame-0 calibration projection, plus
`predicted/urdf_proj_mask/<camera>/` and
`ground_truth/urdf_proj_mask/<camera>/` for the complete reconstructed sequences.

After each rollout, all chunk point clouds are automatically indexed into one
continuous timeline at `<rollout>_4d/timeline/manifest.json`. Adjacent duplicate
boundary frames are removed. Open the timeline locally with:

```bash
python evaluation/view_4d_timeline.py \
    eval_results/robocasa/<task>/0_0_4d
```

The window has a time slider and selectors for `imagined` / `simulation` and
`full` / `robot` / `environment` point clouds. The manifest references the
existing chunk PLY files, so stitching does not copy the point-cloud data.

To evaluate all 24 tasks in parallel:

```bash
for i in $(seq 0 23); do
    python evaluation/robocasa_client.py \
        --env_global_rank $i \
        --world_size 24 \
        --num_evals_per_worker 100 \
        --server_port 10086 \
        --save_root_dir ./eval_results/robocasa/ &
done
wait
```

#### RoboTwin 2.0

If you are using your own RoboTwin installation (not the submodule), modify `ROBOTWIN_ROOT` at the top of `evaluation/robotwin_client.py`:

```python
ROBOTWIN_ROOT = "/path/to/your/RoboTwin"
```

Then launch evaluation for each task:

```bash
python evaluation/robotwin_client.py \
    --task_name adjust_bottle \
    --task_config demo_randomized \
    --num_evals_per_worker 10 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robotwin/
```

To evaluate all 50 tasks:

```bash
TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horiz stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch)

for task in "${TASKS[@]}"; do
    python evaluation/robotwin_client.py \
        --task_name $task \
        --task_config demo_randomized \
        --num_evals_per_worker 100 \
        --server_port 10086 \
        --save_root_dir ./eval_results/robotwin/ &
done
wait
```

## Results

Evaluation results (success rates and rollout videos) are saved to the `--save_root_dir` directory.

## X-WAM RGB-D and 4D Capture (RoboCasa)

The normal evaluation path still stops denoising after actions are ready. Add
`--capture_4d` to one or a small number of RoboCasa clients when validating the
world-model output. The flag makes the policy server finish all RGB/depth
denoising steps and returns the structured future prediction to the client.

```bash
python evaluation/robocasa_client.py \
    --env_global_rank 0 \
    --world_size 1 \
    --num_evals_per_worker 1 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robocasa_4d \
    --capture_4d \
    --capture_stride 1 \
    --capture_fps 5 \
    --action_fps 20 \
    --point_stride 2
```

No extra policy-server flag is needed: `capture_4d` is sent with each request.
Capturing is much slower than policy-only evaluation because the server performs
all video denoising steps and runs the depth branch. Start with one client and
one rollout.

With the default `--capture_stride 1`, simulator RGB-D is captured after every
executed action. A full 32-action chunk therefore exports 33 ground-truth 4D
states: the initial state at action offset 0 plus one post-action state for each
offset 1 through 32. X-WAM prediction remains at 9 video states (5 Hz). The
dense simulator stream is timestamped at the controller rate (20 Hz) and stores
both `action_offsets` and `timestamps_s`; use `--capture_stride 4` to recover the
older 9-state, video-rate capture.

For every action chunk the client saves:

```text
<save_root>/<task>/<rank>_<rollout>_4d/chunks/step_000000/
├── metadata.json
├── predicted_rgbd.npz
├── ground_truth_rgbd.npz
├── predicted/
│   ├── rgb/<camera>.mp4
│   ├── cameras.json
│   ├── depth/<camera>/{frame_*.png,preview.mp4}       # checkpoint-native raw depth
│   ├── depth_metric/<camera>/{frame_*.png,preview.mp4}
│   └── pointclouds/{frame_*.ply,frame_*.npz,manifest.json}
└── ground_truth/
    ├── rgb/<camera>.mp4
    ├── cameras.json
    ├── depth/<camera>/{frame_*.png,preview.mp4}       # uint16 millimetres
    └── pointclouds/{frame_*.ply,frame_*.npz,manifest.json}
```

`ground_truth_rgbd.npz` contains RGB, metric depth, intrinsics, camera-to-world
and camera-to-base transforms for every captured simulator frame, together with
the corresponding action offsets and timestamps.
`predicted_rgbd.npz` contains RGB, raw generated depth, calibrated metric depth,
predicted camera poses, predicted proprioception and the executed nominal action.
All exported point clouds use the robot base frame and metres.

### Important depth note

The released RoboCasa depth target is a video without a metric-scale sidecar.
The capture code therefore always preserves `depth_raw` and, by default, fits an
inverse-depth affine transform against selected measured-depth pixels at frame 0
separately for every view. This same transform is applied to future predicted
frames. The
result is suitable for inspecting temporal 4D consistency, but it must not be
described as independently metric depth. Calibration coefficients are recorded
in `metadata.json`. Use `--pred_depth_representation metric` only for a future
checkpoint whose output is already known to be in metres.

### Blender

Blender can import each `.ply` directly. To create an animated `.blend` where
one point-cloud object is visible per timeline frame:

```bash
blender --background \
    --python evaluation/blender_import_4d.py -- \
    /path/to/pointclouds \
    /path/to/output.blend
```

Blender 4.x uses `bpy.ops.wm.ply_import`. No Blender Python package is installed
into the X-WAM environment.

### Additional packages

The repository requirements already list all capture dependencies:

- `numpy>=1.23.5,<2`
- `scipy>=1.13.1`
- `imageio[ffmpeg]` / `imageio-ffmpeg`

RoboSuite provides the camera calibration and depth-buffer conversion helpers.
No Open3D, PyTorch3D, Trimesh, or Blender add-on is required. On a minimal
machine used only to inspect/rebuild saved captures, install:

```bash
pip install "numpy>=1.23.5,<2" "scipy>=1.13.1" "imageio[ffmpeg]"
```

The lightweight geometry tests do not import Torch, MuJoCo, RoboCasa, or
RoboSuite:

```bash
python -m unittest tests.test_robocasa_4d
```
