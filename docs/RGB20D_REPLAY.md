# RGB20D Black-Box Replay

This path uses the RGB-only dataset contract. The older DP3 evaluator in
`eval_franka_duo.py` still expects point clouds and 34D state and is not this
entry point. Camera and robot drivers stay host-managed.

## Interface

`ChunkRequest(start_step, observation)` is the model boundary.
`observation.policy_input()` contains:

- `observation.images.camera1`: head, uint8 RGB HWC, original size.
- `observation.images.camera2`: left wrist, uint8 RGB HWC, original size.
- `observation.images.camera3`: right wrist, uint8 RGB HWC, original size.
- `observation.state`: float32[20], left xyz + rot6d rows, right xyz + rot6d rows,
  left/right binary grippers (0 closed, 1 open).

The source dimensions, gripper calibration and arm mounting transforms come
from the dataset metadata and `franka_duo_extras/derived_manifest.json`.
Live observations use head timestamps, the same nearest-neighbor tolerances
and lookahead as conversion, and fresh left/right poses and gripper states.
Measured joints and depth are not policy inputs. RGB HWC is the transport
boundary; the eventual server preprocessor must handle CHW and normalization.

The black box returns `ActionChunk(start_step, observation_stamp_ns, actions)`
with float32[H,20]. Row zero targets the next timestep after the observation,
as in the dataset. Values must already be denormalized to meters and rot6d
rows. `DatasetEpisodePolicy.predict_chunk()` currently supplies the recorded
actions; a future server adapter can implement this same method.

Chunks retain absolute step indices. Late rows are discarded, out-of-order
responses are rejected, and only future targets are replaced. Inference runs
in a worker while a separate thread captures observations. A buffer underrun,
stale feedback, or missed deadline aborts publication instead of replaying old
actions or sending a burst of catch-up commands.

## Coordinates And Control

Input: `T_midpoint_EE = T_midpoint_arm @ T_arm_EE`.
Output: `T_arm_EE = inverse(T_midpoint_arm) @ T_midpoint_EE`.
The full mirrored rotation and translation are applied, not just the 5 cm arm
offset. The runtime converts once using the manifest; the dedicated relay
must use `action_frame=link0` and must not convert again. The Cartesian
interface uses the recorded Franka EE frame directly. Do not apply the
0.174 m tool offset used by the separate MoveIt link8 IK path.

The runtime sends targets at 30 Hz to `/franka_duo/rgb20d/action`. Slower replay
holds scheduled targets between source steps while maintaining the 30 Hz
heartbeat. The site relay publishes per-arm PoseStamped targets to the
existing `PolicyCartesianPoseController`. Its 1 kHz loop preserves its
position, orientation, velocity and acceleration across chunk boundaries using
the host's Ruckig library. A pose-error velocity request passes through Ruckig's
jerk-limited velocity interface. Per-axis limits reserve braking headroom and
bound the vector norm; there is no hard velocity clamp on the output. Position
comes from the integrated trajectory; rotation uses integrated spatial angular
increments. The 1 kHz command clock does not scale increments by host scheduling
jitter. The existing kp/kd ratio sets the pose-error velocity gain.
When target delivery stops for 250 ms,
it brakes using those limits rather than chasing an old target indefinitely.
This is a tracking servo, not an exact time-parameterized replay or a collision
planner; actual tracking must be assessed on the robot.

The historical `policy_chunk_jtc_stream` is a different execution path. It
anchors each replacement trajectory at a measured pose with zero boundary
velocity. It must not be used to infer continuity of this RGB20D path.

## Offline Check On The Capture Host

```bash
cd /home/aup/franka_duo_tele_data_infer_20260905
bash scripts/run_rgb20d_replay.sh \
  --dataset datasets/franka_duo_lerobot_rgb20d_v1 --all-episodes
```

This decodes all three videos, checks task-independent 20D inputs and binary
grippers, replays every recorded action through chunk buffering, verifies
next-frame targets and the coordinate round trip, and writes
`outputs/rgb20d_blackbox_report.json`. It publishes no ROS commands. It does
not simulate or validate robot dynamics.

## Live Preparation

Build the site package in the same ROS environment used by the drivers:

```bash
source ~/tmr_env.sh
colcon build --base-paths site/franka_duo_policy_control \
  --build-base site/build --install-base site/install --merge-install \
  --packages-select franka_duo_policy_control
source site/install/setup.bash
```

Both existing controller_manager processes must have this overlay in their
startup environment. Source it before the site-owned arm launch; sourcing it
only in the spawner shell cannot update a running plugin loader. Controllers
must be stationary before switching. Do not auto-recover a robot fault.

When the site's arm drivers are already running, load both policy controllers
inactive using the existing parameter file, then switch each manager strictly:

```bash
ros2 param set /left/controller_manager left_policy_cartesian_pose_controller.type \
  franka_duo_policy_control/PolicyCartesianPoseController
ros2 param set /right/controller_manager right_policy_cartesian_pose_controller.type \
  franka_duo_policy_control/PolicyCartesianPoseController
ros2 run controller_manager spawner left_policy_cartesian_pose_controller \
  --inactive -c /left/controller_manager \
  --controller-ros-args "--ros-args --params-file $PWD/site/franka_duo_policy_control/config/rgb20d_smoke.yaml"
ros2 run controller_manager spawner right_policy_cartesian_pose_controller \
  --inactive -c /right/controller_manager \
  --controller-ros-args "--ros-args --params-file $PWD/site/franka_duo_policy_control/config/rgb20d_smoke.yaml"
ros2 control switch_controllers -c /left/controller_manager --strict \
  --deactivate joint_impedance_controller --activate left_policy_cartesian_pose_controller
ros2 control switch_controllers -c /right/controller_manager --strict \
  --deactivate joint_impedance_controller --activate right_policy_cartesian_pose_controller
```

These switch commands assume the current command controller is the site's
`joint_impedance_controller`; inspect `ros2 control list_controllers` first.
If only state broadcasters are active after PTP, omit `--deactivate`.
Pass the parameter file through `--controller-ros-args` because this controller
reads its limits and interface prefix during initialization, before configure.
Stop other policy/teleoperation command sources. Keep the current pose close
to the selected recorded starting pose. The runtime refuses an initial
translation error over 3 cm or rotation error over 0.2 rad. It does not move
the robot to the start pose automatically. Workspace and tracking limits are
explicit in `configs/tmr_rgb20d.yaml` and must match the physical scene.

## Live Input And Replay

With cameras and robot state available, the live dry-run reads real observations
and exercises recorded action chunks without commands. It does not record MCAP:

```bash
bash scripts/run_rgb20d_replay.sh \
  --dataset datasets/franka_duo_lerobot_rgb20d_v1 --live \
  --episode 0 --start-index 0 --max-steps 60 --speed 0.1 \
  --snapshot-input outputs/rgb20d_input.npz
```

Start the dedicated relay separately. This launch does not load controllers,
switch command interfaces or change hardware state:

```bash
ros2 launch franka_duo_policy_control rgb20d_relay.launch.py \
  enable_robot:=true enable_gripper:=true
```

Then add both `--publish --enable-robot` to the live command for actual motion.
Default replay is 0.1x speed and 60 frames of one episode. Feedback continues
at the head camera rate and targets at 30 Hz. Increase speed only after checking
tracking. The runtime requires the expected relay and both Cartesian
controllers with their watchdogs enabled; it will not switch or restart them.

Add `--record-mcap` only when raw evaluation recording is needed. Its supervised
100 Hz arm relay and raw camera/TF recording keep the capture contract intact.
Inference itself reads driver topics and does not need rosbag2 or a recorder
process. Optional MCAP traces include source stamps, state, original midpoint
action, converted link0 action, dataset frame and publication time.

## Recorded Actions On The Robot

This diagnostic mode reads real paired arm poses and gripper feedback but
does not subscribe to cameras or construct a model RGB request. It sends the
same recorded action chunks through the same buffer and site relay:

```bash
bash scripts/run_rgb20d_replay.sh \
  --dataset datasets/franka_duo_lerobot_rgb20d_v1 \
  --live --recorded-actions --episode 0 --max-steps 60 --speed 0.1 \
  --publish --enable-robot
```

Before that, export exactly the selected first action for the existing PTP tool:

```bash
bash scripts/run_rgb20d_replay.sh \
  --dataset datasets/franka_duo_lerobot_rgb20d_v1 --episode 0 --start-index 0 \
  --export-ptp-target outputs/rgb20d_start_action.json
ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py \
  action_file:="$PWD/outputs/rgb20d_start_action.json" action_index:=0 \
  max_joint_velocity:=0.05 wait_timeout_s:=60.0 execute:=true confirm:=true
```

Run PTP only with both arm command controllers inactive. Wait for both PTP
results before activating the Cartesian controllers. The separate PTP tool
uses link8 IK and its existing 0.174 m tool offset; the runtime does not.
For initial motion checks load `config/rgb20d_smoke.yaml`: 0.05 m/s, 0.10 m/s2,
0.5 m/s3 linear bounds and 0.3 rad/s, 0.5 rad/s2, 2.0 rad/s3 angular bounds.

Interrupting stops publication and allows the controller watchdog to brake.
When recording is enabled it also preserves incomplete MCAP provenance.
The robot's physical stop remains the mechanism for an immediate hardware stop.
