# Franka Duo Left PTP Step

This is a one-shot diagnostic tool. It:

1. reads the left `current_pose` and measured joint state;
2. moves the mounted-tool pose by `offset_m` along the Franka-local base `+X`
   axis;
3. converts that local pose through the MoveIt `left_fr3v2_link0` mounting
   transform;
4. converts the tool target to the MoveIt `left_fr3v2_link8` tip frame using
   the fixed `tool_offset_z_m` (default `0.174 m`);
5. solves `left_arm` IK with the official Duo MoveIt KDL configuration;
6. prints the seven joint targets;
7. sends one `franka_msgs/action/PTPMotion` goal only when both execution gates are enabled.

The IK solution must stay within `max_joint_delta_rad` of the measured joint
seed. This prevents KDL from selecting a different 7-DOF branch for a small
Cartesian step.

Build on the robot host:

```bash
cd ~/franka_duo_tele_data
source /opt/ros/jazzy/setup.bash
colcon build \
  --base-paths site \
  --build-base site/build \
  --install-base site/install \
  --merge-install \
  --packages-select franka_duo_ptp_step
source site/install/setup.bash
```

Dry-run:

```bash
export ROS_DOMAIN_ID=0
ros2 launch franka_duo_ptp_step left_ptp_step.launch.py
```

Execute one left-arm PTP target:

```bash
ros2 launch franka_duo_ptp_step left_ptp_step.launch.py \
  offset_m:=0.01 \
  execute:=true \
  confirm:=true
```

The PTP action receives seven joint positions in radians, seven maximum joint
velocities in radians per second, and one scalar goal tolerance in radians.
Wait for the action result before sending another goal. Do not run this while
another command source is actively commanding the left arm.

## Episode 0 dual-arm target

`duo_ptp_episode.launch.py` reads an 18D action trajectory from the extracted
episode artifact. It converts each midpoint-frame left and right EE pose into
the respective local arm-base frames using the mirrored training transforms,
removes the fixed `0.174 m` tool offset for the MoveIt `link8` tips, solves
`left_arm` and `right_arm` IK, and sends both PTP goals concurrently for each
frame. The next frame is sent only after both previous PTP goals finish.

Dry-run:

```bash
ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py \
  action_file:=/path/to/franka_duo_tmr_lerobot_v6_episode_000000_first_1s_action.json
```

Execute:

```bash
ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py \
  action_file:=/path/to/franka_duo_tmr_lerobot_v6_episode_000000_actions.json \
  action_start_index:=0 \
  action_end_index:=848 \
  log_file:=/path/to/franka_duo_tmr_lerobot_v6_episode_000000_ptp.csv \
  execute:=true \
  confirm:=true
```

The CSV log contains one row per dual-arm PTP frame, including send time,
goal-acceptance time, result time, inter-frame interval, result status, and
IK joint deltas. The node prints the average command rate and interval
statistics after a successful run.

## Joint trajectory streaming

The same node can resample each 16-action IK chunk to one complete 51-point,
50 Hz action trajectory and publish it to a site-owned relay. Each message
also contains one measured-state anchor point at time zero; the 51 action
points begin after the configurable bridge interval. The relay is the only
process that forwards commands to `JointTrajectoryController` topics. The
next 16-action chunk is published only after the previous complete trajectory
has reached its horizon.

First stop or keep inactive both `joint_impedance_controller` instances, then
load the trajectory controllers:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run controller_manager spawner joint_trajectory_controller \
  --controller-manager /left/controller_manager \
  --param-file ~/franka_duo_tele_data_capture/site/install/share/franka_duo_ptp_step/config/left_joint_trajectory_controller.yaml
ros2 run controller_manager spawner joint_trajectory_controller \
  --controller-manager /right/controller_manager \
  --param-file ~/franka_duo_tele_data_capture/site/install/share/franka_duo_ptp_step/config/right_joint_trajectory_controller.yaml
```

Start the relay in dry-run mode first. It drops all robot output by default:

```bash
ros2 launch franka_duo_ptp_step jtc_command_relay.launch.py \
  enable_robot:=false
```

Verify these topics exist before execution:

```bash
ros2 topic list | grep joint_trajectory
```

Dry-run the full IK and 50 Hz resampling without publishing:

```bash
ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py \
  mode:=jtc \
  action_file:=/path/to/franka_duo_tmr_lerobot_v6_episode_000000_actions.json \
  action_start_index:=0 \
  action_end_index:=848 \
  execute:=false \
  confirm:=false
```

Execute the stream only after both trajectory controllers are active and the
robot safety state is clear. Enable the relay separately:

```bash
ros2 launch franka_duo_ptp_step jtc_command_relay.launch.py \
  enable_robot:=true
```

Then run the gated stream:

```bash
ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py \
  mode:=jtc \
  action_file:=/path/to/franka_duo_tmr_lerobot_v6_episode_000000_actions.json \
  action_start_index:=0 \
  action_end_index:=848 \
  stream_rate_hz:=50 \
  chunk_size:=16 \
  stream_log_file:=/path/to/franka_duo_jtc_chunk_stream.csv \
  execute:=true \
  confirm:=true
```

The stream node publishes only to `/franka_duo/eval/{left,right}/joint_trajectory`;
the relay validates joint names, dimensions, timing, and finite values before
forwarding to the controller command topics.
