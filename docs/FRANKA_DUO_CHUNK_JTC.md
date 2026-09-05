# Franka Duo Mobile 真机 Action Chunk 控制（已实现方案）

本文档描述仓库**已经实现**的一次下发一个 action chunk 的控制链路，以及对 franka_ros2
(jazzy) 官方 `franka_mobile_fr3_duo_moveit_config` 的核对结论。`docs/` 下其余文档
（SERL / CRISP / MoveIt Servo）是早期方案调研，依赖的第三方 controller 包并未随官方
发布，仅供参考；其中关节名已统一修正为 `left_fr3v2_joint1..7` / `right_fr3v2_joint1..7`。

## 官方核对结论

- 官方 `franka_ros2` jazzy 分支**没有** MoveIt Servo 包，也没有任何 Servo 配置。
- `franka_mobile_fr3_duo_moveit_config/launch/moveit.launch.py` 启动**一个**
  `ros2_control_node`（`robot_types: [tmrv0_2, fr3v2, fr3v2]`，前缀 `'' / left / right`），
  参数文件 `config/mobile_fr3_duo_controllers.yaml`，激活 `joint_state_broadcaster`、
  `swerve_drive_controller`、`full_body_controller`。
- `full_body_controller` 类型 `mobile_fr3_duo_trajectory_controller/MobileFR3DuoTrajectoryController`，
  是**全身**控制器（planar_x/y/theta + 左右各 7 关节），手臂走 effort 接口做关节阻抗
  （`k_gains [24,24,24,24,10,6,2]`，`d_gains [2,2,2,1,1,1,0.5]`），
  **只接受** `FollowJointTrajectory` action（`<controller>/follow_joint_trajectory`），没有 topic 输入。
- MoveIt 运动学：`kinematics.yaml` 对 `left_arm` / `right_arm` 使用 KDL；
  `joint_limits.yaml` 默认速度/加速度缩放 0.1，关节最大速度 2.62 (j1–4)、5.26 (j5,j7)、4.18 (j6) rad/s。
- 结论：官方栈可以用 MoveIt 的 `robot_description` + KDL 做 IK，但没有现成的
  “连续 chunk 流式执行”接口。本仓库在 site 侧已有的每臂 `joint_trajectory_controller`
  + `jtc_command_relay` 之上补齐了这条链路。

## 数据流

```text
franka-duo-eval (control_mode: chunk)
  └─ PolicyBundle.predict_chunk -> [horizon x 20]（DP3 完整 chunk，不再截断第一行）
  └─ 每行 validate + to_link0_action，写 trace（schema franka_duo_eval_action_chunk_v1）
  └─ --publish --enable-robot 时发布 Float32MultiArray 到 /franka_duo/policy_action_chunk
       layout.dim = [horizon, 20]，layout.data_offset = 本次打算执行的步数
  └─ 睡眠 chunk_execute_steps / fps 后再取下一帧观测

policy_chunk_jtc_stream (site/franka_duo_ptp_step)
  └─ 订阅 chunk；用测量关节状态做 seed，逐行 MoveIt KDL IK（consistency limit）
  └─ 15 Hz 关节路径 -> 50 Hz 线性重采样 + 有限差分速度/加速度
  └─ 首点为当前测量状态 (t=0)，其余点从 jtc_initial_delay_s 开始
  └─ 速度守卫 max_joint_velocity_rad_s；超限整包丢弃
  └─ execute && confirm 时发布 JointTrajectory 到 /franka_duo/eval/{left,right}/joint_trajectory
  └─ enable_gripper 时发布 Float32 到 /{left,right}/gripper/gripper_client/target_gripper_width_percent

jtc_command_relay (enable_robot:=true)
  └─ 校验关节名/维度/时间单调/finite -> /{left,right}/joint_trajectory_controller/joint_trajectory

joint_trajectory_controller (position, interpolate_from_desired_state, splines)
```

新 chunk 到来时 JTC 用新轨迹替换正在执行的轨迹，因此只要评测器在上一 chunk 结束前
发出下一 chunk（`chunk_execute_steps < horizon`），手臂就不会停下，这就是 PTP 方案
不连续问题的解决点。

## 真机启动顺序

```bash
# 0. 编译 site 包（机器人主机）
cd ~/franka_duo_tele_data && source /opt/ros/jazzy/setup.bash
colcon build --base-paths site --build-base site/build --install-base site/install \
  --merge-install --packages-select franka_duo_ptp_step
source site/install/setup.bash

# 1. 双臂 bringup 后，停用阻抗控制器，加载每臂 JTC
ros2 control set_controller_state -c /left/controller_manager joint_impedance_controller inactive
ros2 control set_controller_state -c /right/controller_manager joint_impedance_controller inactive
ros2 run controller_manager spawner joint_trajectory_controller -c /left/controller_manager \
  --param-file $(ros2 pkg prefix franka_duo_ptp_step)/share/franka_duo_ptp_step/config/left_joint_trajectory_controller.yaml
ros2 run controller_manager spawner joint_trajectory_controller -c /right/controller_manager \
  --param-file $(ros2 pkg prefix franka_duo_ptp_step)/share/franka_duo_ptp_step/config/right_joint_trajectory_controller.yaml

# 2. relay（先 dry-run，确认 topic 后再 enable_robot:=true）
ros2 launch franka_duo_ptp_step jtc_command_relay.launch.py enable_robot:=false

# 3. chunk 执行器（dry-run：只做 IK 并打印，不发布）
ros2 launch franka_duo_ptp_step policy_chunk_jtc_stream.launch.py input_action_rate_hz:=15.0

# 4. 评测器 dry-run（configs/tmr_eval.yaml 已设 control_mode: chunk）
./scripts/run_eval.sh /path/to/bundle --device cuda --max-steps 20

# 5. 全部核对后才打开开关
ros2 launch franka_duo_ptp_step jtc_command_relay.launch.py enable_robot:=true
ros2 launch franka_duo_ptp_step policy_chunk_jtc_stream.launch.py execute:=true confirm:=true enable_gripper:=true
./scripts/run_eval.sh /path/to/bundle --device cuda --max-steps 300 --publish --enable-robot
```

## 关键参数

| 位置 | 参数 | 说明 |
| --- | --- | --- |
| tmr_eval.yaml | `control_mode` | `single` 或 `chunk` |
| tmr_eval.yaml | `chunk_topic` | 必须不同于 `trace_topic` |
| tmr_eval.yaml | `chunk_execute_steps` | 每 chunk 执行步数，应小于 horizon 以保证重叠 |
| tmr_eval.yaml | `chunk_inference_timeout_ms` | chunk 推理超时（扩散模型较慢） |
| launch | `input_action_rate_hz` | 必须等于评测器 `fps` |
| launch | `action_frame` | `link0`（评测器已用 manifest 变换）或 `midpoint` |
| launch | `max_joint_velocity_rad_s` | 关节速度守卫，超限丢弃 chunk |
| launch | `max_joint_delta_rad` | IK 相邻解最大跳变，防 7-DOF 分支切换 |

## 替代路径（未实现）

- **官方 `full_body_controller`**：可以把同样的 JointTrajectory 通过
  `FollowJointTrajectory` action 发给 `full_body_controller/follow_joint_trajectory`
  （需带 planar_x/y/theta 三个基座关节并保持为 0）。但 action 接口要等 goal 结束或
  取消才能替换，不如 JTC topic 输入适合流式替换。
- **MoveIt Servo**：需自行引入 `moveit_servo`，配置 `command_out_type:
  trajectory_msgs/JointTrajectory` 输出到同一 JTC；官方未提供配置，见
  `MOVEIT_SERVO_INTEGRATION.md`。

## 本地未验证项

本工作站没有 ROS 2 / MoveIt / `franka_mobile_fr3_duo_moveit_config`，C++ 节点与 launch
只做了静态检查，需在机器人主机 `colcon build` 后再 dry-run 验证。
