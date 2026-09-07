# 目标机双臂驱动与控制方法现状

核查时间：2026-09-05 22:02-22:05，Asia/Shanghai。

本文记录目标机的实际运行状态、已经使用的控制接口和实验结果。文中的“代码已实现”“离线验证通过”和“真机运行成功”分别说明，不相互替代。核查完成后没有重新启动双臂运动，也没有安装 MoveIt Servo。

## 1. 当前结论

| 项目 | 当前状态 |
| --- | --- |
| 左右机械臂 driver | 均已退出，当前没有 `/left/controller_manager`、`/right/controller_manager` 进程 |
| 双臂 Cartesian 控制器 | 已停用，所属 driver 已退出 |
| RGB20D 动作发布程序、Cartesian relay | 已停止 |
| 左右 Robotiq 夹爪 driver | 仍在运行，分别有独立 controller manager |
| PTP 回到数据集第一条 action | 真机成功，两臂均返回 `TARGET_REACHED` |
| Cartesian + Ruckig 连续动作回放 | 真机失败：右臂关节超速及关节速度、加速度不连续，触发 reflex |
| 现有 KDL IK + JTC chunk 路径 | 代码和二进制已存在；历史窗口替换方式存在不连续问题，本次没有重新做真机测试 |
| MoveIt 2 Servo + JTC | 用户提出的下一步方向；当前没有安装 Servo，没有完成该链路配置或真机验证 |
| 模型推理服务器 | 本次未接入，动作来自已经采集的 RGB20D 数据集 |
| MCAP 录制 | 本次连续动作测试未录制；RGB20D 入口已改为默认不录制，显式 `--record-mcap` 才启用 |

右臂最后一次已知硬件状态是 reflex 后被 driver 停用。driver 退出后，本次没有重新读取机器人内部故障状态，不能据此声称已经复位。没有自动调用 error recovery。

## 2. 机器与软件环境

| 项目 | 值 |
| --- | --- |
| 采集/控制机 | `aup@172.16.0.100` |
| 本次工作目录 | `/home/aup/franka_duo_tele_data_infer_20260905` |
| 操作系统 | Ubuntu 24.04.4 LTS |
| 内核 | `6.17.0-1032-oem`；driver 报告不是实时内核 |
| ROS | ROS 2 Jazzy，`ROS_DOMAIN_ID=0` |
| DDS | `rmw_cyclonedds_cpp`；`CYCLONEDDS_URI=file:///home/aup/cyclonedds.xml` |
| 环境入口 | `/home/aup/tmr_env.sh`，内部加载 `/home/aup/recloned_sources/source_migrated_stack.sh` |
| Python | 工作目录现有 `.venv/bin/python`，不在本机执行数据转换或真机推理 |
| MoveIt Core / ROS Planning | `2.12.4` |
| 系统 JTC | `ros-jazzy-joint-trajectory-controller`，`4.40.1` |
| 系统 Ruckig | `ros-jazzy-ruckig`，`0.9.2` |
| MoveIt Servo | `ros2 pkg prefix moveit_servo` 返回 `Package not found`；未安装 |

实际 `controller_manager` 和 `franka_hardware` 来自下面的源码工作空间 overlay，不能仅根据系统安装包版本判断全部运行行为：

```text
/home/aup/recloned_sources/franka_ros2_jazzy_ws/install/controller_manager
/home/aup/recloned_sources/franka_ros2_jazzy_ws/install/franka_hardware
/home/aup/recloned_sources/teleoperation_overlay/install/franka_fr3_arm_controllers
/home/aup/recloned_sources/teleoperation_overlay/install/franka_gripper_manager
```

机器人通信网卡为 `enp193s0`，地址 `172.16.16.100/24`；采集机管理网络为 `enp194s0`，地址 `172.16.0.100/24`。

## 3. 双臂 driver 架构

### 3.1 左右映射

| 项目 | 左臂 | 右臂 |
| --- | --- | --- |
| Franka IP | `172.16.16.12` | `172.16.16.11` |
| 型号 / arm_id | `fr3v2` | `fr3v2` |
| namespace / arm_prefix | `left` | `right` |
| controller manager | `/left/controller_manager` | `/right/controller_manager` |
| 关节名称 | `left_fr3v2_joint1` 至 `joint7` | `right_fr3v2_joint1` 至 `joint7` |
| 本地基座坐标 | `left_fr3v2_link0` | `right_fr3v2_link0` |
| PTP action | `/left/action_server/ptp_motion` | `/right/action_server/ptp_motion` |

IP 与左右侧映射已经由现场配置确认，不应按上游示例交换 `.11` 和 `.12`。

### 3.2 现场配置和启动方式

现场配置文件：

```text
/home/aup/recloned_sources/teleoperation_overlay/install/
  franka_fr3_arm_controllers/share/franka_fr3_arm_controllers/
    config/tmr_duo_config.yaml
    config/controllers.yaml
    launch/franka_fr3_arm_controllers.launch.py
    launch/franka.launch.py
```

通常的双臂遥操作启动入口是：

```bash
source ~/tmr_env.sh
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=tmr_duo_config.yaml
```

该入口会为每臂启动独立 driver，并尝试激活 `joint_impedance_controller`。它不是只读启动。

本次 PTP 和 Cartesian 测试改为分别启动 `franka.launch.py`，保留真实硬件、状态广播，初始不加载遥操作命令控制器。例如左臂：

```bash
source ~/tmr_env.sh
source /home/aup/franka_duo_tele_data_infer_20260905/site/install/setup.bash
ros2 launch franka_fr3_arm_controllers franka.launch.py \
  arm_id:=fr3v2 arm_prefix:=left namespace:=left \
  robot_ip:=172.16.16.12 load_gripper:=false joint_sources:=joint_states
```

右臂对应替换为 `right` 和 `172.16.16.11`。`load_gripper:=false` 是因为现场使用单独驱动的 Robotiq，而非 Franka Hand。

每臂 ros2_control 循环为 1000 Hz。测试中成功获得 FIFO 优先级 50，但日志存在周期 overrun；不能把“配置 1000 Hz”理解为“每个周期均无抖动”。

### 3.3 反馈接口

两臂使用相同的后缀：

```text
/{left,right}/franka_robot_state_broadcaster/current_pose
/{left,right}/franka_robot_state_broadcaster/measured_joint_states
```

`current_pose` 是 Franka 当前 EE 在对应 arm base 下的位姿。测量关节可用于 IK seed、关节限位检查及控制反馈；不能把测量关节冒充模型输出的期望 action。

### 3.4 连接问题与恢复过程

1. 初次检查时，左臂虽然可以 ping，但 libfranka 报 `UDP receive: Timeout`；右臂连接成功。
2. 结束旧会话、单独重连左臂后，报 `Connection to FCI refused ... enable FCI mode in Desk`。
3. 用户确认现场就绪后，两臂均成功连接 FCI，PTP action server 和状态广播正常。
4. 后续连续回放右臂触发 reflex，动作程序因反馈过期退出；左臂控制器随后停用，最终两臂 driver 和 Cartesian relay 均退出。

## 4. 已采用或检查过的控制方法

| 方法 | 下发接口 | 完成情况 / 结论 |
| --- | --- | --- |
| 现场关节阻抗遥操作 | `sensor_msgs/JointState` → `/{side}/gello/joint_states` → `JointImpedanceController` | 现场已有；本次回放没有使用 |
| KDL IK + 单次 PTP | `franka_msgs/action/PTPMotion`，每臂 7 个目标关节 | 已真机验证到达 RGB20D 首条 action，适合起点定位 |
| 逐帧重复 PTP | 同上，每帧等上一个 goal 结束 | 已有工具支持；逐段起停，不适合作为 30 Hz 连续策略执行器 |
| KDL IK + JTC chunk | `JointTrajectory` → site relay → 每臂 JTC topic | 代码已实现；历史 chunk 重置边界造成不连续，本次未重新验证 |
| Cartesian Pose servo | `Float32MultiArray[20]` → site relay → `PoseStamped` → Franka Cartesian Pose Interface | 已实现并真机尝试；即使加入 Ruckig，仍触发关节侧保护，用户已否定该方案 |
| 官方 Duo 全身轨迹控制 | `full_body_controller/follow_joint_trajectory` | 已检查配置和源码；本次未启动、未用于执行 |
| MoveIt 2 Servo + JTC | 拟由 Servo 生成关节轨迹，经 site relay 进入 JTC | 下一步候选；尚未实现和验证 |

### 4.1 现场 JointImpedanceController

类型为 `franka_fr3_arm_controllers/JointImpedanceController`。输入目标关节位置，内部以 1 kHz 计算关节阻抗力矩并通过 effort 接口下发。

当前 `controllers.yaml` 参数：

```yaml
k_alpha: 0.99
k_gains: [240, 240, 240, 240, 100, 60, 20]
d_gains: [20, 20, 20, 10, 10, 10, 5]
```

它要求关节目标。20D 末端 action 不能直接发到 `gello/joint_states`，需要明确的运动学转换和连续关节目标生成。

### 4.2 单次 PTP：本次成功的真机结果

实现：`site/franka_duo_ptp_step/src/duo_ptp_episode.cpp`。

处理步骤为：数据集 midpoint 末端目标 → 对应 link0 → MoveIt 模型坐标 → 去除 link8 到工具的偏移 → KDL IK → 关节边界和 seed 跳变检查 → 双臂 PTP。

PTP goal 包含 `goal_joint_configuration`、`maximum_joint_velocities` 和 `goal_tolerance`。两臂同时发送、分别返回，均完成后才结束该条动作；这不表示双臂共享同一条严格同步时间轨迹。

本次数据源为 `datasets/franka_duo_lerobot_rgb20d_v1`，episode 0、`action[0]`。速度上限 `0.05 rad/s`，`execute=true`、`confirm=true`，结果如下：

| 指标 | 左臂 | 右臂 |
| --- | --- | --- |
| IK 相对当前 seed 的最大关节变化 | 0.85434 rad | 0.17194 rad |
| PTP result code | 4，SUCCEEDED | 4，SUCCEEDED |
| target status | 2，TARGET_REACHED | 2，TARGET_REACHED |
| 从发送到结果返回 | 17.136 s | 3.486 s |
| PTP 后相对首 action 的位置误差 | 约 0.123 mm | 约 0.037 mm |

姿态误差在当时 float32 距离计算中输出为 0，不代表无限精度的完全一致。双臂整条 PTP 总耗时约 17.136 s。

目标机证据文件，均相对于本次工作目录：

```text
outputs/rgb20d_start_action.json
outputs/rgb20d_start_ptp.csv
```

### 4.3 KDL IK + JTC：已有实现及历史问题

已有源文件：

```text
site/franka_duo_ptp_step/src/duo_ptp_episode.cpp          # mode=jtc
site/franka_duo_ptp_step/src/policy_chunk_jtc_stream.cpp
site/franka_duo_ptp_step/src/jtc_command_relay.cpp
site/franka_duo_ptp_step/config/left_joint_trajectory_controller.yaml
site/franka_duo_ptp_step/config/right_joint_trajectory_controller.yaml
```

上述节点的可执行文件已存在于目标机 `site/install/lib/franka_duo_ptp_step/`。

这条路径使用 KDL 把 action chunk 转成关节路径，进行重采样，再经现场 relay 下发：

```text
/franka_duo/eval/{left,right}/joint_trajectory
  -> jtc_command_relay
  -> /{left,right}/joint_trajectory_controller/joint_trajectory
```

JTC 配置为 position command interface，position/velocity state interfaces，`interpolate_from_desired_state=true`，`interpolation_method=splines`。

历史实验的问题是：新窗口从测量位置、零边界速度重新起步，时间从窗口起点重置。连续替换轨迹并不自动保证位置、速度、加速度在边界连续，曾触发 Franka 的速度/加速度不连续保护。本次没有再次执行这条 JTC 真机路径。

因此旧文档中“只要在上一个 chunk 结束前替换，新旧 chunk 就一定连续”的说法不能作为已验证结论。新的 Servo/JTC 方案必须重新核对时间戳、轨迹替换以及实际 JTC 输出的导数。

### 4.4 Cartesian + Ruckig：本次实际失败

实际链路：

```text
20D midpoint action，30 Hz
  -> Python 转换一次到左右 link0
  -> /franka_duo/rgb20d/action
  -> franka_duo_rgb20d_relay
  -> /franka_duo/policy/{left,right}/target_pose
  -> PolicyCartesianPoseController，1 kHz
  -> FrankaCartesianPoseInterface::setCommand()
  -> libfranka Cartesian pose command
```

早期控制器使用自行实现的 PD、速度/加速度/jerk 限幅。之后改为系统 Ruckig 的 velocity 模式，保留控制器内部速度和加速度状态；250 ms 无新目标时，目标速度归零并按限制减速。

本次测试参数：

| 项目 | 值 |
| --- | --- |
| 计划测试范围 | episode 0 前 60 帧，0.1 倍速 |
| action 发送频率 | 30 Hz；慢放时保持目标，维持 heartbeat |
| 线速度 / 加速度 / jerk 上限 | 0.05 m/s、0.10 m/s²、0.5 m/s³ |
| 角速度 / 加速度 / jerk 上限 | 0.3 rad/s、0.5 rad/s²、2.0 rad/s³ |
| 起点位置检查上限 | 0.03 m |
| 运行位置跟踪检查上限 | 0.06 m |
| 输入反馈过期上限 | 200 ms |
| MCAP | 未开启 |

C++ 坐标测试 6 项通过；新增 Ruckig 测试 3 项通过，覆盖目标反转、旋转轴变化、输出导数及刹停。这些测试只验证软件生成的笛卡尔指令，不验证 Franka 内部逆运动学、冗余自由度或真实关节动力学。

2026-09-05 21:53:20，右臂实际故障为：

```text
libfranka: Move command aborted: motion aborted by reflex!
[
  "joint_velocity_violation",
  "cartesian_motion_generator_joint_velocity_discontinuity",
  "cartesian_motion_generator_joint_acceleration_discontinuity"
]
```

右臂硬件及其状态广播被停用，回放程序随后报 `TimeoutError: live RGB/state feedback expired` 并停止发布。左臂控制器随后被显式停用。

**本次没有完成前 60 帧。** 未启用逐步持久化 trace，异常退出也没有输出最终帧计数，因此不能给出准确的已执行帧数或成功完整回放的结论。

已确认的结论是：限制 EE 空间的速度、加速度和 jerk，并不能保证 Franka 内部生成的关节轨迹满足关节限制。用户观察到了异常关节运动。是否由某个具体奇异位形、冗余分支变化或其他因素触发，本次证据不足以确定，不能把其中任何一项写成已定位的根因。

原始日志：

```text
/home/aup/.ros/log/ros2_control_node_456175_1788615529640.log
/home/aup/.ros/log/ros2_control_node_455998_1788615528317.log
```

### 4.5 官方 Duo 配置与 MoveIt Servo + JTC

现场已有 `franka_mobile_fr3_duo_moveit_config`，源目录是：

```text
/home/aup/recloned_sources/franka_ros2_jazzy_ws/src/franka_mobile_fr3_duo_moveit_config
```

已核对：

- `get_robot_descriptions()` 提供统一的 `mobile_fr3_duo_v0_2` URDF/SRDF。
- `kinematics.yaml` 对 `left_arm` 和 `right_arm` 使用 `KDLKinematicsPlugin`。
- 本次 IK solver 的 base 为 `franka_spine_mounting_point`，tip 分别为左右 `fr3v2_link8`。
- `joint_limits.yaml` 给出每臂 j1-4 最大速度 2.62 rad/s，j5/j7 为 5.26 rad/s，j6 为 4.18 rad/s，关节最大加速度 3.75 rad/s²；这些是模型配置值，不是本次测试速度。
- 默认规划速度和加速度缩放系数为 0.1。不能假设 Servo 会自动使用规划请求的缩放系数，需按实际 Servo 版本配置验证。

官方 `mobile_fr3_duo_controllers.yaml` 定义 `full_body_controller`，类型是 `mobile_fr3_duo_trajectory_controller/MobileFR3DuoTrajectoryController`。它管理 planar_x/y/theta 及双臂 14 个关节，手臂采用 effort 关节阻抗；源码建立 `FollowJointTrajectory` action server。这与本次现场“左右独立 controller manager + 每臂 JTC”的架构不同，不能把官方全身启动文件直接叠加到已经占用 FCI 的双臂 driver 上。

用户要求考虑的下一条链路是：

```text
20D EE action chunk
  -> 保持原坐标合同，处理 EE 与 link8/tool 的区别
  -> MoveIt 2 Servo，复用 Duo 模型、关节限制和实时关节状态
  -> 连续 JointTrajectory
  -> site-owned relay
  -> 左右独立 JTC
  -> Franka 关节控制接口
```

这条组合在接口上可行，但截至本次核查尚未安装 Servo，未编写并验证版本匹配的 Servo 配置，也未解决和验证新轨迹进入 JTC 时的连续性。不能宣称它已经解决本次故障。

## 5. 夹爪 driver 与接口

目前运行的是：

```bash
ros2 launch franka_gripper_manager robotiq_gripper_controller_client.launch.py \
  config_file:=example_fr3_duo_config_robotiq.yaml
```

| 项目 | 左夹爪 | 右夹爪 |
| --- | --- | --- |
| namespace | `/left/gripper` | `/right/gripper` |
| 串口 ID | `usb-FTDI_USB_TO_RS-485_DAANVRU5-if00-port0` | `usb-FTDI_USB_TO_RS-485_DAANTK6Q-if00-port0` |
| 当前设备 | `/dev/ttyUSB0` | `/dev/ttyUSB1` |
| 状态 | `/left/gripper/joint_states` | `/right/gripper/joint_states` |
| site 客户端输入 | `/left/gripper/gripper_client/target_gripper_width_percent` | `/right/gripper/gripper_client/target_gripper_width_percent` |
| 底层 action | `/left/gripper/robotiq_gripper_controller/gripper_cmd` | `/right/gripper/robotiq_gripper_controller/gripper_cmd` |

两侧的 `joint_state_broadcaster_robotiq`、`robotiq_activation_controller`、`robotiq_gripper_controller` 当前均为 active，夹爪 manager 配置频率为 500 Hz。

策略侧使用 `0=闭合，1=打开` 的二值状态。现场 `robotiq_gripper_client.py` 接收 Float32 开合比例，再发送 `GripperCommand`；当前源码使用 `command.position = 1 - gripper_position`。应使用已核对的现场客户端合同，不能擅自把底层 action 数值当作米单位开口宽度再套一层换算。

本次读到夹爪位置为 0，即打开；其 effort 字段为 NaN。20D state 只提取位置并二值化，不包含该 effort 字段。

## 6. RGB20D 数据与坐标合同

目标机数据集：

```text
/home/aup/franka_duo_tele_data_infer_20260905/datasets/franka_duo_lerobot_rgb20d_v1
```

既有离线报告记录 24 个 episodes、9861 帧。`action[t] = state[t+1]` 的片内关系、三路 RGB 和 midpoint/link0 坐标往返已验证；这不构成机械臂动态执行验证。

state/action 的维度排列：

```text
[0:3]   左 EE xyz
[3:9]   左 EE rot6d，旋转矩阵前两行
[9:12]  右 EE xyz
[12:18] 右 EE rot6d，旋转矩阵前两行
[18]    左夹爪二值状态
[19]    右夹爪二值状态
```

state 不包含关节角和 spine height。动作是下一帧绝对末端位姿和夹爪状态，不是末端增量。

三路图像按头部时间戳对齐，数据集为 30 FPS，使用原始 RGB HWC，模型侧再处理 CHW。当前报告的实际图像尺寸是：

| 数据集图像 | 模型输入名 | HWC |
| --- | --- | --- |
| head | `observation.images.camera1` | `[360, 640, 3]` |
| wrist_left | `observation.images.camera2` | `[270, 480, 3]` |
| wrist_right | `observation.images.camera3` | `[270, 480, 3]` |

头部当前实际是 360×640，不能把早期示例中的 720×1280 当成本数据集的真实尺寸。深度和点云不进入这条 RGB 模型输入链路。

坐标转换以 `franka_duo_extras/derived_manifest.json` 为准：

```text
观测：T_midpoint_EE = T_midpoint_arm_base * T_arm_base_EE
动作：T_arm_base_EE = inverse(T_midpoint_arm_base) * T_midpoint_EE
```

必须使用完整的左右安装旋转和平移，不能只加减约 5 cm。左右安装方向不同。现有 RGB20D runtime 已转换为 link0，后级 relay 不应再次转换。

`0.174 m` 工具偏移只属于当前 MoveIt link8 IK 工具路径；直接 Franka EE Cartesian 路径未应用该偏移。后续 Servo 必须先确认它实际控制的 tip，不能照搬任一路径的偏移处理。

## 7. 代码入口、验证边界与后续衔接

相对于本次工作目录：

| 文件 | 用途 |
| --- | --- |
| `src/franka_duo_tele_data/rgb20d_io.py` | RGB20D 输入、坐标合同、无相机实时反馈读取 |
| `src/franka_duo_tele_data/cartesian_chunk.py` | action chunk 绝对索引、迟到数据裁剪、跳变和缓存检查 |
| `src/franka_duo_tele_data/replay_rgb20d.py` | 数据集黑盒回放、首帧 PTP 目标导出、可选 MCAP |
| `scripts/run_rgb20d_replay.sh` | 使用目标机现有 ROS 和 `.venv` 的入口 |
| `configs/tmr_rgb20d.yaml` | 实时话题、工作空间及反馈检查参数 |
| `site/franka_duo_policy_control/` | 本次已否定的 Cartesian relay / 控制器实验实现，代码仍保留 |
| `site/franka_duo_ptp_step/` | 已有 PTP、KDL IK、旧 JTC executor / relay |
| `outputs/rgb20d_blackbox_report.json` | 全数据集离线接口验证报告，不含真机动态结论 |

最近针对 RGB20D 输入与 chunk 的 Python 测试为 9 passed、1 skipped；跳过项要求隔离 ROS domain。此前完成的隔离 domain 221 relay 测试验证了动作消息和坐标输出，没有连接机械臂。

当前 `--live --recorded-actions` 可以读取真实双臂位姿/夹爪、从数据集取 action，不要求相机或模型服务器；加上 `--publish --enable-robot` 会走现有 Cartesian 路径，**它尚未被替换为 Servo/JTC，不能把该命令视作新方案入口**。

下一步若继续实施，先完成 Servo/JTC 的版本匹配和关节轨迹连续性检查，再处理现场复位与 PTP 起点定位。控制进程需明确区分 PTP、Cartesian、JTC，避免同臂同时占用接口。默认 dry-run，真实发布保留显式开关并经过 site-owned relay。

旧的 [FRANKA_DUO_CHUNK_JTC.md](FRANKA_DUO_CHUNK_JTC.md)、[MOVEIT_SERVO_INTEGRATION.md](MOVEIT_SERVO_INTEGRATION.md) 包含历史方案描述；其中“窗口替换保证连续”等说法以及示例 Servo 参数不能当作当前机器已验证的事实。本文的运行快照和上述原始日志用于说明截至核查时实际完成了什么。
