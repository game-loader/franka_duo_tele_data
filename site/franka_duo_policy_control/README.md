# Franka Duo Policy Control

这是现场 overlay 的独立 ROS 2 package。它只新增 site-owned relay 和两个
Cartesian pose controller，不修改现场已有驱动、controller 配置或主仓库
`src/franka_duo_tele_data`。

## 数据路径

```text
/franka_duo/policy_action  std_msgs/msg/Float32MultiArray[20]
        |
        v
policy_action_relay
  20D 解包、rot6d_rows、有限值检查
  训练时 midpoint frame -> 左右各自 arm-base frame
  gripper open fraction [0,1] -> target_gripper_width_percent [0,1]
        |
        +--> /franka_duo/policy/left/target_pose
        +--> /franka_duo/policy/right/target_pose
        +--> /left/gripper/gripper_client/target_gripper_width_percent
        +--> /right/gripper/gripper_client/target_gripper_width_percent
                         |
                         v
        PolicyCartesianPoseController (每臂，常驻 active)
        官方 FrankaCartesianPoseInterface
        1 kHz pose command，RealtimeBuffer 接收 30 Hz target
        left_0..left_15/cartesian_pose_command
        right_0..right_15/cartesian_pose_command
```

模型 action 的布局必须保持仓库的 20D contract：

```text
[0:9]   left xyz + rot6d_rows
[9:18]  right xyz + rot6d_rows
[18]    left gripper open fraction
[19]    right gripper open fraction
```

夹爪沿用现场已有的 `target_gripper_width_percent`，不是
`control_msgs/action/GripperCommand`。现场已验证该 topic 的值域是 0..1：
`0.0` 闭合，`1.0` 打开。relay 只在 `enable_robot=true` 时发布位姿目标，
只在 `enable_gripper=true` 时发布夹爪目标；两个 gate 可以独立打开。

## 固定训练坐标转换

RGB20D runtime 会根据数据集 manifest 先转换到 link0，因此使用独立的
`rgb20d_relay.launch.py`（`action_frame=link0`）跳过此处的固定转换。
旧入口默认仍是 `action_frame=midpoint`。两种模式不可叠加转换。
控制器默认 `target_timeout_s=0.25`，目标停止更新后按既有加速度/jerk 限制减速。
完整使用说明见主仓库 `docs/RGB20D_REPLAY.md`。

不在现场 YAML 中配置 arm-base 矩阵。矩阵固定来自采数/训练使用的
`mobile_fr3_duo_v0_2.usd`，遵循列向量约定：

```text
T_midpoint_from_ee = make_transform(policy_xyz, policy_rot)
T_arm_from_ee =
    inverse(T_midpoint_from_arm_base) * T_midpoint_from_ee
```

左右 `T_midpoint_from_arm_base` 的平移分别是 `y=+0.05018 m` 和
`y=-0.05018 m`，并保留 USD 中左右 link0 的镜像旋转。controller 收到的是
各自 arm-base frame 下的 `PoseStamped`，不需要 IK。

relay 输出的 pose target topic 是 `geometry_msgs/msg/PoseStamped`：
`/franka_duo/policy/left/target_pose` 和
`/franka_duo/policy/right/target_pose`。它们是 eval action 与现场
controller 之间的 site-owned topic，不是 Franka 底层 command interface。

默认 pose controller 内部订阅 pose target topic，在 1 kHz 循环内用
`FrankaCartesianPoseInterface` 写入 quaternion + translation。30 Hz target
通过连续速度、加速度和 jerk 的 Cartesian servo 跟踪，避免首拍命令产生
FCI 的 velocity/acceleration discontinuity；不在线计算 IK、Jacobian 或 torque。

## 构建和启动

在 ROS 2 Jazzy 现场环境中：

```bash
cd /path/to/franka_duo_tele_data
source /opt/ros/jazzy/setup.bash
colcon build \
  --base-paths site \
  --build-base site/build \
  --install-base site/install \
  --merge-install \
  --packages-select franka_duo_policy_control
source site/install/setup.bash
ros2 launch franka_duo_policy_control policy_control.launch.py
```

launch 会启动 relay，加载左右 policy controller，并在硬件恢复为 `active` 后自动
停用现场的 `joint_impedance_controller`、激活 policy controller。现场自带 controller
仍由原 arm launch 管理，但不会继续 claim Cartesian pose 接口。默认两个安全开关都是
`false`，夹爪 gate 也是 `false`。

现场现在是两个 per-arm manager：`/left/controller_manager` 和
`/right/controller_manager`。自定义 controller plugin 必须在这两个
`ros2_control_node` 进程启动前已经出现在它们的 `AMENT_PREFIX_PATH` 中；只在
spawner shell 里 source 这个 overlay 不够。若 controller load 日志出现
`Loader ... not found`，需要用包含本 overlay 的环境重启现场 arm launch，例如：

```bash
cd ~/franka_duo_policy_overlay
source ~/tmr_env.sh
source install/setup.bash
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=tmr_duo_config.yaml
```

这一步会重启左右 arm driver/controller manager，应只在现场确认机器人状态安全、
可以短暂停控制器时执行。

启动后会按顺序加载并激活 policy controller，同时停用两侧
`joint_impedance_controller`。`joint_impedance_controller` 可以保持关闭；
现场只保留本 overlay 的两个 policy pose controller active。如果硬件此前因通信
异常回到 `unconfigured`，launch 会先尝试恢复对应 Franka hardware component，
再进行 controller 切换。

启动后确认：

```bash
ros2 control list_controllers --controller-manager /left/controller_manager
ros2 control list_controllers --controller-manager /right/controller_manager
ros2 control list_hardware_interfaces --controller-manager /left/controller_manager | rg 'cartesian_pose_command'
ros2 control list_hardware_interfaces --controller-manager /right/controller_manager | rg 'cartesian_pose_command'
ros2 topic echo /franka_duo/policy/left/target_pose --once
```

默认 launch 会加载两个 policy pose controller，非严格停用现有
`joint_impedance_controller`，再严格激活 policy controller：

```bash
ros2 control switch_controllers \
  --controller-manager /left/controller_manager \
  --deactivate joint_impedance_controller \
  || true
ros2 control switch_controllers \
  --controller-manager /left/controller_manager \
  --strict \
  --activate left_policy_cartesian_pose_controller

ros2 control switch_controllers \
  --controller-manager /right/controller_manager \
  --deactivate joint_impedance_controller \
  || true
ros2 control switch_controllers \
  --controller-manager /right/controller_manager \
  --strict \
  --activate right_policy_cartesian_pose_controller
```

不要同时激活 `joint_impedance_controller` 和 policy pose controller。policy
controller claim 同一臂的 16 个 `cartesian_pose_command` 接口。policy controller
的默认参数已经是
`allow_motion: true`；relay 和 eval 仍保持 dry-run 默认，只有显式打开
`enable_robot` 与发布参数时才会向现场 target topic 输出动作。

测试右臂 1 cm 时，不直接发布 16 个 Cartesian interface。先让右侧
pose controller 激活并保持 `allow_motion: true`，再向
`/franka_duo/policy/right/target_pose` 发布当前姿态附近的
`PoseStamped`。控制器会在 1 kHz 循环内按 `filter_params` 平滑跟踪。

## 配置边界

`policy_control.example.yaml` 是默认示例。允许现场覆盖 topic 名称和
controller manager 名称，但不提供 arm-base 矩阵参数，避免现场配置与训练坐标
约定漂移。relay 不做 workspace、反馈新鲜度或跳变边界检查；只检查 20D 长度、
有限数、rot6d 可构成旋转、夹爪值在 `[0,1]`。eval 进程只发布到
site-owned relay 的 `/franka_duo/policy_action`，不直接写硬件接口。
