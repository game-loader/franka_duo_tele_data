# Franka Duo 真机原始采数与模型推理

`franka-duo-tele-data` 是比赛现场专用工具，默认工作流是：

1. 将 8 路高频双臂流限频到最高 100 Hz，其余相机/夹爪/TF topic 由 ROS 2 rosbag2
   逐 episode 直接写入 MCAP；
2. 在离线机器上做时间同步、16D action/state 聚合、LeRobot v3 转换和 FK；
3. 在真机上加载已导出的 IL / offline RL bundle，推理 20D Cartesian action，同时保存
   eval 的原始 MCAP 证据。

本仓库不包含训练、仿真、Docker、底层 Franka 控制器、IK/轨迹执行器或 ARA。ROS 2、
ZED、RealSense、Franka 和 Robotiq 驱动均由真机主机提供。

## 为什么默认录原始 MCAP

除 8 路高频双臂 stream 的轻量 relay 外，现场采集进程不再自行订阅数据，也不解码、
同步或写 LeRobot。relay 输出与其余直录 topic 一起交给等价于下面的 rosbag2 命令：

```bash
ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --output EPISODE_PATH \
  TOPIC...
```

`storage=mcap` 和 `preset=zstd_fast` 在代码中固定，YAML 和 CLI 都不能改写。这样现场路径：

- 不解码或重新编码 RGB/depth；
- 不重采样 head 15 Hz、wrist 30 Hz、夹爪或 TF；
- 仅对 8 路双臂 stream 做“最新未转发消息、每 10 ms 最多一次”的 100 Hz 上限采样；
- 不在线匹配 RGB/depth/control；
- 不把左右臂/夹爪聚合成伪造的组合 `JointState`；
- 不运行 FK，也不依赖 LeRobot、PyTorch、Pinocchio 或 FFmpeg。

rosbag2 为每条已记录消息保存 bag receipt timestamp，消息 payload 内原有的
`header.stamp` 也保留，两者不是同一个概念。相机、夹爪和 TF 是直录，receipt timestamp
对应 rosbag2 收到原 publisher 消息的时间。8 路 arm 输出是 relay 的新 ROS publication：
rclpy 会做 typed 反序列化和重新序列化，但不解释或修改任何 message field，原
`header.stamp` 也保留；因此逻辑字段不变，但不承诺 CDR 序列化字节逐字相同。bag receipt
timestamp 是 relay 发布时刻，不是 source 到达 relay 的时刻；被更晚消息覆盖的高频样本不会
进入 bag。带 header 的流后处理时
优先使用 source `header.stamp`；没有 header 的夹爪 target `std_msgs/Float32` 只能使用直录
receipt timestamp。MCAP 本身不宣称不同 topic 已同步。

Python 的数据平面工作仅限 arm rate relay，不解码消息内容。它另行发布
`/franka_duo_tele_data/episode_event`，消息类型为
`std_msgs/msg/String`，JSON payload 在 MCAP 内标记 episode start。按下结束键时会先立即停止
rosbag2；end 边界、outcome 和可选 reward 随后写入 bag 外的 dataset/episode manifest。
这些元数据不代替传感器 timestamp，也不改变机器人原始消息。

## TMR 默认录制 Topic

[configs/tmr_mcap.yaml](configs/tmr_mcap.yaml) 是唯一默认采集配置：

| 数据 | 录制方式 / Topic |
|---|---|
| 8 路双臂流 | 只录下节列出的 `/franka_duo_tele_data/rate100/...` relay 输出 |
| 左右夹爪 target | `/left/gripper/gripper_client/target_gripper_width_percent`、右侧同名 topic |
| 左右夹爪 actual | `/left/gripper/joint_states`、`/right/gripper/joint_states` |
| ZED-M RGB | `/head_camera/zed/rgb/color/rect/image` |
| ZED-M registered depth | `/head_camera/zed/depth/depth_registered` |
| ZED-M CameraInfo | `/head_camera/zed/rgb/color/rect/camera_info` |
| 双 D405 RGB | `/wrist_camera_left/color/image_raw`、右侧同名 topic |
| 双 D405 CameraInfo | `/wrist_camera_left/color/camera_info`、右侧同名 topic |
| 可用 TF | `/tf`、`/tf_static` |

YAML 的 `mcap.topics` 恰好包含 21 条：8 条 arm relay 输出加 13 条相机、夹爪和 TF 直录
topic，绝不包含 8 条 arm source。manual recorder 再加入 episode event，共 22 条；eval
在相同 22 条基础上加入 `/franka_duo/eval/action_trace`，共 23 条。

### 双臂 100 Hz Relay 路由

`arm_sampling.rate_hz: 100` 和以下 8 条 source -> recorded route 是显式录制契约：

| Source topic | Recorded topic |
|---|---|
| `/left/franka_robot_state_broadcaster/current_pose` | `/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/current_pose` |
| `/left/franka_robot_state_broadcaster/desired_joint_states` | `/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/desired_joint_states` |
| `/left/franka_robot_state_broadcaster/measured_joint_states` | `/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/measured_joint_states` |
| `/left/franka_robot_state_broadcaster/desired_end_effector_twist` | `/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/desired_end_effector_twist` |
| `/right/franka_robot_state_broadcaster/current_pose` | `/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/current_pose` |
| `/right/franka_robot_state_broadcaster/desired_joint_states` | `/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/desired_joint_states` |
| `/right/franka_robot_state_broadcaster/measured_joint_states` | `/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/measured_joint_states` |
| `/right/franka_robot_state_broadcaster/desired_end_effector_twist` | `/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/desired_end_effector_twist` |

relay 每 10 ms 对每条 route 检查一次单元素缓冲：期间收到多条时只发布最新一条；没有新
消息时不发布，所以不会为了凑 100 Hz 重复 stale payload。它不插值、不修改 message field
或 `header.stamp`，但会重新序列化，而且这是有损限频，不能从 MCAP 恢复被覆盖的 source
samples 或原 source receipt time。source 侧使用 `best_effort/keep_last(1)` 只保留最新值；
recorded 输出侧使用 `reliable/keep_last(10)`，避免已选中的 100 Hz 样本在 relay 到 rosbag2
之间再次被 best-effort 无声丢弃。

D405 depth 明确不录。`/tf` 和 `/tf_static` 用于保留可能存在的离线坐标重建证据，但它们
不保证现场发布了完整且正确标定的 head-optical 到 robot-base 树。录后必须检查 bag 中的
TF；缺失时使用单独标定的静态外参，不能把 camera-frame 点云称为 base/world 点云。

当前 ZED 通常约 14-15 Hz，D405 driver 当前是 `640x480@30`。MCAP 保留各自原始频率，
不会把 wrist 降成“逻辑 15 FPS”。如果现场改 D405 为 `480x270@30`，只修改 driver；原始
recorder 不做尺寸假设，CameraInfo 和图像会原样进入 bag。

TMR 没有经过验证的周期 spine state/target topic，所以默认清单不录 spine。也不录 base
速度、里程计或 D405 depth。要扩大原始证据面，应先版本化修改 `tmr_mcap.yaml`，不要在
同一个 dataset version 中临时改变 topic 集合。

## 安装

原始 MCAP 与 native RL-100 eval 支持 Python 3.10+；LeRobot policy/后处理 extras 要求
Python 3.12+。另外需要 `uv`、ROS 2 rosbag2 和 MCAP storage plugin：

```bash
git clone git@github.com:game-loader/franka_duo_tele_data.git
cd franka_duo_tele_data

# 默认原始 MCAP recorder；兼容 RL-100 的 Python 3.10 / NumPy 1.23.5 环境
uv sync --extra record
```

ROS 依赖由系统安装，不从 PyPI 获取。以 Jazzy 为例，缺少 plugin 时安装对应发行版包：

```bash
sudo apt install ros-jazzy-rosbag2-storage-mcap
ros2 bag list storage
ros2 bag record --help | grep storage-preset-profile
ros2 interface show std_msgs/msg/String
```

recorder 启动时也会执行这些 preflight；找不到 `mcap` 或 preset 参数会立即失败。arm
relay 由 manual recorder 和 eval supervisor 自动启动并等待 ready，不需要另开常驻 relay
终端。它从 live ROS graph 发现每条 source 的消息类型；任一 source 缺失、超时或同时声明
多个类型（ambiguous）时整个 session 在 rosbag 启动前失败，不会猜类型或留下部分契约
bag。`franka-duo-arm-rate-relay` 是独立诊断入口；正常 manual/eval 不要再并行启动第二个
relay 实例。

其余 Python 依赖按阶段隔离：

```bash
# 原生 RL-100 bundle eval，仅额外安装 PyTorch
uv sync --extra eval

# LeRobot DP3 / pretrained_model eval
uv sync --extra eval --extra lerobot-policy
```

只有 `lerobot-policy` 会引入 LeRobot，并固定到已验证的
`game-loader/lerobot_droid@0bab8685280cf53f40f1a4a03b6ff213d7ab8b94`。默认 recorder
不安装 LeRobot writer、视频栈或训练依赖。

## ROS 主机环境

当前分工：`.50` 提供 TMR 与 ZED-M，`.100` 提供双 FR3、双夹爪和双 D405，`.101`
提供 GELLO 与脚踏板。录制进程必须运行在能看到所有目标 topic 的主机上。每个新终端：

```bash
source ~/tmr_env.sh
echo "ROS_DISTRO=$ROS_DISTRO"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
```

三个 `scripts/run_*.sh` 在当前 shell 未加载 ROS 时，会优先 source `~/tmr_env.sh`，否则
回退到 `/opt/ros/jazzy/setup.bash`。SSH 可达不等于 DDS 可见；换主机时必须复用现场
`ROS_DOMAIN_ID=0`、RMW、CycloneDDS interface/peer 和 overlay 配置。

## 现场 Preflight

先启动双 FR3 controller、Robotiq manager、ZED 和双 D405，再检查完整 topic/type。以下
数组明确覆盖所有 8 条 arm source；每条都必须有且仅有一个类型，并实际持续发布：

```bash
ros2 topic list -t | grep -E \
  'head_camera|wrist_camera|franka_robot_state_broadcaster|gripper|/tf'

arm_sources=(
  /left/franka_robot_state_broadcaster/current_pose
  /left/franka_robot_state_broadcaster/desired_joint_states
  /left/franka_robot_state_broadcaster/measured_joint_states
  /left/franka_robot_state_broadcaster/desired_end_effector_twist
  /right/franka_robot_state_broadcaster/current_pose
  /right/franka_robot_state_broadcaster/desired_joint_states
  /right/franka_robot_state_broadcaster/measured_joint_states
  /right/franka_robot_state_broadcaster/desired_end_effector_twist
)
for topic in "${arm_sources[@]}"; do
  ros2 topic info -v "$topic"
  timeout 6s ros2 topic hz "$topic" || true
done

ros2 topic info -v /left/gripper/gripper_client/target_gripper_width_percent
ros2 topic info -v /right/gripper/gripper_client/target_gripper_width_percent
ros2 topic echo /left/gripper/joint_states --once
ros2 topic echo /right/gripper/joint_states --once
```

确认 measured/desired joint stream 是 `sensor_msgs/msg/JointState`，每侧包含 7 个有限
position，且 header stamp 非零。`current_pose` 和 `desired_end_effector_twist` 的唯一实际
类型以 graph discovery 为准，不能凭 topic 名猜测。确认 gripper target 是 `[0,1]` 的
`std_msgs/msg/Float32`。actual 夹爪多 joint 选择和
闭合/打开端点不在原始录包阶段决定，但必须保存现场标定记录供离线 16D 聚合使用。

相机不能只看 topic 名，必须确认实际有数据：

```bash
ros2 topic hz /head_camera/zed/rgb/color/rect/image
ros2 topic hz /head_camera/zed/depth/depth_registered
ros2 topic echo /head_camera/zed/rgb/color/rect/camera_info --once

ros2 topic hz /wrist_camera_left/color/image_raw
ros2 topic hz /wrist_camera_right/color/image_raw
ros2 topic echo /wrist_camera_left/color/camera_info --once
ros2 topic echo /wrist_camera_right/color/camera_info --once

df -h datasets/franka_duo_mcap
```

ZED depth 必须是注册到 rectified RGB 像素坐标的 `depth_registered`。用现场画面核对 D405
serial 的物理左右映射；旧指南曾出现相反的 serial 文字记录。

启动 `run_manual_recorder.sh` 或 `run_eval.sh` 后，在另一终端核对 8 条 relay 输出。每条
输出应有与 source 相同的唯一 message type，活跃 source 的输出频率不得超过 100 Hz：

```bash
arm_outputs=(
  /franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/current_pose
  /franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/desired_joint_states
  /franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/measured_joint_states
  /franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/desired_end_effector_twist
  /franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/current_pose
  /franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/desired_joint_states
  /franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/measured_joint_states
  /franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/desired_end_effector_twist
)
for topic in "${arm_outputs[@]}"; do
  ros2 topic info -v "$topic"
  timeout 6s ros2 topic hz "$topic" || true
done
```

## 原始 MCAP Episode 采集

普通采集不要求 reward：

```bash
./scripts/run_recorder.sh --max-episodes 50
```

人工 reward 采集：

```bash
./scripts/run_manual_recorder.sh --max-episodes 50
```

manual 流程会自动启动并监督 8 路 arm relay，确认全部 route ready 后才允许 rosbag2 使用
21 条 YAML topic 开录。两个脚本都调用同一个 `franka-duo-mcap-record`：空闲时 `r` 开始；
录制中 `e` 或 `s`
结束保存，`d` 丢弃，`q` 丢弃并退出。按 `e/s` 后会先发送 SIGINT，等待 rosbag2 完成 MCAP
封包，再由 manual 脚本恢复终端行输入并要求有限标量 reward；relay 会保持就绪以便复用，
但输入 reward 期间 rosbag2 已关闭，不会再向本 episode 写入任何新消息。reward 只写入
episode/dataset manifest，不写进 MCAP 或任何机器人消息。输入后回到空闲状态，再按 `r`
即可开始下一个 episode。

默认输出：

```text
datasets/franka_duo_mcap/franka_duo_tmr_raw_vN/
  mcap_dataset_manifest.json
  episode_000000/
    metadata.yaml
    *.mcap
    episode_manifest.json
  episode_000001/
    ...
```

每次启动分配新的 `_vN`，不会 resume/追加旧 dataset。Ctrl-C 或异常终止的 bag 会尽量
保留并标记 `incomplete`，不得直接进入训练转换。录完立即检查：

```bash
ros2 bag info datasets/franka_duo_mcap/franka_duo_tmr_raw_v1/episode_000000
```

核对 storage id、duration、每个配置 topic 的 message count、`episode_event` start、manifest
中的 end/reward，以及 `/tf_static` 是否真的有消息。配置列出 topic 不代表 publisher 一定
存在；rosbag2 允许某个
topic 最终 count 为零，正式数据必须在离线验收时拒绝这种 episode。

可用环境变量切换版本化配置：

```bash
FRANKA_MCAP_CONFIG="$PWD/configs/my_site_mcap.yaml" ./scripts/run_recorder.sh
```

## 离线 LeRobot v3 后处理契约

原始 MCAP 不是 LeRobot dataset。转换必须在采集后完成，建议不可变保留 raw bag，并将
派生 LeRobot v3 写到另一目录。**当前仓库尚未提供 MCAP -> LeRobot v3 转换命令**；
转换器将在后续作为独立的离线工作实现，且必须遵循以下顺序：

1. 读取 rosbag receipt timestamp 和原消息 `header.stamp`，验证每个 required topic 数量、
   时间单调性、消息类型、尺寸和 CameraInfo；
2. 用 `episode_event` 验证 start，用 manifest 验证 end/outcome/reward，不把元数据时间当图像时间；
3. 用新的 head RGB header stamp 作为目标帧，按明确阈值匹配 registered depth、左右 wrist、
   rate100 relay 中的 measured/desired arm；arm 使用保留的 source header stamp，headerless
   gripper target 使用直录 receipt timestamp；
4. 生成下述 16D action/state，记录每个派生帧对应的所有 source timestamp 和 skew；
5. 编码三路 RGB，保存与 head RGB 一一对应的深度/标定 sidecar，写 LeRobot v3；
6. 严格验收派生集后，再使用 URDF/mount transform 离线 FK。

目标 16D 契约是后处理结果，不是 MCAP 内在线生成的 topic：

```text
action float32[16]
[0:7]    left arm desired joint positions
[7:14]   right arm desired joint positions
[14]     left gripper target open fraction, [0,1]
[15]     right gripper target open fraction, [0,1]

observation.state float32[16]
[0:7]    left arm measured joint positions
[7:14]   right arm measured joint positions
[14]     left gripper actual calibrated open fraction, [0,1]
[15]     right gripper actual calibrated open fraction, [0,1]
```

转换器必须显式记录同步策略、阈值、drop/missing 统计、夹爪 joint 选择与两侧各自标定，不得
用 measured state 冒充 action，也不得为 spine/base/world EE 填零。双臂 EE 位姿应在后处理
阶段用真实 URDF 和静态 mount transform 做 FK；除非整条外参链已校准并验证，否则不能称为
world pose。

## 模型 Bundle 与 20D Action

Eval 在线读取 ZED RGB/depth 生成 manifest 规定的 XYZ 或 XYZRGB 点云，并读取双 D405
RGB。点数、3/6 通道、空间裁剪、外参、random/FPS 采样必须与训练 bundle 一致。模型输出：

```text
[0:9]    left EE:  xyz + rotation matrix first two rows (rot6d_rows)
[9:18]   right EE: xyz + rotation matrix first two rows (rot6d_rows)
[18]     left gripper open fraction, [0,1]
[19]     right gripper open fraction, [0,1]
```

所有 bundle 必须包含 `manifest.json`，明确 `backend`、20D `action_spec`、workspace、点云
预处理、双腕 image key、可选 state key 和 backend 加载配置。`rl100_native` 的
`model.pt`/`encoder.pt` 不是自描述文件；factory 必须加载训练时相同的 Hydra config、
normalizer、scheduler 和 shape metadata。factory 是可执行 Python，只加载可信 bundle。

```bash
uv run --extra eval franka-duo-export-bundle \
  --checkpoint /path/to/rl100/checkpoint \
  --output /path/to/franka_eval_bundle \
  --factory my_policy.factory:load --python-root policy_code \
  --num-points 512 --channels 3 --sampling fps \
  --workspace-min=-0.8,-0.8,0.0 \
  --workspace-max=0.8,0.8,1.5
```

## Eval、自动 MCAP 与安全门

`run_eval.sh` 默认在模型运行期间同时保存 8 路 rate100 arm 输出和其余直录 TMR topic，
并把每次推理的观测 source stamp、耗时和 20D action 作为
`/franka_duo/eval/action_trace` 写入同一 MCAP。JSONL 只是可选副本；MCAP 是强制的现场
provenance，不能关闭。

`configs/tmr_eval.yaml` 的 `mcap.config` 相对 eval YAML 解析，默认引用 `tmr_mcap.yaml`；
`dataset_name` 为 `franka_duo_tmr_eval`，输出根目录继承 raw config 的
`datasets/franka_duo_mcap`。每次 eval 创建一个新的 `_vN` 和一个自动 episode，MCAP 在
eval ROS node 与推理循环之前启动。eval supervisor 也会自动启动同一 8 路 arm relay，只有
全部 source/type 唯一并且输出 ready 后才启动 bag；21 条 YAML topic、episode event 和
action trace 共 23 条进入同一个固定 `mcap/zstd_fast` bag。

第一步始终 dry-run 单帧；模型 action 会打印，但不会发到机器人 relay：

```bash
./scripts/run_eval.sh /path/to/franka_eval_bundle --device cuda --once

FRANKA_LEROBOT_POLICY=1 \
  ./scripts/run_eval.sh /path/to/lerobot_bundle --device cuda --once

./scripts/run_eval.sh --checkpoint /path/to/checkpoint \
  --manifest /path/to/manifest.json --device cuda --once
```

需要把有限 episode reward 写入 eval manifest 时，使用 `--prompt-reward`；也可用
`--reward VALUE` 做非交互测试。eval 完成最后一步后会先停止并封包 MCAP，再提示输入 reward，
因此输入耗时和期间的传感器消息不会进入该 episode：

```bash
./scripts/run_eval.sh /path/to/franka_eval_bundle --device cuda \
  --max-steps 300 --prompt-reward
```

可用参数覆盖自动录包位置和结束 reward：

```bash
./scripts/run_eval.sh /path/to/franka_eval_bundle --device cuda --max-steps 300 \
  --mcap-output-root /data/eval_mcap \
  --mcap-dataset-name final_site_eval \
  --reward 1.0

# 或在 eval 正常完成时交互输入 reward
./scripts/run_eval.sh /path/to/franka_eval_bundle --device cuda --max-steps 300 \
  --prompt-reward
```

`--reward` 与 `--prompt-reward` 互斥。也可用 `--mcap-config PATH` 或环境变量
`FRANKA_MCAP_CONFIG` 替换 MCAP/arm relay 配置。既不提供 reward 参数时，正常 episode 保存
`reward: null`，不会等待终端输入。

`--once` 或达到 `--max-steps` 才是正常完成并标记 complete。推理/ROS 异常和 Ctrl-C 会
保留 bag 但标记 incomplete；因此正式定长评测应设置 `--max-steps`，不要依靠 Ctrl-C 作为
成功 episode 终点。

正式发布必须同时满足：manifest 有 workspace bounds；输入同步、新鲜度、推理超时、finite
和 rot6d 检查通过；现场 relay 已实现 IK、插值/限速、限位、碰撞、watchdog 和急停；低速
单步核对左右臂/旋转/夹爪索引；最后才同时提供两个开关：

```bash
./scripts/run_eval.sh /path/to/franka_eval_bundle --device cuda \
  --publish --enable-robot
```

即使打开开关，工具也只向 `/franka_duo/policy_action` 发布
`std_msgs/Float32MultiArray.data[20]`，不是 Franka controller 原生命令。本仓库不实现
安全 relay，不能改 topic 绕过它。eval trace topic 必须与 command topic 不同。

### Stateful eval 限制

当前 eval reader 只接受聚合的 `/franka_duo/semantic_joint_states`，本仓库尚未提供该
publisher。因此纯点云 + 双腕 RGB bundle 可直接 eval；声明 `state_key` 的 bundle 必须
先由现场 relay 发布与训练一致的 16D state，并填写夹爪标定。raw MCAP 虽保存了分流 topic，
不会在 eval 进程内自动聚合 state。启用该 relay 后，还必须把
`/franka_duo/semantic_joint_states` 加入 eval 使用的 MCAP topic 配置，否则 provenance
preflight 会拒绝启动。现有 14D state-only 或输出非 20D 的 checkpoint 不能直接控制
Franka Duo。

## 开发验证

```bash
bash -n scripts/*.sh
python -c 'import tomllib; tomllib.load(open("pyproject.toml", "rb"))'
uv run --extra dev ruff check src tests
```

硬件测试需 source ROS 环境。单元测试不得发布机器人动作。
