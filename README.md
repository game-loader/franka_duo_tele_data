# Franka Duo 真机原始采数与模型推理

`franka-duo-tele-data` 是比赛现场专用工具，默认工作流是：

1. 将 4 路双臂流和 2 路夹爪状态流限频到最高 100 Hz，其余相机/TF topic 由 ROS 2 rosbag2
   逐 episode 直接写入 MCAP；
2. 在离线机器上做时间同步、版本化 action/state 构造、LeRobot v3 转换和 FK；
3. 在真机上加载已导出的 IL / offline RL bundle，推理 20D Cartesian action，同时保存
   eval 的原始 MCAP 证据。

本仓库不包含训练、仿真、Docker、底层 Franka 控制器、IK/轨迹执行器或 ARA。ROS 2、
ZED、RealSense、Franka 和 Robotiq 驱动均由真机主机提供。

## 为什么默认录原始 MCAP

除 6 路高频双臂/夹爪状态 stream 的轻量 relay 外，现场采集进程不再自行订阅数据，也不解码、
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
- 仅对 4 路双臂和 2 路夹爪状态 stream 做“最新未转发消息、每 10 ms 最多一次”的 100 Hz
  上限采样；
- 不在线匹配 RGB/depth/control；
- 不把左右臂/夹爪聚合成伪造的组合 `JointState`；
- 不运行 FK，也不依赖 LeRobot、PyTorch、Pinocchio 或 FFmpeg。

rosbag2 为每条已记录消息保存 bag receipt timestamp，消息 payload 内原有的
`header.stamp` 也保留，两者不是同一个概念。相机和 TF 是直录，receipt timestamp 对应
rosbag2 收到原 publisher 消息的时间。6 路高频状态输出是 relay 的新 ROS publication：
rclpy 会做 typed 反序列化和重新序列化，但不解释或修改任何 message field，原
`header.stamp` 也保留；因此逻辑字段不变，但不承诺 CDR 序列化字节逐字相同。bag receipt
timestamp 是 relay 发布时刻，不是 source 到达 relay 的时刻；被更晚消息覆盖的高频样本不会
进入 bag。带 header 的流后处理时优先使用 source `header.stamp`；如果夹爪消息缺少 header，
只能使用 relay receipt timestamp，即 relay 发布时间。MCAP 本身不宣称不同 topic 已同步。

Python 的数据平面工作仅限 arm rate relay，不解码消息内容。它另行发布
`/franka_duo_tele_data/episode_event`，消息类型为
`std_msgs/msg/String`，JSON payload 在 MCAP 内标记 episode start。按下结束键时会先立即停止
rosbag2；end 边界、outcome 和可选 reward 随后写入 bag 外的 dataset/episode manifest。
这些元数据不代替传感器 timestamp，也不改变机器人原始消息。

## TMR 默认录制 Topic

[configs/tmr_mcap.yaml](configs/tmr_mcap.yaml) 是唯一默认采集配置：

| 数据 | 录制方式 / Topic |
|---|---|
| 4 路双臂流 | 只录下节列出的 `/franka_duo_tele_data/rate100/...` relay 输出 |
| 左右夹爪实际状态 | 只录 `/franka_duo_tele_data/rate100/left/gripper/joint_states` 和右侧同名输出 |
| ZED-M RGB | `/head_camera/zed/rgb/color/rect/image` (`640x360`) |
| ZED-M registered depth | `/head_camera/zed/depth/depth_registered` (`640x360`) |
| ZED-M CameraInfo | `/head_camera/zed/rgb/color/rect/camera_info` |
| 双 D405 RGB | `/wrist_camera_left/color/image_raw`、右侧同名 topic (`480x270@30`) |
| 双 D405 CameraInfo | `/wrist_camera_left/color/camera_info`、右侧同名 topic |
| 可用 TF | `/tf`、`/tf_static` |

YAML 的 `mcap.topics` 恰好包含 15 条：6 条高频状态 relay 输出加 9 条相机和 TF 直录
topic，绝不包含 6 条高频 source。manual recorder 再加入 episode event，共 16 条；eval
在相同 16 条基础上加入 `/franka_duo/eval/action_trace`，共 17 条。

### 双臂 100 Hz Relay 路由

`arm_sampling.rate_hz: 100` 和以下 6 条 source -> recorded route 是显式录制契约：

| Source topic | Recorded topic |
|---|---|
| `/left/franka_robot_state_broadcaster/current_pose` | `/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/current_pose` |
| `/left/franka_robot_state_broadcaster/measured_joint_states` | `/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/measured_joint_states` |
| `/right/franka_robot_state_broadcaster/current_pose` | `/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/current_pose` |
| `/right/franka_robot_state_broadcaster/measured_joint_states` | `/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/measured_joint_states` |
| `/left/gripper/joint_states` | `/franka_duo_tele_data/rate100/left/gripper/joint_states` |
| `/right/gripper/joint_states` | `/franka_duo_tele_data/rate100/right/gripper/joint_states` |

relay 每 10 ms 对每条 route 检查一次单元素缓冲：期间收到多条时只发布最新一条；没有新
消息时不发布，所以不会为了凑 100 Hz 重复 stale payload。它不插值、不修改 message field
或 `header.stamp`，但会重新序列化，而且这是有损限频，不能从 MCAP 恢复被覆盖的 source
samples 或原 source receipt time。source 侧使用 `best_effort/keep_last(1)` 只保留最新值；
recorded 输出侧使用 `reliable/keep_last(10)`，避免已选中的 100 Hz 样本在 relay 到 rosbag2
之间再次被 best-effort 无声丢弃。

D405 depth 明确不录。`/tf` 和 `/tf_static` 用于保留可能存在的离线坐标重建证据，但它们
不保证现场发布了完整且正确标定的 head-optical 到 robot-base 树。录后必须检查 bag 中的
TF；缺失时使用单独标定的静态外参，不能把 camera-frame 点云称为 base/world 点云。

当前 ZED RGB 和 registered depth 配置为 `640x360@15`；D405 driver 的目标采集 profile 为
`480x270@30`。MCAP 保留各自原始频率，不会把 wrist 降成“逻辑 15 FPS”。尺寸和 profile
必须在相机 driver 端设置，原始 recorder 不做尺寸假设，CameraInfo 和图像会原样进入 bag。
这个 profile 变化应从旧的 `franka_duo_tmr_raw_v1` 作为新数据版本验收，不能覆盖旧 bag。

### D405 主机端 Profile

仓库脚本不会启动或重配置 RealSense 节点。请在现场已有的 `launch_d405_duo.sh` 或等价的
`rs_multi_camera_launch.py` 调用中，保留实际 serial/name/namespace 映射，并设置下面四个
参数：

```text
depth_module.color_profile1:=480x270x30
depth_module.color_profile2:=480x270x30
enable_depth1:=false
enable_depth2:=false
```

Jazzy 多相机 launch 的参数检查可能把这些参数误报为“不支持”；以实际启动日志和 topic
为准。开始录制前确认尺寸、编码和频率：

```bash
ros2 topic echo /wrist_camera_left/color/image_raw --once | grep -E 'height|width|encoding|step'
ros2 topic echo /wrist_camera_right/color/image_raw --once | grep -E 'height|width|encoding|step'
ros2 topic hz /wrist_camera_left/color/image_raw
ros2 topic hz /wrist_camera_right/color/image_raw
```

### 图像压缩边界

`zstd_fast` 只压缩 MCAP chunk；raw 录制不会在线解码或重新编码相机消息。JPEG 是有损的，
可选方案如下：

| 方案 | 是否无损 | 适用范围和限制 |
|---|---|---|
| PNG | 是 | RGB 可直接无损；对噪声图像仍可能较大。深度通常要求 `16UC1`。 |
| WebP lossless | 是 | RGB 可能比 PNG 更小，但当前 ROS 现场没有标准 WebP transport，适合离线派生。 |
| TIFF/EXR | 是 | 可保存 `32FC1` 浮点深度；不是当前 ROS 直录格式，适合离线 sidecar。 |
| FFV1 / lossless H.264/H.265 | 是 | 视频级压缩，需要重组帧和时间戳，只能做离线视频派生。 |
| zstd/LZ4 | 是 | 对当前 `32FC1` 和 RGB raw 的额外收益有限，MCAP 已经使用 zstd。 |

如果必须保持 `v1` raw contract，以上编码都不要放进 recorder。推荐保留原始 MCAP，离线
生成 PNG/WebP/视频派生集；若允许新的相机数据版本，再评估 driver 原生 compressed topic。

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
uv sync --frozen --no-default-groups --extra record
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
数组明确覆盖所有 6 条高频 source；每条都必须有且仅有一个类型，并实际持续发布：

```bash
ros2 topic list -t | grep -E \
  'head_camera|wrist_camera|franka_robot_state_broadcaster|gripper|/tf'

high_rate_sources=(
  /left/franka_robot_state_broadcaster/current_pose
  /left/franka_robot_state_broadcaster/measured_joint_states
  /right/franka_robot_state_broadcaster/current_pose
  /right/franka_robot_state_broadcaster/measured_joint_states
  /left/gripper/joint_states
  /right/gripper/joint_states
)
for topic in "${high_rate_sources[@]}"; do
  ros2 topic info -v "$topic"
  timeout 6s ros2 topic hz "$topic" || true
done

ros2 topic echo /left/gripper/joint_states --once
ros2 topic echo /right/gripper/joint_states --once
```

确认 measured joint stream 是 `sensor_msgs/msg/JointState`，每侧包含 7 个有限 position，
且 header stamp 非零。`current_pose` 的唯一实际类型以 graph discovery 为准，不能凭 topic
名猜测。夹爪只录 `sensor_msgs/msg/JointState` 实际状态；多 joint 选择和闭合/打开端点不在
原始录包阶段决定，但必须保存现场标定记录供离线状态构造使用。夹爪 target command 不在
此 raw MCAP 中。

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

启动 `run_manual_recorder.sh` 或 `run_eval.sh` 后，在另一终端核对 6 条 relay 输出。每条
输出应有与 source 相同的唯一 message type，活跃 source 的输出频率不得超过 100 Hz：

```bash
high_rate_outputs=(
  /franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/current_pose
  /franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/measured_joint_states
  /franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/current_pose
  /franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/measured_joint_states
  /franka_duo_tele_data/rate100/left/gripper/joint_states
  /franka_duo_tele_data/rate100/right/gripper/joint_states
)
for topic in "${high_rate_outputs[@]}"; do
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

manual 流程会自动启动并监督 6 路高频状态 relay，确认全部 route ready 后才允许 rosbag2 使用
15 条 YAML topic 开录。两个脚本都调用同一个 `franka-duo-mcap-record`：空闲时 `r` 开始；
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
派生 LeRobot v3 写到另一目录。转换器是独立的离线步骤，不会改写 raw bag：

```bash
uv run --extra postprocess franka-duo-mcap-to-lerobot \
  --input-root /Users/logicluo/Downloads/franka_duo_tmr_raw_v1 \
  --output /data/franka_duo_tmr_lerobot_v3 \
  --usd /path/to/benchmark/assets/mobile_fr3_duo_v0_2.usd \
  --fps 15 --num-points 2048 --sampling adaptive --channels 3 \
  --workspace-min 0.4,-0.3,-0.3 --workspace-max 1.2,0.3,0.3 \
  --min-depth 0.05 --max-depth 5.0
```

`postprocess` extra 只安装 `rosbags`、`mcap`、`usd-core`、`pyarrow`、`pandas` 和 `av`，不会进入
现场 recorder 的基础环境。转换器按下面的顺序执行：

1. 读取 rosbag receipt timestamp 和原消息 `header.stamp`，验证每个 required topic 数量、
   时间单调性、消息类型、尺寸和 CameraInfo；
2. 用 `episode_event` 验证 start，用 manifest 验证 end/outcome/reward，不把元数据时间当图像时间；
3. 用新的 head RGB header stamp 作为目标帧，按明确阈值匹配 registered depth、左右 wrist、
   rate100 relay 中的 current pose、measured joints 和夹爪实际状态；这些状态使用保留的
   source header stamp，缺失时使用 relay receipt timestamp。有效同步帧随后按 `--fps` 固定
   时间网格抽样，避免旧 bag 中约 24.2 Hz 的 ZED 源帧被错误标成 15 Hz；每个抽样帧仍保留
   自己的 source stamp/skew；
4. 按训练任务明确选择并版本化 action/state 表示，记录每个派生帧对应的所有 source
   timestamp 和 skew；
5. 将三路 RGB 编为 `videos/<feature>/chunk-000/file-000.mp4`，低维数据写入
   `data/chunk-000/file-000.parquet`，并生成标准 `meta/info.json`、`stats.json`、
   `tasks.parquet` 和 `meta/episodes/...parquet`；
6. 同时写 `meta/derived_manifest.json`，其中保存 source stamp/skew 所在字段的定义、drop
   统计、点云参数、夹爪标定和全部固定坐标矩阵；每帧实际的 timestamp/skew 保存在 Parquet
   的 `observation.source_timestamp_ns` 和 `observation.sync_skew_ns` 列。

### 点云与坐标变换

`mobile_fr3_duo_v0_2.usd` 中已确认：`/left_fr3v2_link0` 和 `/right_fr3v2_link0` 的
世界坐标分别为 `(0.44190, +0.05018, 0.500885)` 和 `(0.44190, -0.05018, 0.500885)`。
所以新 `base` 取两者原点中点 `(0.44190, 0, 0.500885)`，方向采用 USD 根坐标方向；不对
左右镜像四元数做平均。USD 没有 ZED prim，转换器使用 `/head_camera_mounting_point` 作为
ZED 安装点，并把 ROS optical 约定 `(x right, y down, z forward)` 的
`mount -> zed_left_camera_frame_optical` 作为默认 nominal 外参。现场完成标定后可用
`--mount-to-optical m00,...,m33` 覆盖，覆盖矩阵会原样写进 derived manifest。

对样例 `franka_duo_tmr_raw_v1` 的 `/tf` 和 `/tf_static` 检查得到四个不连通组件：
机器人主体（`base`、双臂和移动底盘）、Robotiq 夹爪、双 D405 wrist 相机、ZED 相机链。
ZED 链只有 `zed_camera_link -> zed_camera_center -> zed_left/right_camera_frame(_optical)`，
没有 `base` 或 `head` 到 `zed_camera_link` 的边；wrist 链也没有接到机械臂。故这份 bag 的
TF 不能直接提供相机到新 base 的外参，转换器使用 USD nominal 矩阵，实机训练前应以测量的
静态外参替换。

代码中所有矩阵均为列向量约定：

```text
T_A_from_C = T_A_from_B @ T_B_from_C
p_A = T_A_from_C @ [p_C, 1]
```

参考实现为 [RL100](https://github.com/Starsshine21/RL100) 的
`3D-Diffusion-Policy/diffusion_policy_3d/gym_util/mujoco_point_cloud.py` 和
`gym_util/mjpc_wrapper.py`。每帧先用 ZED `CameraInfo.k` 将 registered `depth/depth_registered` 解投影到 ZED optical
系，得到 XYZ，再乘 `T_newbase_from_zed_optical`，并按 base 工作空间
`x∈[0.4,1.2]、y,z∈[-0.3,0.3]` 裁剪。默认使用自适应 voxel：自动选择体素边长，
每个体素保留距离体素中心最近的真实 XYZ 点，再做空间均匀删减，最终每帧精确输出
`2048×3`；不保存 RGB 点云信息。需要对照 RL100 的 FPS 时仍可显式指定
`--sampling fps`，但它的 CPU 成本更高。

双臂 `current_pose` 按用户约定视为各自 Franka base link 下的末端 pose，先转为
`T_armbase_from_ee`，再计算：

```text
T_newbase_from_ee = T_newbase_from_armbase @ T_armbase_from_ee
```

样例 v1 bag 中两路 `current_pose.header.frame_id` 都观测为 `base`，这只是驱动写入的字符串，
不能单独证明两路 payload 都已经在同一个物理 frame。转换器按左右 topic 的约定分别应用
对应 arm-base 变换；现场应核对驱动语义，必要时用标定矩阵修正，不能把不同 frame 的 pose
直接拼接。输出中的 `observation.ee_pose` 是左右各 `xyz + rot6d_rows` 的 18D 向量；
`rot6d_rows` 是每个 3×3 旋转矩阵前两行按行展平的连续 6D 表示。没有在线 FK，也不使用
measured joints 伪造末端 pose。

### 对齐、state 与 action

每个候选帧的时间轴是 ZED RGB 的 `header.stamp`。转换器从有界时间缓存中匹配最近的 depth、
左右 wrist RGB、左右 current pose、左右 measured joints 和左右 gripper state；带 header
的 topic 使用 header stamp，无 header 时使用 rosbag receipt timestamp。匹配阈值由
`--rgb-tolerance-ms`、`--depth-tolerance-ms`、`--state-tolerance-ms` 控制，结果的 9 路
skew 会保存为 `observation.sync_skew_ns`。匹配成功后以首个有效帧为起点，按 `--fps` 的
固定网格保留每个网格之后的第一帧；被丢弃的源帧计入 `dropped_resampled`，输出 Parquet/video
的 `timestamp` 始终是连续的 `frame_index / fps`。

`observation.state` 是 16D measured state：左 7 个关节、右 7 个关节、左右 actual
gripper open fraction。`action` 是 15Hz 重采样后下一个**有效同步帧**的 18D 双臂相对新
base 末端 pose，旋转使用 `rot6d_rows`；最后一个没有下一帧的候选会丢弃。raw bag 没有 gripper target，故
不会把 actual gripper state 冒充 action，也不会恢复旧的 20D/16D action contract。

`desired_joint_states` 因现场不变化而明确不录。因此这些 MCAP **不能**恢复旧的 16D
desired-joint action，转换器也不得复制 measured joints、填零或前向填充来伪造它。当前每侧
可用的机械臂原始量是 `current_pose` 和 `measured_joint_states`；夹爪只保留 actual joint
state，不包含 target command。
后续 LeRobot action 可以选择经验证的 EE target、夹爪 target 或其他比赛控制表示，但必须先单独定义维度、坐标系、时间
horizon 和归一化，再写入派生数据 manifest。

16D measured state 仍可在完成夹爪标定后离线构造：左右各 7 个 measured joint position，
再加左右夹爪 actual open fraction。它只是 observation state，不是 action。由于 raw MCAP 不含
夹爪 target，任何包含夹爪 action 的派生 action 必须来自另一个明确版本化的数据源或控制日志，
不得用 actual state 冒充 target。

转换器必须显式记录同步策略、阈值、drop/missing 统计、夹爪 joint 选择与两侧各自标定，不得
用 measured state 冒充 action，也不得为 spine/base/world EE 填零。当前后处理直接使用录制的
`current_pose`，在确认其语义为各自 Franka base link 后乘以 USD 静态变换；若现场只有关节
状态，则应另行用真实 URDF 做 FK。除非整条相机外参链已校准并验证，否则不能把 nominal
转换结果称为精确 world pose。

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

`run_eval.sh` 默认在模型运行期间同时保存 6 路 rate100 arm/gripper 输出和其余直录 TMR topic，
并把每次推理的观测 source stamp、耗时和 20D action 作为
`/franka_duo/eval/action_trace` 写入同一 MCAP。JSONL 只是可选副本；MCAP 是强制的现场
provenance，不能关闭。

`configs/tmr_eval.yaml` 的 `mcap.config` 相对 eval YAML 解析，默认引用 `tmr_mcap.yaml`；
`dataset_name` 为 `franka_duo_tmr_eval`，输出根目录继承 raw config 的
`datasets/franka_duo_mcap`。每次 eval 创建一个新的 `_vN` 和一个自动 episode，MCAP 在
eval ROS node 与推理循环之前启动。eval supervisor 也会自动启动同一 6 路高频状态 relay，只有
全部 source/type 唯一并且输出 ready 后才启动 bag；15 条 YAML topic、episode event 和
action trace 共 17 条进入同一个固定 `mcap/zstd_fast` bag。

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
