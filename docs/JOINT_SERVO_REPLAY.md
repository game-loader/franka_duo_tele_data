# 关节空间 Servo 回放（Joint Servo）

本文描述 2026-09-05 新增的关节空间执行链路，以及在目标机上用录制数据集测试它的步骤。
它取代此前被否定的 Cartesian Pose 路径，并修正旧 JTC 路径“每个 chunk 从零速度重启”的不连续问题。

## 为什么走关节阻抗接口

| 进入 Franka 的接口 | 连续性检查 | 非实时内核 overrun 后果 | 本机验证状态 |
| --- | --- | --- | --- |
| Cartesian pose | 末端 + 内部关节 | 一个坏样本即触发 reflex | 真机失败（2026-09-05） |
| Joint position（JTC） | 关节 vel/acc/jerk | 同上 | 历史不连续 |
| Effort（现场 `joint_impedance_controller`） | 仅力矩变化率、碰撞阈值 | 阻抗天然低通 | 遥操作日常使用；数据集即由它录制 |

## 链路

```text
replay_rgb20d.py --joint-servo
  绝对步序 chunk（layout.data_offset = 行 0 的数据集步）-> /franka_duo/joint_servo/action_chunk
  订阅 /franka_duo/joint_servo/status 决定何时请求下一块

site/franka_duo_joint_servo/policy_chunk_joint_servo
  逐行 MoveIt KDL IK（link8 tip，0.174 m 工具偏移，seed=上一规划步，consistency limit）
  JointTimeline：新 chunk 只替换 ceil(now)+commit_lead_steps 之后的步，前 blend_steps 步与旧规划混合
  1 kHz 每臂 Ruckig position 跟踪（位置+速度+加速度前馈），状态跨 chunk 不重置
  -> /franka_duo/joint_servo/{left,right}/target   sensor_msgs/JointState

gello_target_relay（enable_robot 门）
  -> /{left,right}/gello/joint_states
  -> franka_fr3_arm_controllers/JointImpedanceController（effort，1 kHz）
```

现场控制器合同（读自 overlay 源码）：按下标取前 7 个 `position`；目标或 `header.stamp` 超过 0.5 s 未更新则 `rclcpp::shutdown()` 整个 driver。
因此 servo 从启动起持续发布：空闲时跟随测量关节，轨迹结束或故障后保持最后命令。故障需重启节点清除。

## 已验证

- 目标机 gtest 9/9：时间轴替换、混合、Hermite 采样连续性；Ruckig 跟踪器在随机重定目标下速度/加速度/jerk 全程受限，加速度前馈后正弦跟踪误差约 3e-3 rad。
- 隔离 domain 221 无硬件 dry-run（假关节以首帧 IK 解为 seed，episode 0 前 60 帧，0.1 倍速）：3 个 chunk、IK 每块约 3 ms、1 kHz 无 overrun、Ruckig 速度场最大 0.4 rad/s，夹爪目标 60 条。脚本 `scripts/dev_joint_servo_dryrun.sh`。
- **真机（2026-09-05 23:41，Asia/Shanghai）**：error_recovery 两臂成功 → PTP 到 episode 0 首帧（两臂 TARGET_REACHED）→ servo 空闲 → relay 开门 → 激活两侧 `joint_impedance_controller` → 回放 episode 0 前 60 帧，0.1 倍速。结果：3 个 chunk，无 reflex，无故障位，servo 无 overrun；EE 跟踪误差最大左 2.5 cm / 0.094 rad、右 0.9 cm / 0.054 rad；结束位姿与第 59 帧目标差约 2 mm；servo 关节跟踪误差最大约 0.04 rad。日志在目标机 `log/live_20260905/`。
- **真机第二段（2026-09-06 00:10，含夹爪）**：停用阻抗控制器 → 按 PID 停旧 servo/relay → PTP 到 episode 0 第 160 帧 → servo 与 relay 均 `enable_gripper:=true` → 激活阻抗控制器 → 回放第 160 到 219 帧，0.1 倍速。结果：3 个 chunk，无 reflex，无故障位；EE 跟踪误差最大左 0.5 cm / 0.022 rad、右 0.7 cm / 0.043 rad。夹爪命令按 20D 第 18、19 维逐步下发（右 19 条打开后 41 条闭合，左 27 条打开后 33 条闭合），两侧 Robotiq 从 0（开）实际闭合到约 0.79 / 0.75，与数据集第 179、187 帧的闭合事件一致。
- **真机整段（2026-09-06 00:17）**：PTP 回第 0 帧后回放 episode 0 全部 381 帧，0.1 倍速，两侧夹爪打开。23 个 chunk，无 reflex，无故障位，servo 无 overrun；EE 跟踪误差最大左 2.8 cm / 0.064 rad、右 1.2 cm / 0.110 rad；结束位姿与第 380 帧目标差约 3 到 4 mm。夹爪：开始时从上一段遗留的闭合状态打开，随后在数据集闭合事件处两侧闭合并保持到结束，与 action 第 18、19 维一致。servo 关节跟踪误差在前 250 帧约 0.03 rad，第 300 帧后升到约 0.10 rad（接近 0.15 rad 保持阈值），对应动作较快的段落，提速前需先看这一段。
- **真机首次模型推理（2026-09-06 10:26）**：三路相机就位后，`smolvla_once` 读取一帧真实观测（head 360×640，腕 270×480，20D state）发给推理服务，108 ms 返回 32×20 chunk。chunk 内相邻行最大步长 1.2 cm / 0.019 rad，首行与当前位姿差左 0.5 cm、右 1.0 cm。经 servo 以 0.1 倍速执行 32 步，两臂无 reflex、无故障位，servo 关节跟踪误差最大 0.05 rad，结束位姿与最后一行目标差左 0.9 cm / 0.023 rad、右 1.1 cm / 0.059 rad。模型输出夹爪约 0.98，阈值化为 1（开）；左夹爪由 0.29 打开到 0，右夹爪保持打开。原始与处理后的 chunk 落盘 `outputs/smolvla_once_live.json`。
- **连续三次推理（2026-09-06 10:35）**：`smolvla_once --repeat 3` 在已运行的 servo 上直接追加 chunk（起始步 = 当前步 + 3），不重启 servo 或控制器。三轮均 32 步、0.1 倍速，无 reflex、无故障位；推理 109 到 119 ms，往返 400 到 530 ms；servo 关节跟踪误差最大 0.040 rad；每轮结束位姿与最后目标差 2 到 3 mm、0.02 到 0.05 rad。三轮末端累计位移约左 10 cm、右 4 cm；模型输出夹爪始终约 1.0（开），两夹爪保持打开。文件 `outputs/smolvla_live3_r{1,2,3}.json`。
- **连续控制（2026-09-06 10:45，`smolvla_stream`）**：上一块剩余不足半个 horizon 时取新观测、推理并以绝对步（当前步 + 5）追加下一块，servo 端替换与混合，机械臂不停。20 块、72 s、0.1 倍速，无 reflex、无故障位；推理 89 到 154 ms，往返 300 到 520 ms；servo 关节跟踪误差最大 0.040 rad；每块首行与当前位姿差 0.1 到 1.3 cm，块内相邻行最大步长 1.3 cm 以内。运行期间 5130 个状态样本中仅 70 个处于 holding（约 1.4%，为起步与结束段），说明块间基本无停顿。夹爪：模型多块末尾输出闭合，右夹爪在运行中从 0 逐步闭合到 0.36，左夹爪保持打开。已知小问题：每次请求后 servo 状态尚未更新 last_step，导致第二块紧跟第一块发出（成对请求），无害但多耗一次推理，后续可在发布后等待 last_step 更新再判定。日志 `outputs/smolvla_stream.jsonl`。

## 右臂激活 reflex（2026-09-06 凌晨）

**根因（2026-09-06 11:15 定位）**：PTP 回首帧的 KDL IK 解把两臂 joint 6 放在了位置下限 0.4398 rad 上（解为 0.43982 / 0.44004 / 0.44033）。FR3 的位置相关速度限制（`joint_limits.yaml`：j6 `velocity_offset` 0.4885，`deceleration_limit` 5.5）在限位处允许速度约为 0，任何让 j6 在此处移动的命令都会触发 `joint_velocity_violation`；激活瞬间的微小暂态或第一块 chunk 中 j6 的 0.006 rad 运动（离线复现峰值 0.026 rad/s）就足够。右臂 j6 离限位 0.0005 rad，左臂 0.014 rad，左臂因此侥幸存活。之前成功的会话里 j6 在 0.49 到 0.83 rad。与网络、主机负载、相机、碰撞阈值无关；之前的“重启后成功”是那次 PTP 解恰好不在限位。

**修法**：IK 求解（PTP 工具与 servo）拒绝距关节限位不足安全余量（建议 0.15 rad）的解，并用推离限位的 seed 重试或加零空间偏置；servo 的速度守卫改为按位置相关限制计算。

双臂 driver 重启后右臂连续 4 次在 `joint_impedance_controller` 激活后 50 到 70 ms 内 reflex（`joint_velocity_violation` 2 次、`communication_constraints_violation` 2 次），1 kHz 记录显示手臂未动，网络、主机负载、相机、下发目标均已排除，左臂同期正常。用户重启两台机器人控制器并用 `set_full_collision_behavior` 放宽两臂碰撞阈值后，右臂一次激活成功。两个变化同时发生，无法区分哪个起效。ZED 在 50 主机重启后只枚举 HID 接口，需现场重插 USB 3 才恢复。

## 真机运行中的两次失败及修正

1. 回放程序的 joint-servo 路径曾把 midpoint 坐标的行直接发给以 link0 工作的 servo，导致首行 IK 失败并丢弃 chunk（机械臂未动）。已修正为发送前 `to_link0_action`，并给 servo IK 加了重试与诊断日志。
2. 一个开发脚本用 `pkill -f` 误杀了生产 servo；165 s 后 servo 重启，现场 `JointImpedanceController` 收到间隔 >0.5 s 的下一条目标后 `rclcpp::shutdown()` 关闭了右臂 driver（左臂靠竞争存活）。已重启右臂 driver 并复位。修正：`gello_target_relay` 新增 `hold_on_input_loss`（输入断流时按 5 ms 重发带新时间戳的最后目标），开发脚本只杀自己启动的 PID。

**注意**：目标机上当前运行的 relay 仍是不带 hold 逻辑的旧进程。要换新二进制或重启 servo，必须先停用两侧 `joint_impedance_controller`，否则 driver 会被关掉。servo 完成一段回放后处于 `started=true` 保持态，再跑下一段前需重启 servo（同样先停用阻抗控制器）。

## 真机步骤（每步等上一步确认）

```bash
cd /home/aup/franka_duo_tele_data_infer_20260905 && source ~/tmr_env.sh && source site/install/setup.bash

# 1. 两臂 driver，只带广播（左臂示例；右臂换 right / 172.16.16.11）
ros2 launch franka_fr3_arm_controllers franka.launch.py arm_id:=fr3v2 arm_prefix:=left namespace:=left \
  robot_ip:=172.16.16.12 load_gripper:=false joint_sources:=joint_states

# 2. PTP 到 episode 0 首帧；两臂均 TARGET_REACHED 后再继续
ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py action_file:=$PWD/outputs/rgb20d_start_action.json \
  action_index:=0 max_joint_velocity:=0.05 wait_timeout_s:=60.0 execute:=true confirm:=true

# 3. servo（空闲：跟随测量关节，只发 site 话题）
ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:=0.1

# 4. relay 开门；确认 /left/gello/joint_states 恰有一个发布者
ros2 launch franka_duo_joint_servo gello_target_relay.launch.py enable_robot:=true
ros2 topic info /left/gello/joint_states

# 5. 激活阻抗控制器（它会平滑走到 servo 目标，即当前测量位姿）
ros2 run controller_manager spawner joint_impedance_controller -c /left/controller_manager
ros2 run controller_manager spawner joint_impedance_controller -c /right/controller_manager

# 6. 回放
bash scripts/run_rgb20d_replay.sh --dataset datasets/franka_duo_lerobot_rgb20d_v1 \
  --live --recorded-actions --joint-servo --episode 0 --max-steps 60 --speed 0.1 --publish --enable-robot
```

回放程序在发布前只读核查：servo 使用 link0、`playback_speed` 等于 `--speed`、两侧 relay 已开门、每侧只有 `joint_impedance_controller` 占用命令接口。
首帧位姿误差超过 3 cm 或 0.2 rad 时拒绝启动。运行中 servo 的关节跟踪误差超过 0.15 rad 会进入保持。

## 参数

默认限值在 `site/franka_duo_joint_servo/config/joint_servo_smoke.yaml`：关节速度 0.8 rad/s、加速度 2.0 rad/s²、jerk 20 rad/s³，`max_joint_delta_rad` 0.35。
chunk 需要的速度超过 0.9 倍速度上限时整块丢弃并报错。提速前先看 `status` 的 `tracking_error_rad`。
