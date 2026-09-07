# 导航 + 降腰 + YOLO 抓取整合流程

把底盘导航（`.50` 的 `tmr_cycle`）、腰部升降与任务编排（来自 `.100` 的
`tmr-mobile-manipulation`）与本仓库已验证的 ZED + YOLO 抓取链路合成一条完整往返：

```
收臂 → 抬腰 0.70 → 导航到桌前 → 降腰 0.468
→ 让视野姿态 → YOLO 检测 cup/bowl → 起始姿态 → 抓取
→ 抬腰 0.70 → 去字母侧 → 降腰 0.468 → 假放（碰桌、松、合、重新夹起）
→ 抬腰 0.70 → 回取物侧 → 第 20 步（左移 0.85 m、对齐远端桌腿、转 90°）
→ 降腰 0.468 → 真放（碰桌、松开、留下）
```

**每次行进前抬腰、到位后降腰。** 双臂挂在腰部滑座上，抓取姿态是前伸的，
带着这个姿态跑导航会超出底盘包络。所以腰高切换不是可选项，是安全前提。

**全程阻抗控制。** 每一次机械臂运动 —— 收臂、姿态切换、抓取、放置 —— 都走
joint servo → gello relay → 现场 `joint_impedance_controller`。不使用 PTP：
数据集就是在阻抗控制器下录制的，混入位置型运动发生器会重新引入那条链路
当初就是为了绕开的不连续 reflex。

## 放置就是抓取的逆过程

`build_place_rows` 是 `build_grasp_rows` 的镜像：

| 抓取 | 放置 |
| --- | --- |
| 当前位姿 → 目标上方 → 下降 | 当前位姿 → 垂直下降到桌面 |
| 下降完**闭**夹爪 | 下降完**开**夹爪 |
| 抬升带走 | 抬升（假放带走 / 真放留下） |

**X/Y 取自机械臂当前位姿，不复用抓取点。** 放置时底盘早已开到别处，
沿用抓取时存下的 X/Y 会让手臂伸向错误的地方。只有 z 变化 —— 这也正是
`.100` 现有 `move_vertical` 的做法（`--forward-m 0`）。

两种模式：

- `--mode test`（假放）：碰桌 → 松 → **再合** → 抬起带走。字母侧用。
- `--mode final`（真放）：碰桌 → 松 → 空手抬起，物体留下。第 20 步用。

下放高度 = `--table-z` + 物体在 EE 原点以下的长度（`release_clearance_m`：
物体高 − 抓取深度）+ `--margin-m`（默认 5 mm，避免压桌）。

## 为什么腰高和收臂是同一件事

双臂挂在腰部滑座上（`mobile_fr3_duo_v0_2.xacro`：`<xacro:franka_spine
parent="base_link"/>`，再 `fr3_duo base_mount="franka_spine_mounting_point"`），
所以腰一动，双臂整体升降。

导航的碰撞包络写死在 `config/start_to_pickup.yaml`：前后各 `0.40 m`、
宽 `0.58 m`。`tmr_cycle/README.md` 明确要求手臂必须收在这个包络内。

原有导航栈**完全不碰腰和手臂**：`tmr_cycle` 里 spine 只出现一次，是
`03_start_navigation.sh` 里的只读健康检查；`tmr_navigation` 零命中。
腰高切换全部由本流程负责。

## 组成

| 文件 | 作用 |
| --- | --- |
| `src/franka_duo_tele_data/spine_client.py` | 腰部绝对高度：`switch_on` → `move_absolute` → 读回复核 3 mm |
| `src/franka_duo_tele_data/table_grasp_stage.py` | 到桌后：降腰、让视野、检测、起始姿态、抓取 |
| `src/franka_duo_tele_data/table_place_stage.py` | 放置：假放 / 真放 |
| `src/franka_duo_tele_data/table_mission.py` | 跨机协调：腰高切换、ssh 调各段导航、抓取与放置 |
| `configs/grasp_stage_poses.json` | 三组双臂 EE 位姿（**待现场示教**） |
| `scripts/run_table_mission.sh` | 入口，默认 dry-run |

复用的 `.50` 导航段（都经 ssh 调用，不改它们）：

| 脚本 | 作用 |
| --- | --- |
| `07_start_to_pickup.py` | 起点 → 桌前 |
| `13_post_grasp_route.py` | 桌前 → 字母侧（后退 1.70、转 180°、后退 0.25、右移 0.80+0.85）|
| `15_return_from_letter.py` | 字母侧 → 取物侧 |
| `20_after_return_placement.py` | 左移 0.85 m、对齐远端桌腿、转 90°、等待放物 |

三段导航的成功判据各不相同，因此有各自的校验函数：13 是扁平的
`status`/`zero_command_latched`；15 把门口结果嵌在 `door_report` 里；
20 要求停在 `WAITING_FOR_PLACEMENT`。

## 腰高与桌高

- `0.700 m`：行进高度（`.100` 现有 `SPINE_HOME_M`）。每段导航前都会抬到这里。
- `0.468 m`：工作高度。**ZED 外参就是在这个高度标定的**，所以
  `--table-z -0.220` 沿用。换高度就必须重标外参。

桌面 z 都在 **midpoint base 帧**（左右 link0 原点连线的中点，随腰升降），
不是世界高度也不是底盘高度。测量必须在 spine 已经落到 0.468 之后做，
量的是「臂基座中点 → 桌面」的垂直距离。

`--placement-table-z` 是第 20 步放置桌，`--letter-table-z` 是字母侧假放桌
（不给就沿用前者）。**这两张桌不一定和取物桌同高，实跑前需实测。**
单张无深度图像无法反推空桌高度（ZED 的 `depth_mode` 是 `NONE`，
且桌上没有已知高度的参照物），所以这两个值只能实测。

## 现场示教（运行前必做）

`configs/grasp_stage_poses.json` 里三组位姿目前是 `null`，程序会拒绝运行 ——
不会拿猜测的姿态去动机器人。采集方法：Desk 切 Programming 拖臂 → 切回
Execution → `bash scripts/reconnect_arm_state.sh` → 读两臂 `current_pose`，
按 18D（左 xyz+rot6d，右 xyz+rot6d，midpoint 系）填入。

- `travel_stow`：**在 0.70 m 示教**，双臂收进导航包络。
- `camera_clear`：0.468 m，双臂退出 ZED 视野（右臂曾挡住半个托盘）。
- `grasp_ready`：0.468 m，左臂够碗、右臂够杯。

## 运行

```bash
# 只打印策略，不启动任何东西
bash scripts/run_table_mission.sh

# 完整往返（底盘侧需先起好导航栈）
EXECUTE=1 bash scripts/run_table_mission.sh 0.1 \
  --placement-table-z -0.220 --letter-table-z -0.220

# 只抓取、不放置（先验证前半段）
EXECUTE=1 bash scripts/run_table_mission.sh 0.1 --stop-after-grasp
```

底盘侧前置：在 `.50` 上先跑 `scripts/19_ensure_navigation_stack.sh` 和
`scripts/17_control_mode.sh mission`。

分段验证（每步等上一步确认）：

```bash
# 只验腰
.venv/bin/python -m franka_duo_tele_data.spine_client --target-m 0.468 --execute

# 只验收臂
.venv/bin/python -m franka_duo_tele_data.table_grasp_stage --dataset <ds> \
  --pose-only travel_stow --publish --enable-robot

# 只验感知（保存帧，不动机器人）
.venv/bin/python -m franka_duo_tele_data.grasp_cup_bowl --dataset <ds> --image frame.jpg

# 只验放置（手里必须已经夹着东西）
.venv/bin/python -m franka_duo_tele_data.table_place_stage --dataset <ds> \
  --mode test --arm right --target cup --table-z -0.220 --publish --enable-robot
```

## 安全约定

- 机器人动作需要 `--publish` 和 `--enable-robot` **两个**开关同时给。
- 所有配置、位姿、腰高、桌高校验都在 ROS 初始化**之前**完成。
- 每次腰高变化都读回实测值复核 3 mm，不达标就中止，不继续下一步。
- 检测不到目标就停在原地报错，绝不带着猜测下降。
- 放置前检查夹爪确实是闭合的（手里有东西），否则拒绝执行。
- 假放结束后校验夹爪仍闭合 —— 物体必须还在手上，否则报错。
- 放置目标平面高于当前手位会被拒绝（那是往上跳，不是放下）。
- 断点：`outputs/table_mission.json` 记录 phase。已经出发过的 phase 会拒绝
  重跑路线，除非把底盘退回起点并显式加 `--fresh-start-confirmed`。
- 重启 servo/relay 前必须先停用两侧 `joint_impedance_controller`。

## 跨机边界

两台机的 ROS 图不混：`.50` 是 Humble、域 97；`.100` 是 Jazzy、域 0。
每一段导航都通过 ssh 以子进程方式调起，共用同一把 `flock`，
保证任何时刻只有一段路线拥有速度通道。`/tmr_cycle/arm_command`
虽有发布端但两棵树里都没有订阅者，是条死接口，本流程不使用它。
