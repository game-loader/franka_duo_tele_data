# FR3 双臂 LeRobot v3 数据集说明：34D state / 14D action

本文对应最终数据集 `labs_fr3_link8_delta14_20260916`，不是之前的绝对位姿 20D action 版本。所有维度索引从 **0** 开始，切片采用 Python 左闭右开规则。

## 1. 数据集位置与概况

远程机器：`agile@100.90.202.124`。

```text
数据集目录：
/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916/

压缩包：
/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916.tar.zst

校验文件：
/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916.tar.zst.sha256
```

| 项目 | 内容 |
| --- | --- |
| 格式 | LeRobot v3.0 |
| 轨迹数量 | 122 episodes |
| 总帧数 | 46,502 |
| 名义帧率 | 30 fps |
| state | `observation.state`，`float32[34]` |
| action | `action`，`float32[14]` |
| 相机 | 头部、左腕、右腕，共三路 RGB |
| 视频数量 | 366 个 MP4，H.264 / yuv420p |
| 数据划分 | 元数据中全部为 `train: 0:122`，未另外划分验证集 |

所有帧的任务描述统一为：

> Use the left arm to place the square head into the yellow box on the left, and the right arm to place the screw into the green box on the right.

## 2. 图像宽、高、通道顺序

**宽 W = 640 像素，高 H = 480 像素，RGB 三通道。宽高没有交换。**

| 图像字段 | 物理含义 | 宽 × 高 | `meta/info.json` 中 shape | 官方 LeRobot 读取后的单帧张量 |
| --- | --- | --- | --- | --- |
| `observation.images.head` | 头部相机 | 640 × 480 | `[480, 640, 3]`，HWC | `[3, 480, 640]`，CHW |
| `observation.images.wrist_left` | 左腕相机 | 640 × 480 | `[480, 640, 3]`，HWC | `[3, 480, 640]`，CHW |
| `observation.images.wrist_right` | 右腕相机 | 640 × 480 | `[480, 640, 3]`，HWC | `[3, 480, 640]`，CHW |

图像处理过程：

1. 从 MCAP 解码相机消息为 RGB。
2. 统一直接 resize 到宽 640、高 480；没有裁剪或 letterbox。头部原图为宽 1280、高 720，直接缩放改变了宽高比例；左右腕原图已为宽 640、高 480。
3. 编码保存为 H.264 / yuv420p 视频；这是有损编码，不保证与原始 RGB 像素逐值相同。
4. 从绝对 20D 数据集转换到 14D 时直接复用视频，没有再次 resize 或重新编码。

没有在导出阶段做数据增强、图像均值中心化、按图像标准差缩放或 ImageNet 归一化。官方 LeRobot 0.4.3 / PyAV 读取图像时，解码并返回 CHW 浮点张量，像素范围为 `[0,1]`，对应 RGB 字节值除以 255；这与训练时可能执行的 mean/std 标准化是两个步骤。

## 3. 位姿坐标系与旋转表示

末端定义为本臂 **link8**，参考坐标系为本臂 **link0**：

| 手臂 | 参考坐标系 | 末端 |
| --- | --- | --- |
| 左臂 | `left_fr3_link0` | `left_fr3_link8` |
| 右臂 | `right_fr3_link0` | `right_fr3_link8` |

没有额外加 TCP 中点偏移。左、右臂的数据各自在自己的 link0 下表达，不能直接把两者坐标当成已标定的同一个世界坐标系。

旋转矩阵 `R` 将 link8 中的向量表示转换到对应 link0 下。state 的旋转使用 **rot6d_columns**，只保存 R 的前两列，顺序固定为：

```text
[R00, R10, R20, R01, R11, R21]
```

前三个元素是第一列，后三个元素是第二列。它们不是欧拉角，也不是四元数。矩阵元素通常在 `[-1,1]`，这是旋转矩阵本身的性质，并不代表做过 min-max 归一化。

## 4. observation.state：34 维逐维定义

结构：左右实测末端位姿 18 维 + 左右实测夹爪 2 维 + 左右原始实测关节角 14 维。

| 索引 | `meta/info.json` 中的维度名称 | 含义 | 单位/取值 |
| --- | --- | --- | --- |
| 0 | left_x | 左臂实测 link8 原点在本臂 link0 下的 x 坐标 | m |
| 1 | left_y | 左臂实测 link8 原点在本臂 link0 下的 y 坐标 | m |
| 2 | left_z | 左臂实测 link8 原点在本臂 link0 下的 z 坐标 | m |
| 3 | left_rot6d_col0_x | 左臂旋转矩阵 R 的元素 R00（第 1 列） | 无量纲 |
| 4 | left_rot6d_col0_y | 左臂旋转矩阵 R 的元素 R10（第 1 列） | 无量纲 |
| 5 | left_rot6d_col0_z | 左臂旋转矩阵 R 的元素 R20（第 1 列） | 无量纲 |
| 6 | left_rot6d_col1_x | 左臂旋转矩阵 R 的元素 R01（第 2 列） | 无量纲 |
| 7 | left_rot6d_col1_y | 左臂旋转矩阵 R 的元素 R11（第 2 列） | 无量纲 |
| 8 | left_rot6d_col1_z | 左臂旋转矩阵 R 的元素 R21（第 2 列） | 无量纲 |
| 9 | right_x | 右臂实测 link8 原点在本臂 link0 下的 x 坐标 | m |
| 10 | right_y | 右臂实测 link8 原点在本臂 link0 下的 y 坐标 | m |
| 11 | right_z | 右臂实测 link8 原点在本臂 link0 下的 z 坐标 | m |
| 12 | right_rot6d_col0_x | 右臂旋转矩阵 R 的元素 R00（第 1 列） | 无量纲 |
| 13 | right_rot6d_col0_y | 右臂旋转矩阵 R 的元素 R10（第 1 列） | 无量纲 |
| 14 | right_rot6d_col0_z | 右臂旋转矩阵 R 的元素 R20（第 1 列） | 无量纲 |
| 15 | right_rot6d_col1_x | 右臂旋转矩阵 R 的元素 R01（第 2 列） | 无量纲 |
| 16 | right_rot6d_col1_y | 右臂旋转矩阵 R 的元素 R11（第 2 列） | 无量纲 |
| 17 | right_rot6d_col1_z | 右臂旋转矩阵 R 的元素 R21（第 2 列） | 无量纲 |
| 18 | left_gripper_open | 左夹爪实测二值开合状态 | 0=闭合，1=张开 |
| 19 | right_gripper_open | 右夹爪实测二值开合状态 | 0=闭合，1=张开 |
| 20 | left_joint1_position | 左臂 joint1 的实测关节角 | rad |
| 21 | left_joint2_position | 左臂 joint2 的实测关节角 | rad |
| 22 | left_joint3_position | 左臂 joint3 的实测关节角 | rad |
| 23 | left_joint4_position | 左臂 joint4 的实测关节角 | rad |
| 24 | left_joint5_position | 左臂 joint5 的实测关节角 | rad |
| 25 | left_joint6_position | 左臂 joint6 的实测关节角 | rad |
| 26 | left_joint7_position | 左臂 joint7 的实测关节角 | rad |
| 27 | right_joint1_position | 右臂 joint1 的实测关节角 | rad |
| 28 | right_joint2_position | 右臂 joint2 的实测关节角 | rad |
| 29 | right_joint3_position | 右臂 joint3 的实测关节角 | rad |
| 30 | right_joint4_position | 右臂 joint4 的实测关节角 | rad |
| 31 | right_joint5_position | 右臂 joint5 的实测关节角 | rad |
| 32 | right_joint6_position | 右臂 joint6 的实测关节角 | rad |
| 33 | right_joint7_position | 右臂 joint7 的实测关节角 | rad |

可按以下切片读取：

```python
left_position = state[0:3]
left_rotation_columns = state[3:9]
right_position = state[9:12]
right_rotation_columns = state[12:18]
left_gripper, right_gripper = state[18:20]
left_joints = state[20:27]
right_joints = state[27:34]
```

### state 做过哪些处理

state **不是整条原始 ROS 消息直接落盘**，具体处理如下：

| 内容 | 实际处理 |
| --- | --- |
| 时间匹配 | 以头部图像时间为锚，测量关节和夹爪使用 header 时间戳的最近邻匹配，最大偏差 50 ms；最近邻可能在锚点之前或之后 |
| 左右 xyz / rotation | 用匹配到的实测关节角，通过该机器的 URDF 做 FK，得到 link0 到 link8 的位姿 |
| 两个实测夹爪值 | 将 knuckle 角转换成开度，再二值化为 0/1 |
| 末尾 14 个关节角 | 保留选中实测消息的原始角度数值和 joint1..7 顺序；仅转为 float32 存储 |
| 数据类型 | 整个 state 保存为 float32；相对于原 float64 可能存在浮点舍入 |
| 20D action 转 14D | 不改变任何 state 数值；已逐列验证保持一致 |

**没有**对保留的关节角做线性插值、平滑滤波、角度缩放、减均值、除标准差或 min-max 变换。当前 labs 适配层使用时间匹配，不使用 `SciPy interp1d` 对 state/action 做逐维插值。state 的关节角不是通过 IK 反算得到的。

实测夹爪二值化的具体公式为：

```python
opening = clip(1.0 - knuckle_angle_rad / 0.8, 0.0, 1.0)
gripper = 1.0 if opening >= 0.5 else 0.0
```

其中 `0.8 rad` 是本次转换采用的标称闭合归一化端点。这一步是夹爪字段的开度映射和二值化，并非对整个 state 做统计归一化。输入有效性另按实际驱动的 `0.7929 * (feedback_byte - 3) / 227` 校验；超过标称闭合角但仍在驱动有效反馈范围内的值也可接受，并映射为闭合。

## 5. action：14 维逐维定义

**每帧 action 都相对同一帧实测 state；平移和旋转增量都在各自 link0 下表示。**

目标位姿来自 MCAP 中实际录制的 follower **目标关节指令**经 FK 得到的 link8 位姿；不是将实测 state 复制成 action，也不是用下一帧实测状态作为 action。

| 索引 | `meta/info.json` 中的维度名称 | 含义 | 单位/取值 |
| --- | --- | --- | --- |
| 0 | left_delta_x | 左臂目标位置减实测位置，沿本臂 link0 的 x 轴 | m |
| 1 | left_delta_y | 左臂目标位置减实测位置，沿本臂 link0 的 y 轴 | m |
| 2 | left_delta_z | 左臂目标位置减实测位置，沿本臂 link0 的 z 轴 | m |
| 3 | left_delta_rotvec_x | 左臂相对旋转向量在本臂 link0 下的 x 分量 | rad |
| 4 | left_delta_rotvec_y | 左臂相对旋转向量在本臂 link0 下的 y 分量 | rad |
| 5 | left_delta_rotvec_z | 左臂相对旋转向量在本臂 link0 下的 z 分量 | rad |
| 6 | right_delta_x | 右臂目标位置减实测位置，沿本臂 link0 的 x 轴 | m |
| 7 | right_delta_y | 右臂目标位置减实测位置，沿本臂 link0 的 y 轴 | m |
| 8 | right_delta_z | 右臂目标位置减实测位置，沿本臂 link0 的 z 轴 | m |
| 9 | right_delta_rotvec_x | 右臂相对旋转向量在本臂 link0 下的 x 分量 | rad |
| 10 | right_delta_rotvec_y | 右臂相对旋转向量在本臂 link0 下的 y 分量 | rad |
| 11 | right_delta_rotvec_z | 右臂相对旋转向量在本臂 link0 下的 z 分量 | rad |
| 12 | left_gripper_open | 左夹爪目标开合值，绝对值，不做差分 | 0=闭合，1=张开 |
| 13 | right_gripper_open | 右夹爪目标开合值，绝对值，不做差分 | 0=闭合，1=张开 |

对每条手臂，记同帧实测位姿为 `(p_state, R_state)`，目标位姿为 `(p_target, R_target)`：

```text
delta_xyz    = p_target - p_state
delta_R      = R_target @ R_state.T
delta_rotvec = Log(delta_R)
```

`delta_rotvec` 是旋转向量：方向为旋转轴，向量长度为旋转角，单位 rad。它的三个分量不是 roll/pitch/yaw 的差，也不是角速度。平移增量没有除以帧间隔，因此也不是线速度。

实现使用 SciPy `Rotation` 的旋转矩阵/旋转向量转换，取主值旋转角 `[0, pi]`。计算相对旋转前，列式 6D 旋转会通过正交化恢复完整旋转矩阵。该几何处理与统计归一化无关。

反向还原：

```text
p_target = p_state + delta_xyz
R_target = Exp(delta_rotvec) @ R_state
```

必须使用转换时对应的同一帧 state。**不能跨帧累加这些 delta** 来还原动作序列。训练时如果构造未来多步 action chunk，需要保留或另行明确每一步所对应的参考 state；本数据集默认每行使用各自的 state，不是整个 chunk 共用首帧 state。

action 的夹爪来自录制的目标开度，原始开度 `>= 0.5` 记为 1，否则为 0。转换到 14D 时直接复制绝对 20D action 的夹爪两维，没有相减、累加或再次改变阈值。

## 6. 是否进行了归一化

| 数据 | 保存/读取时的处理 | 是否做 z-score / min-max |
| --- | --- | --- |
| state 中的位置 | 物理单位 m，float32 | 否 |
| state 中的旋转 | 旋转矩阵前两列，float32 | 否 |
| state 中的原始关节角 | 物理单位 rad，float32 | 否 |
| state / action 中的夹爪 | 开度映射并二值化为 0/1 | 没有做基于数据集统计量的标准化 |
| action 中的平移增量 | 物理单位 m，float32 | 否 |
| action 中的旋转增量 | 旋转向量，物理单位 rad，float32 | 否 |
| 图像 | 编码为视频；官方读取器输出 RGB/255 的 `[0,1]` 张量 | 未做按通道 mean/std 或 min-max 统计变换 |

`meta/conversion_manifest.json` 中的 `normalized: false` 表示数值 state/action 仍是上述物理量和二值夹爪。计算并保存统计量，不会自动将 Parquet 中的数据变成归一化数据。

如果后续模型训练的 preprocessor 选择 mean/std、min-max 或其他归一化，那是训练配置决定的额外处理。此次导出没有生成训练模型、已拟合的 policy normalizer 或其 checkpoint。

## 7. 归一化统计量保存在哪里

**转换到 14D 时，已经用全部 46,502 行新的 action 重新计算，并覆盖了新数据集根目录 `meta/stats.json` 的 `action` 统计项。当前 `mean/std/min/max` 各有 14 个值，`count=[46502]`。打包时包含的也是这份新的统计文件。** state 和图像没有改变，所以保留各自已计算好的统计量。

当前 **34D state / 14D action** 应使用：

```text
数据集根目录/meta/stats.json

远程完整路径：
/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916/meta/stats.json
```

压缩包中已包含该文件。解压后的相对路径相同。

| JSON key | `min/max/mean/std` 的 shape | 对应含义 |
| --- | --- | --- |
| `observation.state` | `[34]` | 按本文 state 索引顺序，逐维统计 |
| `action` | `[14]` | 按本文 action 索引顺序，逐维统计 |
| `observation.images.head` | `[3,1,1]` | 头部图像 R/G/B 三通道统计 |
| `observation.images.wrist_left` | `[3,1,1]` | 左腕图像 R/G/B 三通道统计 |
| `observation.images.wrist_right` | `[3,1,1]` | 右腕图像 R/G/B 三通道统计 |

每个字段都保存 `min`、`max`、`mean`、`std`、`count`。这些字段的 `count` 均为 `[46502]`。统计文件也有时间戳和索引等字段的统计，它们不是机器人 state/action 的维度。

统计计算方式：

- 数值 state/action：对全部导出帧逐维统计，标准差使用总体标准差 `sqrt(E[x²] - E[x]²)`。转换为 delta14 后，action 的统计量已从全部 46,502 行新动作重新计算；state 统计保持原值。
- 图像：对最终视频解码成 RGB，以 `rgb / 255` 的范围统计；每帧取 `rgb[::8, ::8]` 的空间网格，按 R/G/B 通道聚合全部帧。包含帧内像素方差，不是只统计每张图的均值。图像的 `count` 表示帧数，不是像素个数；min/max 为采样网格上的极值。
- 这些统计量覆盖当前全部 122 集。以后若重新划分训练/验证集、修改数据，应按实际训练范围重新确定统计量。
- 当前文件没有保存分位数 `q01/q99` 或训练时的 epsilon、归一化策略选择；这些不能从 `stats.json` 中假定存在。

**不要使用 `meta/source_metadata/stats.json` 中的 action 统计量归一化当前 action。** 那是原始绝对 20D 数据集的归档，action 参数是 20 维；当前 14D 动作必须用根目录的 `meta/stats.json`。

读取示例：

```python
import json
from pathlib import Path
import numpy as np

root = Path("labs_fr3_link8_delta14_20260916")
stats = json.loads((root / "meta/stats.json").read_text())

state_mean = np.asarray(stats["observation.state"]["mean"], dtype=np.float32)  # [34]
state_std = np.asarray(stats["observation.state"]["std"], dtype=np.float32)    # [34]
action_mean = np.asarray(stats["action"]["mean"], dtype=np.float32)           # [14]
action_std = np.asarray(stats["action"]["std"], dtype=np.float32)             # [14]
head_mean = np.asarray(stats["observation.images.head"]["mean"])             # [3,1,1]
head_std = np.asarray(stats["observation.images.head"]["std"])                # [3,1,1]

# 仅示例：如果训练配置明确采用逐维 z-score，可执行：
# state_normalized = (state - state_mean) / np.maximum(state_std, 1e-8)
# action_normalized = (action - action_mean) / np.maximum(action_std, 1e-8)
# 以上并未在数据集导出时执行；夹爪是否另行保留为 0/1 由训练配置决定。
```

当前文件中三路图像通道 mean/std 的数值如下，供核对；完整精度以 JSON 为准，顺序均为 R、G、B：

| 相机 | mean | std |
| --- | --- | --- |
| 头部 | `[0.414135076, 0.420549006, 0.383668833]` | `[0.337644769, 0.330207850, 0.330814938]` |
| 左腕 | `[0.461432165, 0.433523597, 0.342746587]` | `[0.288080912, 0.251123084, 0.252348727]` |
| 右腕 | `[0.389833113, 0.486282333, 0.416894286]` | `[0.195088221, 0.208953093, 0.187139901]` |

## 8. 同步和时间字段

头部图像的原始 header 时间戳作为同步锚点：

| 信号 | 匹配方式 | 最大时间偏差 |
| --- | --- | --- |
| 左、右腕图像 | 最近 header 时间戳 | 45 ms |
| 左、右实测关节及夹爪 | 最近 header 时间戳 | 50 ms |
| 左、右目标关节及目标夹爪 | MCAP log_time 不晚于锚点的最后一条消息 | 最长历史 100 ms |

目标指令的 MCAP log_time 用作录制接收时间的代理值。目标匹配保证不会使用头部图像锚点之后收到的目标。不存在有效匹配的帧会跳过，没有使用实测 state 补成 action。

| 数据字段 | 含义 |
| --- | --- |
| `timestamp` | episode 内名义时间，`frame_index / 30`，秒 |
| `observation.source_timestamp_ns` | 头部图像的原始时间戳，纳秒 |
| `observation.sync_skew_ns` | 各匹配消息时间减头部锚点时间，纳秒，10 维 |
| `frame_index` | episode 内从 0 开始的帧索引 |
| `episode_index` | episode 编号 |
| `index` | 数据集全局帧索引 |
| `task_index` | 任务编号，本数据集为 0 |

`observation.sync_skew_ns` 的顺序为：左腕图像、右腕图像、左实测关节、右实测关节、左实测夹爪、右实测夹爪、左目标关节、右目标关节、左目标夹爪、右目标夹爪。最后四维均不大于 0。

30 fps 是输出视频和名义时间的约定。源相机存在时间抖动/缺帧，且无效匹配会被跳过，因此不能假定所有相邻源时间戳都恰好相差 1/30 秒；实际时间以 `observation.source_timestamp_ns` 为准。

## 9. 文件位置与验证

| 文件 | 用途 |
| --- | --- |
| `data/chunk-*/file-*.parquet` | state、action、索引和时间字段 |
| `videos/<图像字段>/chunk-*/file-*.mp4` | 三路 RGB 视频 |
| `meta/info.json` | 字段名称、shape、dtype、帧率、图像宽高等 |
| `meta/stats.json` | 当前 34D state 和 14D action 及图像的统计量 |
| `meta/conversion_manifest.json` | 坐标系、增量约定、维度含义和转换来源 |
| `meta/episodes/chunk-*/file-*.parquet` | 每个 episode 的数据/视频范围等 |
| `meta/tasks.parquet` | 任务文本 |
| `meta/action_conversion_validation.json` | 全部行的转换与往返误差检查 |
| `meta/restored_vs_original_validation.json` | 还原 20D 后与原始数据集的逐行比较 |
| `meta/official_reader_validation.json` | 官方 LeRobot 0.4.3 读取检查 |
| `meta/source_metadata/` | 原绝对位姿 20D 数据集的元数据归档 |

已验证 122 集、46,502 帧往返转换；最大位置误差 `1.862645149230957e-09 m`，最大旋转误差 `8.588537754633716e-08 rad`。state、夹爪和其他非 action 列保持不变；366 个视频未重新编码。官方读取器已验证每集首尾帧共 244 帧。

双向工具位于仓库 `src/franka_duo_tele_data/labs_action_delta.py`：

```bash
python -m franka_duo_tele_data.labs_action_delta to-delta \
  --input /path/to/absolute20_dataset --output /path/to/delta14_dataset

python -m franka_duo_tele_data.labs_action_delta to-absolute \
  --input /path/to/delta14_dataset --output /path/to/restored20_dataset
```
