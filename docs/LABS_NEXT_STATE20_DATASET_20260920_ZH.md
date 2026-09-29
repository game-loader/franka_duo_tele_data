# 双臂绝对末端轨迹：20D state / 下一帧实测 20D action

2026-09-20 按用户明确指定的标签定义导出：预测后续实际达到的末端轨迹。
本版本使用**绝对位姿**，不计算 delta，不需要选择相对位姿参考帧。

服务器：`agile@100.90.202.124`。

```text
数据集：
/home/agile/work/labs/data/lerobot_next_state20/labs_fr3_link8_next_state20_20260916/

压缩包与 SHA-256 文件：同目录的
labs_fr3_link8_next_state20_20260916.tar.zst
labs_fr3_link8_next_state20_20260916.tar.zst.sha256
```

## 标签定义

每个原始 episode 内，令 `q[t]` 为已同步的双臂实测关节角，`g[t]` 为实测夹爪开合。

```text
state[t]  = [FK_left(q_left[t]),   FK_right(q_right[t]),   g_left[t],   g_right[t]]
action[t] = [FK_left(q_left[t+1]), FK_right(q_right[t+1]), g_left[t+1], g_right[t+1]]
```

因此在输出 episode 内存在下一行时，`action[t] == state[t+1]`，包含夹爪，逐值相等。
每段原有 N 帧生成 N−1 个样本：最后一个 action 使用原段末帧实测状态，但原段末帧
不作为 observation 样本。不会生成跨 episode 的状态转移，也不会用末帧自身填充 action。

原数据为 122 段、46,502 帧；输出为 **122 段、46,380 帧**。
三路图像仍对应当前 observation 时刻 t，action 对应下一次保留的同步 observation。
名义频率 30 FPS；实际源时间间隔保留在时间戳列中，少量原始缺帧处可约为 67 ms。

这是新的监督标签语义：`action` 表示下一帧实际达到的状态，不表示录制的控制器命令。
旧的真实目标关节数据集保持独立。本版本的 schema 为：
`labs_fr3_link8_columns_state20_next_measured_action20_v1`。

## 20 维排列、坐标系和单位

`observation.state`、`action` 均为 `float32[20]`，排列完全相同：

| 索引，从 0 开始 | 内容 |
| --- | --- |
| 0–2 | 左末端 x、y、z，单位米 |
| 3–8 | 左旋转矩阵前两列：`R00,R10,R20,R01,R11,R21` |
| 9–11 | 右末端 x、y、z，单位米 |
| 12–17 | 右旋转矩阵前两列：`R00,R10,R20,R01,R11,R21` |
| 18 | 左夹爪，0=闭合，1=张开 |
| 19 | 右夹爪，0=闭合，1=张开 |

左臂使用 `left_fr3_link0 -> left_fr3_link8`，右臂使用 `right_fr3_link0 -> right_fr3_link8`。
使用此前从现场获取并记录 SHA-256 的 URDF；每臂 7 个关节全部参与 FK。
没有额外 TCP 偏移，左右臂各自在本臂坐标系表达。
旋转是绝对旋转矩阵的列编码；没有位置差分、旋转差分或统计归一化。

夹爪沿用原数据的实测二值开度：knuckle 角度经 `clip(1 - angle/0.8, 0, 1) >= 0.5` 得到。
本版本 action 的夹爪也来自下一帧实测值，不使用旧目标夹爪列。

## 图像和统计量

LeRobot v3.0；头部、左腕、右腕三路 RGB 均为 640×480、名义 30 FPS。
366 个原视频逐字节复制，不再次编码。
每个视频物理上仍包含原段的末帧，但该帧位于新 episode 元数据引用范围之外，
不会成为一个 observation 样本。视频引用区间是 `[0, (N−1)/30)`。

`meta/stats.json` 内所有数值特征的统计均按新行重算，包括 state、action 和索引。
图像统计通过逐帧解码、每隔 8 个像素采样 RGB 重算，仅包含保留的 observation 帧；
根统计及 episode 统计的 count 均与新的样本数一致。
`meta/source_metadata/` 仅归档旧数据集的契约和统计，不能用于新标签归一化。

## 时间及来源字段

| 字段 | 含义 |
| --- | --- |
| `observation.source_timestamp_ns` | 当前源 observation 的头部图像时间 |
| `observation.sync_skew_ns` | 当前左右腕图像、左右关节、左右夹爪共 6 个匹配偏移 |
| `action_source_timestamp_ns` | 下一源 observation 的头部图像时间 |
| `action_source_frame_index` | 下一帧在源 episode 内的索引 |
| `action_source_sync_skew_ns` | 下一帧左右关节、左右夹爪共 4 个匹配偏移 |

`action_source_*` 是标签来源审计字段，不是 observation 输入。
旧的 follower 目标匹配偏移已从当前 observation 匹配字段中移除。

`meta/measured_provenance/episode_XXXXXX.npz` 保留完整源段的实测关节、实测 pose20、
时间戳和测量匹配偏移，包含仅用于最后一个 action 的原段末帧。
`meta/kinematics/` 附带左右 URDF；`meta/conversion_manifest.json` 记录来源和文件校验值。

## 验证及重建

`meta/next_state_validation.json` 记录全量 FK 重算、下一状态标签、episode 边界、
全部视频解码及统计重算检查。`meta/artifact_validation.json` 为独立读取成品后的检查报告。

在包含项目 postprocess 依赖的环境中运行：

```bash
python -m franka_duo_tele_data.labs_next_state_dataset \
  --input /home/agile/work/labs/data/lerobot/labs_fr3_link8_columns_20260916 \
  --output /home/agile/work/labs/data/lerobot_next_state20/labs_fr3_link8_next_state20_20260916 \
  --config configs/labs_fr3_31
```

输出目录必须不存在；默认从输入的同名 `.work` 目录读取实测关节来源。
本次仅导出离线数据集，没有更改现有机器人执行接口。
