# FR3 双臂 LeRobot：16 维关节 state / action

2026-09-20 将 2026-09-16 采集的数据导出为新的关节空间版本。
用户已确认保留每臂全部 joint1–joint7，加上各自夹爪，共 16 维。

服务器：`agile@100.90.202.124`。新数据集目录：

```text
/home/agile/work/labs/data/lerobot_joint16/labs_fr3_joint16_20260916/
```

同目录提供 `labs_fr3_joint16_20260916.tar.zst` 和对应 `.tar.zst.sha256`。
LeRobot v3.0，122 episodes，46,502 帧，30 FPS；三路 RGB，640×480，366 个视频。
任务文本、episode/frame 顺序、图像和时间戳沿用原导出。

## 输入和输出

`observation.state` 与 `action` 都是 `float32[16]`：

| 索引（从 0 开始） | state | action |
| --- | --- | --- |
| 0–6 | 左臂 joint1–joint7 实测关节角 | 左臂 joint1–joint7 录制目标关节角 |
| 7 | 左夹爪实测开合 | 左夹爪目标开合 |
| 8–14 | 右臂 joint1–joint7 实测关节角 | 右臂 joint1–joint7 录制目标关节角 |
| 15 | 右夹爪实测开合 | 右夹爪目标开合 |

关节角单位为弧度（rad），表示绝对关节位置，未做归一化或差分。
夹爪沿用原导出的二值表示：**0=闭合，1=张开**。
state 的 34 维笛卡尔位姿等旧特征已移除；三路图像仍作为独立 observation 字段。

state 关节来自 `/left|right/franka_robot_state_broadcaster/measured_joint_states`；
action 关节来自 `/left|right/follower/gello/joint_states` 中录制的真实 follower 目标。
action 没有用实测关节或下一帧实测关节替代，也没有通过 IK 从旧位姿反解。

夹爪 state 来源是 `/left|right/follower/gripper/joint_states` 的 knuckle 实测角度，
按 `clip(1 - angle/0.8, 0, 1) >= 0.5` 转为二值开度；action 来源是
`/left|right/follower/gripper/gripper_client/target_gripper_width_percent`，按 `>= 0.5` 二值化。

## 同步和统计

直接复用旧导出已对齐的逐帧关节来源文件，并逐行核对头部相机时间戳。
实测状态沿用距头部图像最近、误差不超过 50 ms 的消息；目标使用图像时刻之前最新收到的
消息，最大年龄 100 ms；腕部图像匹配误差不超过 45 ms。
保留 `observation.source_timestamp_ns` 和 `observation.sync_skew_ns`，视频为名义 30 FPS。

`meta/stats.json` 中 `observation.state` 和 `action` 的 min/max/mean/std 均按新 16 维数据
重新计算，count 均为 `[46502]`；episode 元数据也包含各段的新 state/action 统计量。
图像逐字节复制、未经再次编码，图像统计量沿用原数据。
`meta/source_metadata/` 是旧 34D state / 20D action 的归档，不能用于新向量的归一化。

## 来源与验证

原数据集：`/home/agile/work/labs/data/lerobot/labs_fr3_link8_columns_20260916`。
关节来源：原数据集的同名 `.work` 目录内每段 `joint_provenance.npz`。
新数据集 `meta/joint_provenance/` 附带所有 122 段的 float64 实测关节与录制目标及时间戳。

- `meta/conversion_manifest.json`：版本化 schema、关节排列、消息来源、同步约定和文件 SHA-256。
- `meta/joint_conversion_validation.json`：全量 46,502 帧来源匹配，366 个视频复制校验，原数值数据和元数据未变。
- `meta/artifact_validation.json`：独立重读 Parquet，验证所有关节值及统计量；核对每个视频的帧数、尺寸、时间索引，并解码每个视频首尾图像。

所有 46,502 帧都至少有一个目标关节与同帧实测关节相差大于 1e-6 rad。
这份导出是独立的 `labs_fr3_absolute_joint_state16_action16_v1` 数据集契约。
使用模型时需要匹配此关节空间输入输出；现有 20D 笛卡尔执行接口没有随本次离线导出而更改。

## 重建

在安装了项目 postprocess 依赖的环境中，对一个尚不存在的输出目录运行：

```bash
python -m franka_duo_tele_data.labs_joint_dataset \
  --input /home/agile/work/labs/data/lerobot/labs_fr3_link8_columns_20260916 \
  --output /home/agile/work/labs/data/lerobot_joint16/labs_fr3_joint16_20260916
```

默认从输入的同名 `.work` 目录读取逐帧来源，也可用 `--provenance` 指定。
转换拒绝覆盖已有输出；未完成的中间结果保留在 `.assembling` 目录。
