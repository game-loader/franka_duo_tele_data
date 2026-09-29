# 2026-09-24 后续 98 条叠毛巾数据

任务英文：**Fold the towel.**

排除当日已导出的 105 条叠碗数据，仅纳入本次快照时已完成且成功的其余录制。
本次源目录为 `/home/agile/work/labs/data/raw_episodes/2026/09/24`，共选中 98 条，
均为 `SAVED + REVIEW_SUCCESS + completed`，与叠碗清单交集为 0，合计覆盖 203 条。

按 `20260924_bowls105/source_snapshot.json` 的 episode ID 精确排除；
排除清单路径和 SHA-256 保存在新快照中。额外截止时间为
`2026-09-24T23:26:00+08:00`，后续新增录制不进入本次固定快照。
原始录制总时长为 5206.223049 秒，第一条开始于 17:54:20，最后一条结束于 22:58:24。
任务文本以用户要求为准，覆盖原始元数据中遗留的 red box 文案，原始文件不改写。

## 数据定义和路径

复用已有 MCAP 转换和下一时刻实测状态标签程序，输出 LeRobot v3.0：

- `observation.state` 和 `action` 均为 float32[20]。
- 排列为左 xyz + rotation6D、右 xyz + rotation6D、左夹爪、右夹爪。
- rotation6D 为旋转矩阵前两列，位姿由各臂 link0 到 link8 实测关节 FK 计算。
- `action[t]` 为同一 episode 下一个保留 observation 的实测 20D 状态。
- 绝对位姿、未归一化；夹爪二值 0=闭合、1=张开。
- 每条去掉末尾一个 observation；末帧状态用于该段最后一个 action，不跨 episode。
- head、wrist_left、wrist_right 三路 RGB 视频为 640×480，名义 30 FPS。

详细语义见 `LABS_NEXT_STATE20_DATASET_20260920_ZH.md`。

```text
数据集：/home/agile/work/labs/data/lerobot_next_state20/labs_fr3_link8_next_state20_20260924_towel98
任务记录：/home/agile/work/labs/data/lerobot_next_state20/jobs/20260924_towel98
```

任务记录中的 `source_snapshot.json` 为固定清单，`export.log` 为进度，
`launch.json` 保存实际命令。只有转换及内置完整性检查成功后才生成 `completed.json`。
本次导出脚本增加 `--exclude-snapshot`，其余转换程序沿用已完成的叠碗批次版本。
