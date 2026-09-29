# 2026-09-24 17:35 前的叠碗数据

用户指定仅转换当日 17:35（Asia/Shanghai）前已录制完成并标记成功的 105 条。
原始目录：`/home/agile/work/labs/data/raw_episodes/2026/09/24`。
按 `record_metadata.end_timestamp <= 2026-09-24T17:35:00+08:00`，且
`SAVED + REVIEW_SUCCESS + completed` 筛选，数量确认为 105。
最后一条录制于 17:33:28.826 完成，原始录制总时长 1596.794559 秒。

任务英文：**Stack the three bowls together.**

该指令覆盖原始元数据中遗留的 red box 任务文案；原始元数据不改写，并随来源清单归档。

## 完成结果

2026-09-24 23:24 已完成，`completed.json` 状态为 `passed`：

- 105 episodes，45,470 个输出样本；源同步 observation 为 45,575 帧。
- 105 个末帧 observation 被移除，所有 action 均与下一源实测 pose20（含夹爪）一致。
- 跨 episode 转移为 0，FK 重算与源 pose20 的最大误差为 0。
- 315 个视频完整解码，136,725 个物理视频帧；统计仅包含保留的 observation。
- 全部特征统计已重算，源数据保持不变。
- 相邻 observation 实际间隔最小/中位/最大为 31.166/33.370/66.993 ms，
  大于 50 ms 的间隔共 99 个，原始时间戳保留供检查。

## 数据定义

复用 `labs_mcap_to_lerobot` 和 `labs_next_state_dataset`，输出 LeRobot v3.0：

- `observation.state` 和 `action` 均为 float32[20]。
- 排列：左 xyz + rotation6D，右 xyz + rotation6D，左夹爪，右夹爪。
- rotation6D 为旋转矩阵前两列；各臂使用本臂 link0 到 link8 的实测关节 FK。
- `action[t]` 为同一 episode 下一个保留 observation 的实测 20D 状态。
- 夹爪沿用 0=闭合、1=张开；绝对位姿、未归一化。
- 每条 episode 去掉最后一个 observation 样本，末帧仅用于最后一个 action。
- 三路 head、wrist_left、wrist_right RGB 视频，640×480，名义 30 FPS。

详细定义见 `LABS_NEXT_STATE20_DATASET_20260920_ZH.md`。

## 主机上的输出和任务记录

```text
/home/agile/work/labs/data/lerobot_next_state20/labs_fr3_link8_next_state20_20260924_bowls105
/home/agile/work/labs/data/lerobot_next_state20/jobs/20260924_bowls105
```

任务目录下 `source_snapshot.json` 固定 episode 清单，`export.log` 记录进度，
仅成功完成导出及内置数据完整性检查后生成 `completed.json`。
本次脚本新增 `--completed-before` 和 `--expected-episodes`，防止重跑时混入后续采集。

```bash
PYTHONPATH=/home/agile/work/labs/data/tools/labs31_export_20260923/src \
/home/agile/work/labs/data/tools/labs31_export/.venv/bin/python -u \
  /home/agile/work/labs/data/tools/labs31_export_20260924/scripts/labs_export_next_state20.py \
  --input /home/agile/work/labs/data/raw_episodes/2026/09/24 \
  --job /home/agile/work/labs/data/lerobot_next_state20/jobs/20260924_bowls105 \
  --output /home/agile/work/labs/data/lerobot_next_state20/labs_fr3_link8_next_state20_20260924_bowls105 \
  --config /home/agile/work/labs/data/tools/labs31_export_20260923/configs/labs_fr3_31 \
  --task 'Stack the three bowls together.' \
  --workers 6 \
  --completed-before '2026-09-24T17:35:00+08:00' \
  --expected-episodes 105
```

不要与正在运行的同一任务并发执行。重跑使用已保存的快照和完成的 MCAP 转换检查点。
