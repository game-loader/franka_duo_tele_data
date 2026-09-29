# 历史原始 MCAP 无损归档

## 当前任务：9 月 23 日快速归档

此前 22 级任务已按用户要求停止；9 月 16 日原始 MCAP 及压缩副本已另行按用户
要求删除，LeRobot 成品保留。2026-09-24 21:19 左右启动的新任务只处理 9 月 23 日：
110 个已完成录制的 MCAP（包括当天一条失败标签记录），采用 Zstandard 3 级、
8 线程，保留完整解压 SHA-256 验证后替换原文件的流程。

当前脚本支持 `--level 1..22`，默认 3。任务目录为：

```text
/home/agile/work/labs/data/tools/raw_mcap_archive_fast_20260923/
```

查看 `progress.json` 的 `completed`、`total`、`freed_bytes` 获取实际进度和释放量。
只有全部完成才生成 `completed.json`。前两条实测从 12.92 GB 压到 6.69 GB，
节省约 48%；这是初期样本，最终结果以任务记录为准。

## 此前已停止的高压缩等级任务

2026-09-24 按用户要求，对 100.90.202.124 上 2026/09/16、2026/09/23
两个目录中已完成录制的 232 个 MCAP 文件启动无损压缩；不包含 9 月 24 日数据。

使用 `scripts/labs_archive_raw_mcap.py`，Zstandard `--ultra -22 -T12 --check`。
压缩是文件级归档，不解码或修改 ROS 消息，不改变图像、时间戳、关节值或夹爪值。
每个文件先计算原始 SHA-256，压缩后完整解压计算 SHA-256，校验一致且确认原始
文件未被修改后，持久化压缩文件和验证记录，才移除未压缩副本。失败即停止；不会
删除尚未验证的原文件。JSON/YAML 元数据和 LeRobot 数据集保留。

机器人主机上的任务目录：

```text
/home/agile/work/labs/data/tools/raw_mcap_archive_20260924/
```

其中 `snapshot.json` 是固定输入清单，`archive.log` 是日志，`progress.json`
记录已完成数量与实际释放字节，全部完成时才生成 `completed.json`。
本文是启动记录，不代表 232 个文件已经全部压缩完成。

原文件旁生成 `mcap_0.mcap.zst` 和 `mcap_0.mcap.zst.verification.json`。
归档后的 MCAP 需要解压才能供原来的 ROS/转换工具使用；`metadata.yaml` 中的
原文件名保留，恢复后继续适用。在有足够空间的目录中恢复：

```bash
zstd -d mcap_0.mcap.zst -o mcap_0.mcap
sha256sum mcap_0.mcap
```

校验值应与相邻 verification JSON 的 `sha256_uncompressed` 相同。
解压命令保留压缩文件。不要把 `.zst` 仅改名成 `.mcap`。

32 MiB 样本的 22 级压缩结果：9/16 保留约 32.9%，9/23 保留约 42.5%；
这是抽样结果，最终压缩率以任务记录为准。最高等级的整批压缩可能耗时数十小时。
