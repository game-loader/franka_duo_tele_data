# Labs 双臂控制启动与恢复

在机器人主机 `/home/agile/work/labs/data/tools/labs31_client` 执行：

```bash
# 只查看状态
bash scripts/labs_robot_control.sh --status

# 启动双臂控制，在当前位置保持；已就绪则直接返回
bash scripts/labs_robot_control.sh --start --publish --enable-robot

# 更新 relay 或四任务起点配置后，在当前位置重新建立保持（不调用硬件恢复）
bash scripts/labs_robot_control.sh --start --restart-relay --publish --enable-robot

# 清除硬件错误及 relay 锁存错误，恢复双臂控制
bash scripts/labs_robot_control.sh --recover --publish --enable-robot
```

先退出推理和回放客户端。脚本不发送策略动作、不回 episode 起点、不打开夹爪。
需要回初始位置时，再使用推理脚本的 `--restore`。
FastWAM 用 `--restore --task 1/2/3/4` 选择任务。每个任务读取自身数据集的
episode 0/frame 0 实测双臂关节，映射见 `configs/labs_fr3_31/task_starts.json`。
relay 启动时同时读取四个起点，正常切换任务不需要重启 controller。

脚本使用站点已有的 `franka-robot` / `robotiq-gripper` 容器，停止的容器会启动；
不创建容器、不安装驱动。它停止已有 `controller-coordinator`，避免与推理控制竞争。
确认双臂 follower 和重力补偿控制器停用后，才停止本目录启动的 relay。
其他来源的目标话题发布者会阻止接管，不会被自动终止。
`--recover` 调用左右 `/action_server/error_recovery` 的 Franka ErrorRecovery action；
若急停、FCI 或人工恢复条件仍未解除，失败即退出，不跳过。

恢复阶段先以实测关节建立保持目标，再激活 follower，等待双侧 FOLLOWING、
relay holding/ready 和低于 0.01rad 的保持误差。激活失败会尝试停用已激活的 follower。
恢复不会重放上次失败动作。关节及观测时效保护继续生效。

状态报告：`outputs/labs_control_service/latest.json`。
relay 日志：`outputs/labs_relay/relay.log`。

2026-09-21 已用 `--start` 启动双臂：两个 follower 均 active/FOLLOWING，
relay ready/holding、无故障，最大保持误差约 0.00216rad。
按用户要求没有额外运行测试套件或主动制造故障验证 recovery 分支。
