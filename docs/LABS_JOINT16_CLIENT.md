# Labs joint16 绝对关节推理

FastWAM 入口 `bash scripts/labs_control.sh` 默认连接
`wss://u730748-b58d-17b41c61.bjb1.seetacloud.com:8443/infer`，使用
`absolute_joint16` 和 `fastwam.msgpack.v1`，请求服务器默认策略。
不再查询 `/health`、`/info`，不绑定模型名称、policy 或 checkpoint。
更换模型但保持输入输出格式时不需要修改客户端。

返回 `H×16`（当前32行）完整保存，默认全部执行。可用 `--execute-steps N`
只执行前 N 行，剩余预测不排队。9Hz 下32行参考时长约3.556秒，结束后重新观测。
通用入口 `scripts/labs_joint16_control.sh` 需显式提供 URL。
SmolVLA 的末端入口接受 `H×20` 绝对末端位姿；C23 保留 delta14。

state 和 action 都按下列顺序，共 16 维：

```text
[左 joint1–7, 左夹爪, 右 joint1–7, 右夹爪]
 0:7          7       8:15         15
```

关节角为 rad 绝对位置。state 从实测关节消息按名字提取；夹爪沿用数据集的
knuckle 映射，二值 0=关闭、1=打开。模型 state 不做客户端归一化，返回 action
不做客户端反归一化。只对预测的两个夹爪值做 `>=0.5` 二值化，关节值不裁剪。

输入还有 head、wrist_left、wrist_right 三张 640×480 RGB PNG 和 task。
内部仍保留 state34、URDF 与 FK，用于已有观测校验、恢复起点和误差分析；
发给模型的只有原始 state16。执行 action 不调用 IK，也不进行 delta 累积或 Cartesian 重建。

## 启动

在机器人主机 `/home/agile/work/labs/data/tools/labs31_client`：

```bash
# 指定 joint16 模型地址，获取真实观测和预测，不发布机器人动作
bash scripts/labs_joint16_control.sh --infer --url ws://HOST:PORT/infer

# 回到 episode 0 起点，然后连续推理
bash scripts/labs_joint16_control.sh --restore --infer --url ws://HOST:PORT/infer \
  --max-chunks 100 --publish --enable-robot

# 只回初始位置
bash scripts/labs_joint16_control.sh --restore --publish --enable-robot
```

也可设置 `LABS_JOINT16_SERVER_URL`。在获得实际 joint16 地址前没有内置模型 URL；
脚本不会把 joint16 请求误发给现有 delta14 服务。默认数据集：
`/home/agile/work/labs/data/lerobot_joint16/labs_fr3_joint16_20260916`，可用
`LABS_JOINT16_DATASET` 或 `--dataset` 覆盖。关节回零使用该 episode 首行**实测**关节，
保持现有夹爪状态，不以 action 目标冒充初始 state。

## 接口

二进制 MessagePack。通用 joint16 脚本默认子协议 `smolvla.msgpack.v1`，
通过 `--wire-protocol NAME`（兼容 `--joint16-protocol NAME`）指定其他子协议。
请求含 `request_id`、`state`、`images`、`task`，不发送具体模型或策略名称。

响应需回显 `request_id` 并包含有限 `actions[H,16]`，其中 1 ≤ H ≤ 256。
若有 `action`，需等于首行；若有 `chunk_size`，需等于实际行数。
`prediction_horizon`、`n_action_steps`、模型和 checkpoint 名称不限制执行长度。

选定入口表示输入原始 state16、输出 rad 绝对关节位置，不做归一化。
服务若声明归一化、delta、错误维度、排列或单位，客户端会拒绝；可选物理
声明不能与所选格式冲突。没有标签来源声明时不会推断这些输出是录制控制指令。
完整原始响应保留在记录中，不因模型元信息变化而拒绝推理。

断线或请求超时会重连一次，重新采集观测再发新请求，不重发机器人命令。
详见 `LABS_CLIENT.md`。

## 执行和记录

所选行数的绝对关节位置与两个夹爪序列直接交给站点 relay，命令 schema 是
`labs_fr3_absolute_joint16_command_v1`，integration 是 `absolute_joint16_v1`。
整个 chunk 先经过连续轨迹规划检查，再以 100 Hz 向 follower 发布，
参考行频率 9 Hz（32/9 秒加收敛尾段）。每行不是一次启动停止。

保留物理 URDF 关节限位、相邻目标关节跳变 0.25 rad、跟踪误差 0.15 rad、
速度 0.8 rad/s、加速度 2 rad/s²、jerk 20 rad/s³ 等检查。
动作段完成不再要求终点位置误差小于 0.05 rad；轨迹播放结束后，仅等待实测
关节速度不超过 0.02 rad/s 并连续保持 0.5 秒，最多等待 3 秒。
恢复数据集起点和启动保持仍分别校验其目标位置。
直接关节目标不再使用 Cartesian 平移/旋转范围或 IK 残差检查。
连续插值若越过关节限位仍会拒绝。现有规划器不构成碰撞校验。

必须同时提供 `--publish --enable-robot` 才发布到站点 relay；客户端不直接发布
controller command topic。升级后的 relay 会声明支持 joint16 schema；客户端会在
发布前拒绝复用不支持该 schema 的旧 relay。重载 relay 时必须先停用 follower，
新 relay 建立当前位置保持后才能重新激活。

Ctrl+C/正常退出/错误时保存 `actions.msgpack`：包含实际发送的 state16、内部 state34、
原始 action16、实际绝对目标、约 20 Hz 关节反馈、完成状态和关节误差。
兼容 `scripts/labs_inspect_actions.sh`；可视化中的 Cartesian 目标由记录的绝对关节
做离线 FK 得到，不参与执行。

## 2026-09-20 部署验证

已部署到 `agile@100.90.202.124:/home/agile/work/labs/data/tools/labs31_client`。
139 项相关测试通过，包含真实本机 WebSocket 收发、16 维排列、32 行原值传递、
直接关节规划、关节跳变/限位、双发布门禁、恢复起点、记录与离线查看，及旧入口回归。
Ruff、TOML、shell 语法及 diff 检查通过。

机器人端使用真实三路相机和实测关节，连接本机 echo 测试服务，验证实际发送 state16、
接收 32×16、构造绝对目标和无 IK 的规划流程；没有发布测试轨迹。
该测试不代表远端模型推理已验证，远端地址仍待提供。
记录：`outputs/joint16_live_camera_echo_dryrun_20260920/actions.msgpack`。

重载共享 relay 前先停用了两个 follower；新 relay 建立实测当前位置保持后重新激活。
连续 5 秒检查保持就绪、无故障，并声明支持 `labs_fr3_absolute_joint16_command_v1`。
激活时最大保持误差约 0.000106 rad。旧部署备份：
`outputs/joint16_deploy_backup_20260920/before.tar.gz`。
