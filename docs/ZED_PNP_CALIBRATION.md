# ZED 头部相机外参标定（PnP，网页手动标点）

目的：求 `T_newbase_from_zed_optical`，即 ZED 光学坐标系到数据集 midpoint 基座的刚体变换，替代 manifest 里的 USD 名义值。相机与基座刚性连接，本体在平地上移动不需要重标。

代码：`src/franka_duo_tele_data/zed_pnp_calib.py`，入口 `scripts/run_zed_pnp_calib.sh`。测试 `tests/test_zed_pnp_calib.py`（ROS-free，含会话与网页路由）。

## 原理

- 夹爪闭合，两指并成一个尖。指尖在 EE 坐标系 z 轴上偏移 `tool_offset_m`（EE 原点若已设在指尖则为 0）。
- 每个样本：读 `current_pose`，经 manifest 的臂基座矩阵换到 midpoint 坐标，加偏移得 3D 点；在网页上点出该指尖在 ZED 图中的像素。
- 12 到 15 对点做 `solvePnP`（SQPNP 初解加迭代精化，8 点以上先 RANSAC 剔外点），输出外参、重投影误差、留一法误差、按臂统计。
- 运行时用像素射线与 `z = 桌面高度 + 托盘厚度 + 物体口沿高度` 的水平面求交，函数 `intersect_height`。

## 现场步骤

1. 两臂 driver 只跑 broadcaster，阻抗控制器停用。ZED 出图且 `camera_info` 正常。目标机无桌面会话，所以用网页。

   **拖臂与 FCI 的冲突（2026-09-06 现场确认）**：Desk 在 Execution 模式且 driver 连着 FCI 时，pilot 引导键拖不动手臂；切到 Programming 模式能拖，但 FCI 断开，driver 硬件接口掉成 unconfigured，`current_pose` 停发。因此每个标定点的循环是：Desk 切 Programming，拖臂到位，切回 Execution，在目标机运行

   ```bash
   bash scripts/reconnect_arm_state.sh          # 或 left / right
   ```

   它把硬件接口置 active、激活两个只读广播器并确认 `current_pose` 出数，打印 READY 后再在网页点 Capture。网页服务不必重启，订阅会自动恢复。该脚本在阻抗控制器 active 时拒绝执行。
2. 量 EE 原点到指尖的距离 `tool_offset_m`。EE 原点由 Desk 的 End Effector 配置决定，本机约在法兰 z 轴 0.174 m 处，接近闭合指尖；可读 `robot_state` 的 `f_t_ee` 确认。
3. 在目标机启动：

```bash
bash scripts/run_zed_pnp_calib.sh --tool-offset-m 0.0 --web 8765
```

4. Mac 浏览器打开 `http://172.16.0.100:8765/`。流程：选臂，手臂停稳，点 Capture 冻结一帧（手臂在动或位姿过期会被拒绝）。zoom 滑条放大到 4 倍，在图上点指尖接触点，出现黄十字，点 Accept 保存。Discard 丢弃本帧。Drop # 删已保存样本。Solve 试解，Solve + write 写文件。
5. 采集分布：覆盖托盘可能出现的整个区域；高度分两三层（桌面上方 0 到 25 cm）；左右臂各采一半；采一臂时另一臂不要挡住指尖。样本与原图实时落盘到 `outputs/zed_pnp/`，重启加 `--resume` 续采。
6. 结果写到 `configs/zed_pnp_calibration.json`。验收：
   - 重投影均值低于 2 像素，最大低于 4 像素（手动点选）。
   - 留一法 xy 误差最大低于 1 cm。
   - `per_side` 里两臂的 `mean_signed_residual_px` 都接近 0。一侧有系统性偏移说明该臂到 midpoint 的基座矩阵有误，此时可分臂各标一份。
   - `nominal_comparison` 与名义外参差在几厘米、几度以内；差得离谱多半是 `tool_offset_m` 方向或左右臂矩阵用反。

可选自动检测：贴饱和色标记后加 `--auto-detect`（默认橙色，红色加 `--hue-range 0 10 --hue-range 170 180`），网页会预填绿十字，点击可覆盖。不带 `--web` 的 stdin 模式只接受自动检测。

不用 ROS 重解：

```bash
bash scripts/run_zed_pnp_calib.sh --solve outputs/zed_pnp/samples.json
```

## 开工复核

```bash
bash scripts/run_zed_pnp_calib.sh --tool-offset-m 0.0 --check configs/zed_pnp_calibration.json --arm left
```

复核依赖自动检测，需要贴标记。拖臂到托盘区域任一位置，程序比较预测像素与检测像素，默认阈值 5 像素。

## 注意

- 不改数据集 manifest。运行时读取 `configs/zed_pnp_calibration.json`。
- `camera_info` 必须与 rectified 图配套，640×360。
- 目标机 `.venv` 用 `uv venv --system-site-packages` 建，复用系统 rclpy 与 cv2 4.6；装包用 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。启动必须走 wrapper 脚本，它会追加而不是覆盖 `PYTHONPATH`。
- 网页服务无鉴权，只在内网用，用完 Ctrl-C 停掉。

## 抓取（2026-09-07 新增）

`src/franka_duo_tele_data/grasp_cup_bowl.py`，入口 `scripts/run_grasp_cup_bowl.sh [cup|bowl] [right|left] [SPEED]`，默认 dry-run，`EXECUTE=1` 才发布。

- 感知：yolo11n-seg（权重 `outputs/zed_pnp/yolo11n-seg.pt`，CPU 预热后约 15 ms）。杯取置信度最高的 cup；碗取 bowl 中掩膜内深色像素比例最高者（咖啡豆），盘子因此被排除。口沿像素取掩膜上 45% 边界拟合的椭圆中心，用分臂外参与 `z = 桌面 + 托盘厚度 + 口沿高度` 求交。碗的抓取点在口沿上勺柄对侧。
- 轨迹：当前位姿 → 目标上方 10 cm → 下降到口沿下 5.5 cm → 闭合（6 行停顿 + 14 行闭合）→ 抬 10 cm。姿态复用数据集 episode 0 第 178/186 帧。另一臂全程保持。行步长 1 cm / 0.08 rad，走 `sanitize`同款校验后以绝对步 chunk 发给 servo。
- 执行顺序：`reconnect_arm_state.sh` → `start_servo_and_activate.sh 0.1`（wrapper 会自动调）→ `EXECUTE=1 bash scripts/run_grasp_cup_bowl.sh cup right 0.1`。右臂在阻抗激活瞬间可能 reflex（`communication_constraints_violation`）；处理：`reconnect_arm_state.sh right --no-restart`，`ros2 action send_goal /right/action_server/error_recovery`，再 `switch_controllers --activate joint_impedance_controller`，2026-09-07 01:20 一次成功。
- `--image FILE` 用保存帧做感知（仍读实时位姿），只允许 dry-run。
