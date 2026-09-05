# 墙壁伺服控制脚本

基于激光雷达的机器人底盘伺服控制脚本，用于自动导航到墙壁附近。

## 功能描述

该脚本实现了以下自动控制流程：

1. **向前运动**：机器人向前移动，直到前方激光雷达检测到与墙壁的距离为0.5米
2. **顺时针旋转90°**：机器人原地顺时针旋转90度
3. **向后运动**：机器人向后移动，直到前方激光雷达再次检测到与墙壁的距离为0.5米
4. **向左运动**：机器人向左横向移动（默认持续5秒）

## 使用的接口

从 `local_navigation` 包中分析得出的关键接口：

### 订阅话题
- `/navigation/scan` (sensor_msgs/LaserScan): 合并后的前后激光雷达数据
- `/navigation/odom` (nav_msgs/Odometry): 机器人里程计信息，用于旋转角度控制

### 发布话题
- `/cmd_vel` (geometry_msgs/Twist): 速度控制命令
  - `linear.x`: 前后线速度 (m/s)
  - `linear.y`: 左右线速度 (m/s) - 全向底盘支持
  - `angular.z`: 旋转角速度 (rad/s)

## 安装

1. 确保脚本有执行权限：
```bash
chmod +x scripts/wall_servo_control.py
```

2. 如果需要集成到ROS2包，需要在 `setup.py` 中添加入口点。

## 运行方式

### 方式1: 直接运行Python脚本
```bash
cd /home/droid/project/franka_duo_tele_data
python3 scripts/wall_servo_control.py
```

### 方式2: 使用ROS2运行
```bash
ros2 run tmr_local_navigation wall_servo_control.py
```

### 方式3: 使用Launch文件（推荐）
```bash
ros2 launch scripts/wall_servo_control.launch.py
```

### 自定义参数运行
```bash
ros2 launch scripts/wall_servo_control.launch.py \
    target_wall_distance:=0.6 \
    linear_speed:=0.2 \
    angular_speed:=0.4 \
    left_movement_duration:=10.0
```

## 参数配置

| 参数名称 | 默认值 | 单位 | 说明 |
|---------|--------|------|------|
| `scan_topic` | `/navigation/scan` | - | 激光雷达话题 |
| `cmd_vel_topic` | `/cmd_vel` | - | 速度控制话题 |
| `odom_topic` | `/navigation/odom` | - | 里程计话题 |
| `target_wall_distance` | `0.5` | 米 | 目标墙壁距离 |
| `distance_tolerance` | `0.05` | 米 | 距离到达容差 |
| `rotation_angle` | `90.0` | 度 | 旋转角度 |
| `angle_tolerance` | `5.0` | 度 | 角度到达容差 |
| `linear_speed` | `0.15` | m/s | 线速度 |
| `angular_speed` | `0.3` | rad/s | 角速度 |
| `left_movement_duration` | `5.0` | 秒 | 向左移动持续时间 |

## 控制逻辑

### 状态机
脚本使用状态机模式实现控制流程：

```
MOVE_FORWARD → ROTATE_CLOCKWISE → MOVE_BACKWARD → MOVE_LEFT → COMPLETED
```

### 前方距离检测
- 使用激光雷达前方±30度扇区内的最小距离
- 过滤掉无效数据（inf、超出量程等）

### 旋转控制
- 基于里程计提供的四元数计算偏航角
- 使用角度归一化确保[-π, π]范围内的计算
- 顺时针旋转通过负角速度实现

### 全向运动
- 向前/向后：控制 `linear.x`
- 向左/向右：控制 `linear.y`（需要全向底盘支持）

## 安全注意事项

1. **首次运行前**：确保机器人周围有足够的安全空间
2. **障碍物检测**：脚本仅检测前方±30度扇区，注意侧方障碍物
3. **急停准备**：随时准备按下急停按钮或使用 `Ctrl+C` 终止
4. **速度限制**：根据实际环境调整 `linear_speed` 和 `angular_speed`
5. **监控日志**：观察终端输出了解当前状态和距离信息

## 故障排查

### 问题：机器人不移动
- 检查 `/cmd_vel` 话题是否正确
- 确认底盘控制器是否正常运行
- 验证速度参数是否过小

### 问题：无法获取激光雷达数据
- 检查激光雷达是否正常工作：`ros2 topic echo /navigation/scan`
- 确认 `dual_laser_merger` 节点是否运行

### 问题：旋转角度不准确
- 检查里程计话题是否正常：`ros2 topic echo /navigation/odom`
- 调整 `angle_tolerance` 参数
- 验证底盘的角速度控制是否准确

### 问题：前方距离检测异常
- 检查激光雷达数据质量
- 调整前方扇区角度范围（代码中的 `0.52` 弧度）
- 确认墙壁在激光雷达的有效检测范围内

## 调试技巧

1. **查看实时日志**：脚本会每0.5秒输出当前状态和距离信息
2. **可视化激光雷达**：使用RViz查看激光雷达数据
   ```bash
   ros2 run rviz2 rviz2
   ```
3. **监控话题**：
   ```bash
   ros2 topic list
   ros2 topic echo /cmd_vel
   ros2 topic hz /navigation/scan
   ```

## 扩展建议

1. **增加安全距离检测**：在侧方和后方也添加安全距离监控
2. **动态速度调整**：根据距离墙壁的远近动态调整速度
3. **PID控制**：使用PID控制器实现更平滑的距离保持
4. **多段路径**：扩展为支持更复杂的多段导航路径
5. **碰撞恢复**：添加碰撞检测和恢复逻辑

## 相关文件

- `scripts/wall_servo_control.py` - 主控制脚本
- `scripts/wall_servo_control.launch.py` - Launch启动文件
- `local_navigation/tmr_local_navigation/dual_laser_merger.py` - 激光雷达合并节点
- `local_navigation/config/nav2_params.yaml` - Nav2参数配置

## 依赖项

- ROS 2 (Humble或更高版本)
- rclpy
- sensor_msgs
- geometry_msgs
- nav_msgs
- 激光雷达驱动和合并节点
- 底盘控制器（支持全向运动）
