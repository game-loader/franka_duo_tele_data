# MoveIt Servo 实时控制方案

## 概述

MoveIt Servo是MoveIt2提供的实时伺服控制接口，支持：
- Cartesian速度命令（TwistStamped）
- Joint速度命令
- Cartesian位姿流（Pose streaming）

参考：https://github.com/gjcliff/FrankaTeleop

## 为何适合Action Chunk

MoveIt Servo设计为**流式控制**，不是点到点：

```
Action Chunk → 逐个转换为速度命令 → MoveIt Servo → Joint velocity → 机器人
   [16个]           30Hz Twist              内部IK          平滑输出
```

## 安装和配置

### 1. 确保MoveIt2已安装
```bash
sudo apt install ros-jazzy-moveit-servo
```

### 2. 创建Servo配置文件

`moveit_servo_config.yaml`:
```yaml
# MoveIt Servo配置
moveit_servo:
  # 规划组
  move_group_name: "left_arm"  # 或 "right_arm"
  
  # 控制频率
  publish_period: 0.01  # 100Hz输出（1kHz太高，100Hz足够）
  
  # 输入源
  command_in_type: "speed_units"  # 接受TwistStamped（速度）
  
  # Cartesian速度限制
  linear_velocity_limit: 0.5   # m/s
  angular_velocity_limit: 1.0  # rad/s
  
  # 关节速度限制
  joint_velocity_limit: 2.0    # rad/s
  
  # IK求解器
  use_gazebo: false
  robot_link_command_frame: "left_fr3v2_link0"  # 命令参考系
  
  # 碰撞检测
  check_collisions: true
  collision_check_rate: 10.0
  
  # 平滑参数（关键！）
  smoothing_filter_plugin_name: "online_signal_smoothing::ButterworthFilterPlugin"
  
  # Butterworth滤波器参数
  butterworth_filter:
    low_pass_filter_coeff: 2.0  # 截止频率系数
```

### 3. 创建Franka Duo的MoveIt配置

如果官方没有提供，需要生成：

```bash
# 方法1：使用MoveIt Setup Assistant
ros2 launch moveit_setup_assistant setup_assistant.launch.py

# 方法2：基于FR3单臂配置修改
cp -r /opt/ros/jazzy/share/franka_fr3_moveit_config ~/franka_duo_moveit_config
# 然后编辑srdf添加左右两个planning group
```

## Python实现：将Action Chunk转为速度命令

创建 `moveit_servo_policy_executor.py`：

```python
#!/usr/bin/env python3
"""Use MoveIt Servo to execute policy action chunks."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from control_msgs.msg import JointJog
from moveit_msgs.srv import ServoCommandType
import numpy as np
from scipy.spatial.transform import Rotation

class MoveItServoPolicyExecutor(Node):
    def __init__(self):
        super().__init__('moveit_servo_policy_executor')
        
        # MoveIt Servo命令发布器
        self.left_twist_pub = self.create_publisher(
            TwistStamped,
            '/left_arm/servo_server/delta_twist_cmds',
            10
        )
        self.right_twist_pub = self.create_publisher(
            TwistStamped,
            '/right_arm/servo_server/delta_twist_cmds',
            10
        )
        
        # 当前状态
        self.current_left_pose = None
        self.current_right_pose = None
        
        # Action chunk buffer
        self.action_buffer = []
        self.buffer_index = 0
        
        # 控制循环：30Hz
        self.timer = self.create_timer(1.0/30.0, self.control_step)
        
    def add_action_chunk(self, actions: np.ndarray):
        """添加action chunk [horizon, 20]."""
        self.action_buffer = actions.copy()
        self.buffer_index = 0
    
    def control_step(self):
        if self.buffer_index >= len(self.action_buffer):
            return
        
        # 取当前action
        action = self.action_buffer[self.buffer_index]
        
        # 计算速度命令（这是关键！）
        left_twist = self.compute_velocity_command(
            action[0:9], 
            self.current_left_pose
        )
        right_twist = self.compute_velocity_command(
            action[9:18],
            self.current_right_pose
        )
        
        # 发布
        self.left_twist_pub.publish(left_twist)
        self.right_twist_pub.publish(right_twist)
        
        self.buffer_index += 1
    
    def compute_velocity_command(
        self, 
        target_action: np.ndarray,  # [9]: xyz + rot6d
        current_pose: PoseStamped
    ) -> TwistStamped:
        """将目标action转换为速度命令.
        
        关键思想：
        1. target_action是目标位姿
        2. 计算从current到target的差值
        3. 除以时间步长(dt=1/30)得到速度
        """
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = "left_fr3v2_link0"
        
        if current_pose is None:
            # 第一帧，零速度
            return twist
        
        dt = 1.0 / 30.0  # 30Hz
        
        # 1. 线速度 = (target_xyz - current_xyz) / dt
        target_xyz = target_action[0:3]
        current_xyz = np.array([
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z
        ])
        linear_velocity = (target_xyz - current_xyz) / dt
        
        # 限速
        max_linear_vel = 0.5  # m/s
        if np.linalg.norm(linear_velocity) > max_linear_vel:
            linear_velocity = linear_velocity / np.linalg.norm(linear_velocity) * max_linear_vel
        
        twist.twist.linear.x = float(linear_velocity[0])
        twist.twist.linear.y = float(linear_velocity[1])
        twist.twist.linear.z = float(linear_velocity[2])
        
        # 2. 角速度 = log(R_target * R_current^T) / dt
        # 从rot6d恢复旋转矩阵
        target_rot_matrix = self.rot6d_to_matrix(target_action[3:9])
        target_quat = Rotation.from_matrix(target_rot_matrix).as_quat()
        
        current_quat = np.array([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])
        
        # 旋转误差
        current_rot = Rotation.from_quat(current_quat)
        target_rot = Rotation.from_quat(target_quat)
        error_rot = target_rot * current_rot.inv()
        
        # 转为角速度（轴角表示 / dt）
        angle_axis = error_rot.as_rotvec()
        angular_velocity = angle_axis / dt
        
        # 限速
        max_angular_vel = 1.0  # rad/s
        if np.linalg.norm(angular_velocity) > max_angular_vel:
            angular_velocity = angular_velocity / np.linalg.norm(angular_velocity) * max_angular_vel
        
        twist.twist.angular.x = float(angular_velocity[0])
        twist.twist.angular.y = float(angular_velocity[1])
        twist.twist.angular.z = float(angular_velocity[2])
        
        return twist
    
    @staticmethod
    def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        """将6D旋转表示转为3x3矩阵."""
        x = rot6d[0:3]
        y = rot6d[3:6]
        
        x = x / np.linalg.norm(x)
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        
        return np.column_stack([x, y, z])

def main():
    rclpy.init()
    executor = MoveItServoPolicyExecutor()
    rclpy.spin(executor)
```

## 启动流程

```bash
# 1. 启动MoveIt Servo节点（左臂）
ros2 launch franka_duo_moveit_config moveit_servo_left.launch.py

# 2. 启动MoveIt Servo节点（右臂）
ros2 launch franka_duo_moveit_config moveit_servo_right.launch.py

# 3. 启动策略执行器
ros2 run franka_duo_tele_data moveit_servo_policy_executor
```

## MoveIt Servo的优势

### 1. **内置平滑滤波**
Butterworth滤波器自动平滑速度命令：

```
原始速度 → Butterworth滤波 → 平滑速度 → IK → 关节速度
  抖动         消除高频          连续
```

### 2. **自动碰撞检测**
```yaml
check_collisions: true
```
会自动避免自碰撞和环境碰撞。

### 3. **奇异点避免**
MoveIt Servo内置奇异点检测和处理。

### 4. **Joint velocity输出**
直接输出到`joint_velocity_controller`，无需手动IK。

## 调试和可视化

### 查看Servo状态：
```bash
ros2 topic echo /left_arm/servo_server/status
```

输出示例：
```
joint_velocity_limits_enforced: true
collision_detected: false
message: "Servo running normally"
```

### 可视化：
```bash
# RViz2
rviz2 -d $(ros2 pkg prefix franka_duo_moveit_config)/share/franka_duo_moveit_config/config/moveit.rviz

# 添加MarkerArray显示速度向量
# Topic: /left_arm/servo_server/twist_marker
```

## 参数调优

### 如果运动不够快：
```yaml
linear_velocity_limit: 1.0     # 增大
angular_velocity_limit: 2.0
joint_velocity_limit: 3.0
```

### 如果有抖动：
```yaml
# 降低截止频率，更强平滑
butterworth_filter:
  low_pass_filter_coeff: 1.0   # 从2.0降到1.0
```

### 如果频繁碰撞检测误报：
```yaml
collision_check_rate: 5.0   # 从10降到5
# 或临时禁用
check_collisions: false
```

## 与其他方案对比

| 特性 | Impedance Controller | MoveIt Servo |
|------|---------------------|--------------|
| 控制接口 | 位姿（Pose） | 速度（Twist） |
| 平滑方式 | 阻抗弹簧-阻尼 | Butterworth滤波 |
| IK求解 | 不需要（Cartesian） | 内置（实时） |
| 碰撞检测 | 无 | 内置 |
| 配置复杂度 | 低 | 中 |
| 真机测试 | 多 | 中等 |

## 常见问题

**Q: MoveIt Servo会自动处理action chunk吗？**
A: 不会。需要你逐个发送TwistStamped命令（30Hz）。

**Q: 如何处理夹爪？**
A: MoveIt Servo只控制臂，夹爪单独发到gripper action server。

**Q: 支持双臂协调吗？**
A: 需要启动两个独立的Servo实例（left_arm和right_arm）。

## 总结

MoveIt Servo适合：
- ✅ 需要碰撞检测
- ✅ 没有现成的Cartesian Impedance Controller
- ✅ 想要灵活的配置
- ⚠️ 需要自己实现pose→velocity转换
