# SERL Franka Controllers 方案

## 项目背景

SERL是UC Berkeley RAIL实验室专门为Franka机器人的在线强化学习和学习策略（包括Diffusion Policy）设计的控制器。

GitHub: https://github.com/rail-berkeley/serl_franka_controllers

## 核心特性

1. **Cartesian Impedance Controller with Reference Limiting**
   - 专为30Hz策略输出设计
   - 内置轨迹限制和安全边界
   - 平滑的力矩控制

2. **Joint Position Controller**（用于reset）
   - 快速回到初始位姿

## 安装

```bash
cd ~/franka_duo_workspace/src
git clone https://github.com/rail-berkeley/serl_franka_controllers.git
cd serl_franka_controllers

# 安装依赖
rosdep install --from-paths . --ignore-src -r -y

# 构建
cd ~/franka_duo_workspace
colcon build --packages-select serl_franka_controllers
source install/setup.bash
```

## 控制器配置

```yaml
# serl_cartesian_impedance_controller.yaml
serl_cartesian_impedance_controller:
  ros__parameters:
    arm_id: "left"  # 或 "right"
    
    # 关节列表
    joints:
      - left_fr3v2_joint1
      - left_fr3v2_joint2
      - left_fr3v2_joint3
      - left_fr3v2_joint4
      - left_fr3v2_joint5
      - left_fr3v2_joint6
      - left_fr3v2_joint7
    
    # Cartesian阻抗参数
    translational_stiffness: 200.0
    rotational_stiffness: 10.0
    
    # Reference limiting（关键特性！）
    max_translation_velocity: 0.5      # m/s
    max_rotation_velocity: 1.0         # rad/s
    max_translation_acceleration: 5.0  # m/s²
    max_rotation_acceleration: 10.0    # rad/s²
    
    # Nullspace配置
    nullspace_stiffness: 0.5
    
    # 输入接口
    equilibrium_pose_topic: "/equilibrium_pose"  # PoseStamped
```

## 为什么SERL适合你的场景

### 1. **Reference Limiting（参考限制）**
这是SERL的核心创新，解决了"action不连续"的根本问题：

```
策略输出 → Reference Limiter → 平滑轨迹 → Impedance Controller
   30Hz        限速/限加速         连续          1kHz力矩
```

即使策略输出有跳变，Reference Limiter会：
- 检查速度是否超过`max_translation_velocity`
- 如果超过，自动插值生成平滑过渡
- 保证加速度连续

### 2. **针对学习策略优化**
SERL在真实Franka上测试了：
- Diffusion Policy
- ACT (Action Chunking Transformer)
- 在线RL策略

所有这些都是输出action chunk的模型。

### 3. **安全边界**
内置workspace限制：
```yaml
workspace_limits:
  x: [0.3, 0.8]
  y: [-0.4, 0.4]
  z: [0.1, 0.6]
```

## 集成到你的代码

创建新的控制节点 `serl_policy_executor.py`：

```python
#!/usr/bin/env python3
"""SERL-based policy executor with action chunk support."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from franka_duo_tele_data.action_chunk_executor import ActionChunkExecutor
from franka_duo_tele_data.rl100_eval_policy import load_policy_bundle
import numpy as np

class SERLPolicyExecutor(Node):
    def __init__(self, bundle_path: str):
        super().__init__('serl_policy_executor')
        
        # 加载策略
        self.bundle = load_policy_bundle(bundle_path)
        
        # Action chunk执行器
        self.chunk_executor = ActionChunkExecutor(
            chunk_horizon=16,
            overlap=8,
            blend_overlap=True  # 平滑混合
        )
        
        # 发布到SERL controller
        self.left_pub = self.create_publisher(
            PoseStamped,
            '/left/equilibrium_pose',
            10
        )
        self.right_pub = self.create_publisher(
            PoseStamped,
            '/right/equilibrium_pose',
            10
        )
        
        # 30Hz控制循环
        self.timer = self.create_timer(1.0/30.0, self.control_step)
        self.need_inference = True
        
    def control_step(self):
        # 获取observation（省略细节）
        observation = self.get_observation()
        
        # 推理action chunk
        if self.need_inference:
            action_chunk = self.bundle.predict(observation)  # [16, 20]
            self.chunk_executor.add_chunk(action_chunk)
            self.need_inference = False
        
        # 取下一个action
        action, self.need_inference = self.chunk_executor.get_next_action()
        
        # 发布到SERL controllers
        self.publish_action(action)
    
    def publish_action(self, action: np.ndarray):
        """将20D action转换为双臂PoseStamped."""
        # 左臂: action[0:9]
        left_pose = self.action_to_pose(action[0:9], "left_fr3v2_link0")
        self.left_pub.publish(left_pose)
        
        # 右臂: action[9:18]
        right_pose = self.action_to_pose(action[9:18], "right_fr3v2_link0")
        self.right_pub.publish(right_pose)
        
        # 夹爪: action[18:20]（单独处理）
        self.publish_gripper(action[18], action[19])

def main():
    rclpy.init()
    executor = SERLPolicyExecutor('/path/to/bundle')
    rclpy.spin(executor)
```

## 启动流程

```bash
# 1. 启动SERL controllers
ros2 control load_controller serl_cartesian_impedance_controller_left
ros2 control load_controller serl_cartesian_impedance_controller_right

ros2 control set_controller_state serl_cartesian_impedance_controller_left activate
ros2 control set_controller_state serl_cartesian_impedance_controller_right activate

# 2. 启动策略执行器
ros2 run franka_duo_tele_data serl_policy_executor --bundle /path/to/model
```

## 调试技巧

### 检查reference limiting是否工作：
```bash
# 订阅实际发送给controller的limited pose
ros2 topic echo /left/serl_cartesian_impedance_controller/limited_pose
```

如果看到平滑变化而非跳变，说明reference limiter在工作。

### 可视化轨迹：
```bash
ros2 run rviz2 rviz2
# Add -> TF
# Add -> PoseStamped (topic: /equilibrium_pose)
```

## 参数调优

**问题：机器人反应太慢**
```yaml
# 增加允许的最大速度
max_translation_velocity: 1.0   # 从0.5提升到1.0
max_rotation_velocity: 2.0
```

**问题：仍有抖动**
```yaml
# 增加平滑时间窗口
smoothing_time_constant: 0.1  # 秒
# 或降低刚度
translational_stiffness: 150.0
```

## 与其他方案对比

| 特性 | PolicyCartesianPoseController | CRISP | SERL |
|------|------------------------------|-------|------|
| Reference Limiting | ❌ | 部分 | ✅ 专门设计 |
| 学习策略测试 | 无 | 少量 | 大量真机验证 |
| 文档完整度 | 中 | 高 | 高 |
| 部署难度 | 低 | 中 | 中 |
| Action chunk原生支持 | ❌ | ✅ | ✅✅ |

## 源码参考

关键实现在 `cartesian_impedance_controller.cpp` 的 `limitReferencePose()` 函数：

```cpp
// 伪代码示例
void CartesianImpedanceController::limitReferencePose(
    const Eigen::Vector3d& target_position,
    const Eigen::Quaterniond& target_orientation
) {
    // 计算速度
    Eigen::Vector3d velocity = (target_position - current_position_) / dt;
    
    // 限制速度
    if (velocity.norm() > max_translation_velocity_) {
        velocity = velocity.normalized() * max_translation_velocity_;
        target_position = current_position_ + velocity * dt;
    }
    
    // 类似地限制加速度...
    // 类似地限制旋转速度...
}
```

这就是为什么SERL能平滑处理action chunk的关键！
