# Action Chunk控制方案完整对比与推荐

## 🎯 核心问题回顾

你遇到的问题：
> "第一种方法（PolicyCartesianPoseController）在实际使用中根本动不了，因为不够连续"

**根本原因**：
1. ❌ **单纯的Pose控制器** = 硬性位置跟踪 = 不连续、抖动
2. ❌ **没有速度/加速度限制** = 每个action之间可能有大跳变
3. ❌ **缺少compliance** = 不能吸收轨迹误差

## 📊 三种方案详细对比

| 方案 | CRISP Impedance | SERL Controllers | MoveIt Servo |
|------|----------------|------------------|--------------|
| **控制类型** | 力矩/阻抗 | 力矩/阻抗+限制 | 速度 |
| **输入** | PoseStamped | PoseStamped | TwistStamped |
| **输出** | 关节力矩 | 关节力矩 | 关节速度 |
| **平滑机制** | 阻抗动力学 | 阻抗+Reference Limiting | Butterworth滤波 |
| **Action chunk支持** | ✅ 原生 | ✅✅ 专门设计 | ⚠️ 需手动转换 |
| **真机验证** | 有（TUM） | ✅✅ 大量（Berkeley） | 中等 |
| **配置难度** | 中 | 中 | 中-高 |
| **IK需求** | 不需要 | 不需要 | 内置实时IK |
| **碰撞检测** | 无 | 无 | ✅ 内置 |
| **适用场景** | 通用学习策略 | Diffusion Policy/ACT | 需要碰撞检测 |
| **部署成熟度** | 新（2024） | ✅ 成熟（2023+） | 成熟 |

## 🏆 推荐方案：SERL Controllers

基于以下理由：

### 1. **专为Action Chunk设计**
SERL的Reference Limiting就是为了解决你的问题：

```cpp
// Reference Limiter伪代码
if (pose_jump_too_large(new_target, current_target)) {
    // 自动插值
    smooth_target = interpolate(current_target, new_target, max_velocity, dt);
} else {
    smooth_target = new_target;
}
```

### 2. **大量真机测试**
Berkeley RAIL在多个项目中使用：
- [SERL](https://github.com/rail-berkeley/serl_franka_controllers) - 在线RL
- Diffusion Policy真机实验
- ACT (Action Chunking Transformer)

### 3. **简单部署**
只需：
```bash
git clone https://github.com/rail-berkeley/serl_franka_controllers
colcon build
# 配置controller
# 运行
```

## 🚀 快速开始指南

### Step 1: 克隆并构建SERL

```bash
cd ~/franka_duo_workspace/src
git clone https://github.com/rail-berkeley/serl_franka_controllers.git

cd ~/franka_duo_workspace
colcon build --packages-select serl_franka_controllers
source install/setup.bash
```

### Step 2: 配置双臂controllers

创建 `franka_duo_serl_config.yaml`：

```yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # 1kHz
    
    serl_left_cartesian_impedance_controller:
      type: serl_franka_controllers/CartesianImpedanceController
    
    serl_right_cartesian_impedance_controller:
      type: serl_franka_controllers/CartesianImpedanceController

serl_left_cartesian_impedance_controller:
  ros__parameters:
    arm_id: "left"
    joints:
      - left_fr3v2_joint1
      - left_fr3v2_joint2
      - left_fr3v2_joint3
      - left_fr3v2_joint4
      - left_fr3v2_joint5
      - left_fr3v2_joint6
      - left_fr3v2_joint7
    
    # 阻抗参数（从保守值开始）
    translational_stiffness: 200.0
    rotational_stiffness: 10.0
    nullspace_stiffness: 0.5
    
    # Reference Limiting（关键！）
    max_translation_velocity: 0.5      # m/s
    max_rotation_velocity: 1.0         # rad/s
    max_translation_acceleration: 5.0  # m/s²
    max_rotation_acceleration: 10.0    # rad/s²
    max_translation_jerk: 50.0         # m/s³
    
    # 输入话题
    equilibrium_pose_topic: "/left/equilibrium_pose"
    
    # 安全边界
    workspace_limits:
      x: [0.3, 0.8]
      y: [-0.1, 0.5]
      z: [0.05, 0.6]

serl_right_cartesian_impedance_controller:
  ros__parameters:
    arm_id: "right"
    joints:
      - right_fr3v2_joint1
      - right_fr3v2_joint2
      - right_fr3v2_joint3
      - right_fr3v2_joint4
      - right_fr3v2_joint5
      - right_fr3v2_joint6
      - right_fr3v2_joint7
    
    translational_stiffness: 200.0
    rotational_stiffness: 10.0
    nullspace_stiffness: 0.5
    
    max_translation_velocity: 0.5
    max_rotation_velocity: 1.0
    max_translation_acceleration: 5.0
    max_rotation_acceleration: 10.0
    max_translation_jerk: 50.0
    
    equilibrium_pose_topic: "/right/equilibrium_pose"
    
    workspace_limits:
      x: [0.3, 0.8]
      y: [-0.5, 0.1]
      z: [0.05, 0.6]
```

### Step 3: 集成到eval代码

修改 `eval_franka_duo.py` 的主要部分：

```python
# 在imports中添加
from action_chunk_executor import ActionChunkExecutor

# 在run()函数中，创建publishers部分：
left_pose_pub = node.create_publisher(
    PoseStamped,
    "/left/equilibrium_pose",
    10
)
right_pose_pub = node.create_publisher(
    PoseStamped,
    "/right/equilibrium_pose",
    10
)

# 创建action chunk executor
chunk_executor = ActionChunkExecutor(
    chunk_horizon=16,
    overlap=8,
    blend_overlap=True  # 启用时间混合
)

need_inference = True

# 主循环
while True:
    observation = reader.next(timeout_s=1.0, require_state=require_state)
    
    # 只在需要时推理
    if need_inference:
        model_observation = build_policy_observation(observation, bundle.manifest)
        
        # 模型推理 - 假设返回 [batch, horizon, action_dim]
        action_output = bundle.predict(model_observation)
        
        # 如果是chunk，取出来
        if action_output.ndim == 2 and action_output.shape[0] == 16:
            action_chunk = action_output  # [16, 20]
        else:
            # 单个action，重复成chunk
            action_chunk = np.repeat(action_output[np.newaxis, :], 16, axis=0)
        
        chunk_executor.add_chunk(action_chunk)
        need_inference = False
    
    # 从chunk中取下一个action
    action, need_inference = chunk_executor.get_next_action()
    
    # 验证和转换
    model_action = action_spec.validate(action)
    
    # 转换为PoseStamped
    left_pose = action_to_pose_stamped(
        model_action[0:9],  # xyz + rot6d
        frame_id="left_fr3v2_link0",
        stamp=observation.stamp_ns
    )
    right_pose = action_to_pose_stamped(
        model_action[9:18],
        frame_id="right_fr3v2_link0",
        stamp=observation.stamp_ns
    )
    
    # 发布到SERL controllers
    left_pose_pub.publish(left_pose)
    right_pose_pub.publish(right_pose)
    
    # 夹爪单独处理
    publish_gripper_commands(model_action[18], model_action[19])
    
    # 记录和trace（保留原有逻辑）
    # ...
```

### Step 4: 辅助函数

```python
def action_to_pose_stamped(
    action: np.ndarray,  # [9]: xyz + rot6d
    frame_id: str,
    stamp: int
) -> PoseStamped:
    """将9D action转为PoseStamped."""
    from geometry_msgs.msg import PoseStamped
    from scipy.spatial.transform import Rotation
    
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp.sec = stamp // 1_000_000_000
    pose.header.stamp.nanosec = stamp % 1_000_000_000
    
    # 位置
    pose.pose.position.x = float(action[0])
    pose.pose.position.y = float(action[1])
    pose.pose.position.z = float(action[2])
    
    # 旋转：rot6d → matrix → quaternion
    rot6d = action[3:9]
    matrix = rot6d_to_matrix(rot6d)
    quat = Rotation.from_matrix(matrix).as_quat()  # [x,y,z,w]
    
    pose.pose.orientation.x = float(quat[0])
    pose.pose.orientation.y = float(quat[1])
    pose.pose.orientation.z = float(quat[2])
    pose.pose.orientation.w = float(quat[3])
    
    return pose

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D旋转表示转3x3矩阵."""
    x_raw = rot6d[0:3]
    y_raw = rot6d[3:6]
    
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    
    return np.column_stack([x, y, z])
```

### Step 5: 启动

```bash
# Terminal 1: 加载SERL controllers
ros2 control load_controller serl_left_cartesian_impedance_controller
ros2 control load_controller serl_right_cartesian_impedance_controller

ros2 control set_controller_state serl_left_cartesian_impedance_controller activate
ros2 control set_controller_state serl_right_cartesian_impedance_controller activate

# Terminal 2: 运行eval
ros2 run franka_duo_tele_data eval_franka_duo \
  --bundle /path/to/pretrained_model \
  --config configs/eval_config.yaml \
  --publish \
  --enable-robot
```

## ⚙️ 调试流程

### 1. 先dry-run测试
```bash
# 不加 --enable-robot
ros2 run franka_duo_tele_data eval_franka_duo --bundle /path/to/model
```

检查：
- action值是否合理
- 是否有NaN/Inf
- workspace是否在范围内

### 2. 检查controller状态
```bash
ros2 control list_controllers
# 应该看到两个controller都是"active"

ros2 topic hz /left/equilibrium_pose
# 应该是~30Hz
```

### 3. 可视化
```bash
rviz2
# Add -> TF
# Add -> PoseStamped -> Topic: /left/equilibrium_pose
# Add -> PoseStamped -> Topic: /right/equilibrium_pose
```

应该看到平滑的位姿轨迹，而非跳变。

### 4. 监控limited pose
```bash
ros2 topic echo /left/serl_cartesian_impedance_controller/limited_pose
```

对比输入的`equilibrium_pose`和输出的`limited_pose`，看Reference Limiter的效果。

## 🔧 常见问题与解决

### 问题1: 机器人完全不动

**可能原因**：
- Controller未激活
- 刚度太低
- 输入pose在workspace外

**解决**：
```bash
# 检查controller状态
ros2 control list_controllers

# 检查topic
ros2 topic echo /left/equilibrium_pose --once

# 尝试增加刚度
# 在config中: translational_stiffness: 400.0
```

### 问题2: 运动太慢/滞后

**原因**：速度限制太保守

**解决**：
```yaml
max_translation_velocity: 1.0     # 增大
max_translation_acceleration: 10.0
```

### 问题3: 仍有抖动

**原因**：
- Action chunk overlap太小
- 模型输出本身不平滑
- 刚度太高

**解决**：
```python
# 增加overlap
chunk_executor = ActionChunkExecutor(chunk_horizon=16, overlap=12)

# 启用blend
chunk_executor = ActionChunkExecutor(chunk_horizon=16, overlap=8, blend_overlap=True)
```

```yaml
# 降低刚度
translational_stiffness: 150.0
rotational_stiffness: 5.0
```

### 问题4: FCI通信错误

**原因**：1kHz循环被打断

**解决**：
- 使用实时内核
- 降低控制频率到500Hz
- 检查CPU负载

## 📚 相关资源

1. **SERL论文和代码**：
   - https://github.com/rail-berkeley/serl_franka_controllers
   - Paper: "Sample-Efficient Robotic Reinforcement Learning"

2. **CRISP（备选）**：
   - https://arxiv.org/pdf/2509.06819v1
   - GitHub: TUM-ICS/crisp

3. **MoveIt Servo文档**：
   - https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html

4. **Franka ROS2官方**：
   - https://github.com/frankaemika/franka_ros2

## 🎓 总结

你的问题根源是：**PolicyCartesianPoseController缺少Reference Limiting机制**。

**最佳解决方案**：使用SERL的Cartesian Impedance Controller with Reference Limiting。

**为什么不用JTC**：虽然你的仓库有JTC实现，但：
- 需要IK求解（可能有跳变）
- 需要缓冲16个actions
- 延迟较大
- 不如直接Cartesian控制平滑

**行动计划**：
1. ✅ 安装SERL controllers
2. ✅ 使用我提供的ActionChunkExecutor
3. ✅ 集成到eval_franka_duo.py
4. ✅ 从保守参数开始调试
5. ✅ 逐步提高速度限制

祝调试顺利！如果还有问题，请提供：
- Controller状态输出
- Topic频率
- 实际运动视频或日志
