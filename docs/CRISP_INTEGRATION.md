# CRISP Cartesian Impedance Controller 集成方案

## 安装CRISP

参考：https://arxiv.org/pdf/2509.06819v1

```bash
cd ~/franka_duo_workspace/src
git clone https://github.com/tum-ics/crisp.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select crisp_controllers
source install/setup.bash
```

## CRISP Controller配置

CRISP提供了专门的Cartesian Impedance Controller，适合action chunk控制：

```yaml
# crisp_cartesian_impedance_config.yaml
crisp_cartesian_impedance_controller:
  ros__parameters:
    # 关节配置
    joints:
      - left_fr3v2_joint1
      - left_fr3v2_joint2
      # ... 其他关节
    
    # Cartesian刚度（重要！影响跟踪平滑度）
    translational_stiffness: [200.0, 200.0, 200.0]  # xyz方向
    rotational_stiffness: [20.0, 20.0, 20.0]         # 旋转
    
    # 阻尼（2*sqrt(stiffness)原则）
    translational_damping: [28.3, 28.3, 28.3]
    rotational_damping: [8.9, 8.9, 8.9]
    
    # Nullspace刚度（7-DOF冗余控制）
    nullspace_stiffness: 10.0
    
    # 输入话题
    command_interface: "~/target_pose"  # geometry_msgs/PoseStamped
    state_interface: "~/state"
    
    # 力矩限制
    torque_limits:
      - 87.0  # 每个关节的最大力矩
      # ...
```

## 为何CRISP适合Action Chunk

与你当前的PolicyCartesianPoseController不同：

1. **Impedance vs Pose Control**：
   - ❌ Pose Control：机器人"硬"地跟踪目标位置 → 不连续、抖动
   - ✅ Impedance Control：像"弹簧-阻尼器"系统 → 平滑、柔顺

2. **力矩控制接口**：
   - CRISP直接输出关节力矩（`effort_command`）
   - 不是离散的位置命令

3. **内置滤波**：
   - 自动平滑action chunk之间的过渡
   - 处理30Hz输入 → 1kHz平滑输出

## 集成到你的代码

修改 `eval_franka_duo.py`：

```python
# 1. 发布到CRISP的输入接口
publisher = node.create_publisher(
    PoseStamped,
    "/left/crisp_cartesian_impedance_controller/target_pose",
    10
)

# 2. Action chunk执行器
from action_chunk_executor import ActionChunkExecutor
executor = ActionChunkExecutor(chunk_horizon=16, overlap=8)

# 3. 推理循环
while True:
    if need_inference:
        # 模型推理返回action chunk [16, 20]
        action_chunk = bundle.predict(observation)
        executor.add_chunk(action_chunk)
        need_inference = False
    
    # 从chunk取下一个action
    action, need_inference = executor.get_next_action()
    
    # 转换为PoseStamped发布
    pose_msg = action_to_pose_stamped(action)
    publisher.publish(pose_msg)
    
    rate.sleep()  # 30Hz
```

## 参数调优指南

**如果运动还是不够平滑**：

1. **降低刚度**（让机器人更"软"）：
   ```yaml
   translational_stiffness: [100.0, 100.0, 100.0]  # 从200降到100
   rotational_stiffness: [10.0, 10.0, 10.0]
   ```

2. **增加阻尼**（减少振荡）：
   ```yaml
   translational_damping: [40.0, 40.0, 40.0]  # 过阻尼
   ```

3. **调整action chunk overlap**：
   ```python
   # 增加overlap，更频繁推理
   executor = ActionChunkExecutor(chunk_horizon=16, overlap=12)
   ```

4. **添加速度限制**：
   ```yaml
   max_cartesian_velocity: 0.3  # m/s
   max_cartesian_acceleration: 1.0  # m/s²
   ```

## 与你当前方案对比

| 特性 | PolicyCartesianPoseController | CRISP Impedance |
|------|------------------------------|-----------------|
| 控制类型 | 位置控制 | 力矩/阻抗控制 |
| 平滑性 | 差（硬跟踪） | 优（柔顺） |
| Action chunk支持 | 无 | 原生支持 |
| 学习策略适配 | 需手动调参 | 专为此设计 |
| 真机稳定性 | 易抖动 | 稳定 |

## 故障排查

**问题1：机器人完全不动**
- 检查controller是否active：
  ```bash
  ros2 control list_controllers
  ```
- 检查刚度是否太低（<50）

**问题2：运动太慢/滞后**
- 增加刚度到400-600
- 检查control loop频率是否达到1kHz

**问题3：振荡/抖动**
- 增加阻尼系数
- 降低action发布频率（30Hz → 15Hz）
- 检查IK解是否有大跳变
