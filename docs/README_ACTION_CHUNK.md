# Action Chunk控制实现指南 - 文档索引

## 📖 问题背景

当使用学习策略（如Diffusion Policy）控制Franka Duo Mobile时，模型通常输出**action chunk**（如16个未来actions），而不是单个action。简单的Cartesian pose控制器会导致：

- ❌ 运动不连续、抖动严重
- ❌ 速度/加速度突变
- ❌ 机器人"根本动不了"或响应异常

本指南提供了**三种经过真机验证的解决方案**，并详细说明了每种方案的原理、实现和调试方法。

---

## 📚 文档列表

### 1. [完整解决方案总结](./ACTION_CHUNK_SOLUTION.md) ⭐ **从这里开始**

**内容**：
- 问题根源深度分析
- 三种方案完整对比表
- **推荐方案：SERL Controllers**
- 快速开始指南（5步部署）
- 常见问题与解决方案
- 调试流程

**适合**：想快速了解全貌并直接上手的开发者

---

### 2. [SERL Cartesian Impedance Controller](./SERL_INTEGRATION.md) 🏆 **推荐方案**

**内容**：
- SERL项目背景（UC Berkeley RAIL实验室）
- Reference Limiting机制详解
- 完整配置文件示例
- Python集成代码
- 参数调优指南
- 与其他方案的详细对比

**为什么推荐SERL**：
- ✅ 专为action chunk设计（内置Reference Limiting）
- ✅ 大量真机验证（Diffusion Policy、ACT等）
- ✅ 部署简单，文档完善
- ✅ 支持双臂独立控制

**GitHub**: https://github.com/rail-berkeley/serl_franka_controllers

---

### 3. [CRISP Compliant Controllers](./CRISP_INTEGRATION.md) 🔬 **研究方向**

**内容**：
- CRISP控制器架构（TUM，2024）
- Cartesian Impedance控制原理
- 阻抗参数调优（刚度、阻尼）
- 力矩控制接口说明
- 与学习策略的集成

**适合场景**：
- 需要最新研究成果
- 想深入理解阻抗控制
- 需要高度可定制的控制器

**论文**: https://arxiv.org/pdf/2509.06819v1

---

### 4. [MoveIt Servo实时控制](./MOVEIT_SERVO_INTEGRATION.md) 🤖 **灵活方案**

**内容**：
- MoveIt Servo速度控制接口
- Action → Velocity转换算法
- Butterworth滤波器配置
- 碰撞检测集成
- 双臂协调控制

**适合场景**：
- 需要内置碰撞检测
- 已有MoveIt配置
- 想要灵活的控制架构

**特点**：
- ✅ 输入：速度命令（TwistStamped）
- ✅ 内置IK求解器
- ✅ 自动奇异点避免
- ⚠️ 需要手动实现pose→velocity转换

---

## 🚀 快速开始

### 最简部署（SERL方案）

```bash
# 1. 克隆SERL
cd ~/franka_duo_workspace/src
git clone https://github.com/rail-berkeley/serl_franka_controllers.git
cd ~/franka_duo_workspace
colcon build --packages-select serl_franka_controllers
source install/setup.bash

# 2. 使用提供的配置和代码
# 见 ACTION_CHUNK_SOLUTION.md 的 Step 2-4

# 3. 启动controllers
ros2 control load_controller serl_left_cartesian_impedance_controller
ros2 control set_controller_state serl_left_cartesian_impedance_controller activate

# 4. 运行eval
ros2 run franka_duo_tele_data eval_franka_duo \
  --bundle /path/to/model \
  --publish \
  --enable-robot
```

---

## 🔧 核心组件

### ActionChunkExecutor

位置：`src/franka_duo_tele_data/action_chunk_executor.py`

**功能**：
- 管理action chunk buffer
- 决定何时请求新推理
- 支持时间混合（temporal blending）
- 单元测试覆盖

**使用示例**：
```python
from action_chunk_executor import ActionChunkExecutor

executor = ActionChunkExecutor(
    chunk_horizon=16,    # chunk大小
    overlap=8,           # 何时请求新chunk
    blend_overlap=True   # 启用平滑混合
)

# 添加模型输出的chunk
executor.add_chunk(action_chunk)  # [16, 20]

# 逐个取出action (30Hz循环)
while True:
    action, need_inference = executor.get_next_action()
    
    if need_inference:
        # 请求新的模型推理
        new_chunk = model.predict(observation)
        executor.add_chunk(new_chunk)
    
    # 发布action到controller
    publish_action(action)
    rate.sleep()
```

---

## 📊 方案对比总结

| 特性 | SERL | CRISP | MoveIt Servo |
|------|------|-------|--------------|
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **部署难度** | 简单 | 中等 | 中等 |
| **真机验证** | 大量 | 中等 | 中等 |
| **Action Chunk原生支持** | ✅✅ | ✅ | ⚠️ 需转换 |
| **平滑机制** | Reference Limiting | 阻抗动力学 | 滤波器 |
| **碰撞检测** | ❌ | ❌ | ✅ |
| **学习成本** | 低 | 中 | 中-高 |
| **文档完整度** | ✅✅ | ✅ | ✅ |

---

## 🐛 故障排查

### 机器人不动？

**检查清单**：
1. Controller是否active？
   ```bash
   ros2 control list_controllers
   ```

2. 是否在发布pose？
   ```bash
   ros2 topic hz /left/equilibrium_pose
   ```

3. Pose是否在workspace内？
   ```bash
   ros2 topic echo /left/equilibrium_pose --once
   ```

4. 刚度是否太低？
   - 查看controller配置，translational_stiffness应该 > 100

### 运动不平滑？

**解决方法**：
1. 增加action chunk overlap
2. 启用blend_overlap
3. 降低控制器刚度
4. 检查Reference Limiting参数

详见：[ACTION_CHUNK_SOLUTION.md - 常见问题](./ACTION_CHUNK_SOLUTION.md#常见问题与解决)

---

## 📖 扩展阅读

### 学术论文

1. **SERL: Sample-Efficient Robotic Reinforcement Learning**
   - Berkeley RAIL, 2023
   - 详细描述了Reference Limiting机制

2. **CRISP: Compliant ROS2 Controllers**
   - TUM, 2024年9月
   - 论文: https://arxiv.org/pdf/2509.06819v1
   - 阻抗控制理论和实现

3. **Diffusion Policy**
   - Columbia & MIT, 2023
   - Action chunk生成原理

### 相关项目

1. **franka_ros2** (官方)
   - https://github.com/frankaemika/franka_ros2
   - Franka ROS2基础接口

2. **panda_impedance_control**
   - https://github.com/fida-121/panda_impedance_control
   - 单臂阻抗控制示例

3. **FrankaTeleop with MoveIt Servo**
   - https://github.com/gjcliff/FrankaTeleop
   - VR遥操作示例

---

## 💡 开发建议

### 调试顺序

1. **先仿真测试**
   - 在Gazebo中验证action chunk逻辑
   - 确保没有NaN/Inf

2. **Dry-run模式**
   - 不加 `--enable-robot`
   - 检查action值范围

3. **保守参数开始**
   - 低速度限制（0.3 m/s）
   - 中等刚度（200）
   - 小workspace

4. **逐步放宽**
   - 增加速度
   - 扩大workspace
   - 调整刚度

### 参数调优技巧

**如果想要更快的响应**：
- ↑ 刚度（translational_stiffness）
- ↑ 速度限制（max_translation_velocity）
- ↓ overlap（更频繁推理）

**如果想要更平滑**：
- ↓ 刚度
- ↑ 阻尼（damping）
- ↑ overlap
- 启用 blend_overlap

---

## 🤝 贡献

如果你发现文档有误或有改进建议，欢迎：
1. 提交Issue
2. 创建Pull Request
3. 分享你的调参经验

---

## 📧 支持

遇到问题时，请提供：
1. 使用的方案（SERL/CRISP/MoveIt Servo）
2. Controller状态输出
3. Topic频率（`ros2 topic hz`）
4. 配置文件内容
5. 错误日志或视频

---

## ✅ 检查清单

部署前确认：

- [ ] 已阅读 [ACTION_CHUNK_SOLUTION.md](./ACTION_CHUNK_SOLUTION.md)
- [ ] 选择了适合的控制方案
- [ ] 安装了所需的依赖
- [ ] 创建了ActionChunkExecutor实例
- [ ] 配置了controller参数
- [ ] 在仿真中测试通过
- [ ] 进行了dry-run测试
- [ ] 设置了安全的workspace限制
- [ ] 准备好了急停措施

**安全第一！祝你成功！🎉**
