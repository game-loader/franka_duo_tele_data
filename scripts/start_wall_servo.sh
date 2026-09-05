#!/bin/bash
# 墙壁伺服控制快速启动脚本

echo "=========================================="
echo "墙壁伺服控制器 - 快速启动"
echo "=========================================="
echo ""
echo "控制流程："
echo "1. 向前运动直到距离墙壁 0.5m"
echo "2. 顺时针旋转 90°"
echo "3. 向后运动直到距离墙壁 0.5m"
echo "4. 向左横向移动（5秒）"
echo ""
echo "=========================================="
echo ""

# 检查是否需要自定义参数
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h              显示此帮助信息"
    echo "  --distance <值>         设置目标墙壁距离（米，默认0.5）"
    echo "  --speed <值>            设置线速度（m/s，默认0.15）"
    echo "  --angular <值>          设置角速度（rad/s，默认0.3）"
    echo "  --left-time <值>        设置向左移动时间（秒，默认5.0）"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认参数"
    echo "  $0 --distance 0.6 --speed 0.2         # 自定义距离和速度"
    echo ""
    exit 0
fi

# 解析参数
TARGET_DISTANCE="0.5"
LINEAR_SPEED="0.15"
ANGULAR_SPEED="0.3"
LEFT_TIME="5.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --distance)
            TARGET_DISTANCE="$2"
            shift 2
            ;;
        --speed)
            LINEAR_SPEED="$2"
            shift 2
            ;;
        --angular)
            ANGULAR_SPEED="$2"
            shift 2
            ;;
        --left-time)
            LEFT_TIME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "当前参数配置："
echo "  目标墙壁距离: ${TARGET_DISTANCE} m"
echo "  线速度: ${LINEAR_SPEED} m/s"
echo "  角速度: ${ANGULAR_SPEED} rad/s"
echo "  向左移动时间: ${LEFT_TIME} s"
echo ""
echo "准备启动... 按 Ctrl+C 可随时停止"
echo "=========================================="
echo ""

# 等待3秒让用户准备
for i in 3 2 1; do
    echo "启动倒计时: $i 秒..."
    sleep 1
done
echo ""

# 检查是否在ROS2环境中
if ! command -v ros2 &> /dev/null; then
    echo "错误: 未找到 ros2 命令"
    echo "请先 source ROS2 环境:"
    echo "  source /opt/ros/humble/setup.bash"
    exit 1
fi

# 启动控制节点
echo "正在启动墙壁伺服控制器..."
ros2 run python3 "$(dirname "$0")/wall_servo_control.py" \
    --ros-args \
    -p target_wall_distance:=${TARGET_DISTANCE} \
    -p linear_speed:=${LINEAR_SPEED} \
    -p angular_speed:=${ANGULAR_SPEED} \
    -p left_movement_duration:=${LEFT_TIME}
