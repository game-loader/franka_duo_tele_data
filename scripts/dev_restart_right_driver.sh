#!/usr/bin/env bash
# Cleanly restart the right arm driver after a reflex, recover, and report state.
# Kills only ros2_control_node / launch processes in the /right namespace (not gripper).
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
LOG=log/live_20260905/driver_right3.log

right_pids() {
  ps -eo pid=,args= | awk '/ros2_control_node/ && /__ns:=\/right / && !/gripper/ {print $1}'
  ps -eo pid=,args= | awk '/ros2 launch franka_fr3_arm_controllers franka.launch.py/ && /namespace:=right/ {print $1}'
}
echo "=== right procs before"; ps -eo pid=,args= | awk '/__ns:=\/right / && /ros2_control_node/ && !/gripper/' | cut -c1-90
for P in $(right_pids); do kill "$P" 2>/dev/null; done; sleep 4
for P in $(right_pids); do kill -9 "$P" 2>/dev/null; done; sleep 1
echo -n "remaining right procs: "; right_pids | wc -l

nohup setsid ros2 launch franka_fr3_arm_controllers franka.launch.py arm_id:=fr3v2 arm_prefix:=right namespace:=right robot_ip:=172.16.16.11 load_gripper:=false joint_sources:=joint_states > "$LOG" 2>&1 < /dev/null &
sleep 14
echo "=== right driver3"; grep -i 'connected\|refused\|fail\|reflex\|activated' "$LOG" | tail -4 | cut -c1-160
echo "=== recovery"; timeout 30 ros2 action send_goal /right/action_server/error_recovery franka_msgs/action/ErrorRecovery '{}' 2>/dev/null | grep -i status | head -1
sleep 3
echo "=== right state"; timeout 15 ros2 control list_controllers -c /right/controller_manager 2>/dev/null | awk '{print $1, $NF}'
echo -n 'mode='; timeout 5 ros2 topic echo --once /right/franka_robot_state_broadcaster/robot_state --field robot_mode 2>/dev/null | head -1
echo -n 'errors='; timeout 5 ros2 topic echo --once /right/franka_robot_state_broadcaster/robot_state 2>/dev/null | grep -c ': true'
echo "=== gello vs measured, both arms"
for S in left right; do
  echo "-- $S"
  timeout 5 ros2 topic echo --once /$S/franka_robot_state_broadcaster/measured_joint_states --field position 2>/dev/null | head -1 | cut -c1-150
  timeout 5 ros2 topic echo --once /$S/gello/joint_states --field position 2>/dev/null | head -1 | cut -c1-150
  echo -n 'vel: '; timeout 5 ros2 topic echo --once /$S/franka_robot_state_broadcaster/measured_joint_states --field velocity 2>/dev/null | head -1 | cut -c1-150
done
echo "=== servo status"; timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | grep status_v1 | head -1
echo "=== left"; echo -n 'impedance='; timeout 10 ros2 control list_controllers -c /left/controller_manager 2>/dev/null | grep impedance | awk '{print $NF}'
echo -n 'errors='; timeout 5 ros2 topic echo --once /left/franka_robot_state_broadcaster/robot_state 2>/dev/null | grep -c ': true'
echo -n 'overruns left (last 500 lines): '; tail -500 log/live_20260905/driver_left.log | grep -c 'Overrun detected'
echo -n 'overruns right2 (last 500 lines): '; tail -500 log/live_20260905/driver_right2.log | grep -c 'Overrun detected'
