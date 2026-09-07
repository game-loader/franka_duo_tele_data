#!/usr/bin/env bash
# Reproduce a servo IK failure in an isolated domain using given measured joints as seed.
# Only kills processes it started itself; never touches a production servo.
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
export ROS_DOMAIN_ID=221
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 2; for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT
LQ='[-1.2271039485931396,-1.2427222728729248,1.4997297525405884,-2.7091259956359863,1.4235601425170898,0.8329773545265198,-1.5618269443511963]'
RQ='[1.305359,-1.571608,-1.472989,-2.587288,-1.697254,0.489400,1.249989]'
python3 /tmp/fake_joints.py "{\"left\": $LQ, \"right\": $RQ}" >/dev/null 2>&1 &
PIDS+=($!)
sleep 1
ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:=0.1 max_tracking_error_rad:=10.0 "$@" > /tmp/servo221.log 2>&1 &
PIDS+=($!)
sleep 7
cat > /tmp/probe_chunk.py <<'EOF'
import rclpy, sys, numpy as np, time
sys.path.insert(0, "src")
from pathlib import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from franka_duo_tele_data.rgb20d_io import RGB20DContract
from franka_duo_tele_data.replay_rgb20d import DatasetEpisodePolicy
from franka_duo_tele_data.joint_servo_client import chunk_payload
c = RGB20DContract(Path("datasets/franka_duo_lerobot_rgb20d_v1")); p = DatasetEpisodePolicy(c, 0, horizon=32)
rclpy.init(); n = rclpy.create_node("probe"); pub = n.create_publisher(Float32MultiArray, "/franka_duo/joint_servo/action_chunk", 10)
t = time.time()
while pub.get_subscription_count() < 1 and time.time() - t < 5: rclpy.spin_once(n, timeout_sec=0.05)
rows = np.stack([c.action_spec.to_link0_action(a) for a in p.actions[0:32]]); data, dims, off = chunk_payload(rows, 0)
m = Float32MultiArray(data=data); m.layout.dim = [MultiArrayDimension(label="rows", size=32, stride=640), MultiArrayDimension(label="action", size=20, stride=20)]; m.layout.data_offset = 0
pub.publish(m); rclpy.spin_once(n, timeout_sec=0.3); rclpy.shutdown()
EOF
for i in 1 2 3; do timeout 20 .venv/bin/python /tmp/probe_chunk.py; sleep 2; done
grep -i 'chunk\|IK\|drop\|seed\|target' /tmp/servo221.log | grep -v 'launch\|kdl_kin' | tail -12
