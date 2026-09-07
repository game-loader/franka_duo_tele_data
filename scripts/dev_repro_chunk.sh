#!/usr/bin/env bash
# Offline reproduction (isolated domain, no hardware) of the chunk that preceded the right-arm
# reflex: seed fake joints at the arms' last servo targets, feed the saved chunk to a dry servo,
# record the right target stream and report per-joint jump and peak velocity.
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
export ROS_DOMAIN_ID=221
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 2; for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT
ROUND="$1"; LQ="$2"; RQ="$3"
python3 /tmp/fake_joints.py "{\"left\": $LQ, \"right\": $RQ}" >/dev/null 2>&1 &
PIDS+=($!)
sleep 1
ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:=0.1 max_tracking_error_rad:=10.0 idle_follow_timeout_s:=2.0 > /tmp/servo_repro.log 2>&1 &
PIDS+=($!)
sleep 9
cat > /tmp/repro_chunk.py <<'EOF'
import rclpy, sys, json, time, threading, numpy as np
sys.path.insert(0, "src")
from pathlib import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from franka_duo_tele_data.rgb20d_io import RGB20DContract
from franka_duo_tele_data.joint_servo_client import chunk_payload
c = RGB20DContract(Path("datasets/franka_duo_lerobot_rgb20d_v1"))
r = json.load(open(sys.argv[1])); actions = np.array(r["actions"], dtype=np.float32)
rclpy.init(); n = rclpy.create_node("repro"); pub = n.create_publisher(Float32MultiArray, "/franka_duo/joint_servo/action_chunk", 10)
tgt = {"left": [], "right": []}
for s in ("left", "right"):
    n.create_subscription(JointState, f"/franka_duo/joint_servo/{s}/target", lambda m, s=s: tgt[s].append((time.monotonic(), list(m.position), list(m.velocity))), 10)
th = threading.Thread(target=rclpy.spin, args=(n,), daemon=True); th.start()
t = time.time()
while pub.get_subscription_count() < 1 and time.time() - t < 5: time.sleep(0.05)
time.sleep(0.5)
link0 = np.stack([c.action_spec.to_link0_action(a) for a in actions]); data, dims, off = chunk_payload(link0, 0)
m = Float32MultiArray(data=data); m.layout.dim = [MultiArrayDimension(label="rows", size=dims[0], stride=dims[0]*20), MultiArrayDimension(label="action", size=20, stride=20)]; m.layout.data_offset = 0
n0 = len(tgt["right"]); pub.publish(m); time.sleep(3.0)
for s in ("left", "right"):
    rows = tgt[s]; q = np.array([x[1] for x in rows]); v = np.array([x[2] for x in rows])
    q0 = q[n0-1] if n0 > 0 else q[0]
    print(s, "latched q", q0.round(3).tolist())
    print(s, "max |q - latched| per joint over chunk start (3 s):", np.abs(q[n0:] - q0).max(axis=0).round(3).tolist())
    print(s, "peak |velocity| per joint (rad/s):", np.abs(v[n0:]).max(axis=0).round(3).tolist())
    dq = np.diff(q[n0:], axis=0) / 0.001; print(s, "peak |dq/dt| from positions:", np.abs(dq).max(axis=0).round(3).tolist())
rclpy.shutdown()
EOF
timeout 30 .venv/bin/python /tmp/repro_chunk.py "$ROUND" 2>&1 | grep -v "type hash"
echo '=== servo repro log'
grep -i 'chunk\|IK\|drop\|fault' /tmp/servo_repro.log | grep -v launch | tail -4 | cut -c1-200
