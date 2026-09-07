#!/usr/bin/env bash
# Isolated-domain (no hardware) dry-run of the joint servo with recorded actions.
# Only kills processes it started itself; never touches a production servo.
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
export ROS_DOMAIN_ID=221
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 2; for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; PIDS=(); }
trap cleanup EXIT

cat > /tmp/fake_joints.py <<'EOF'
import rclpy, sys, json
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
Q = json.loads(sys.argv[1])
class N(Node):
    def __init__(self):
        super().__init__("fake_joints")
        self.p = {s: self.create_publisher(JointState, f"/{s}/franka_robot_state_broadcaster/measured_joint_states", qos_profile_sensor_data) for s in ("left", "right")}
        self.create_timer(0.01, self.tick)
    def tick(self):
        for s, pub in self.p.items():
            m = JointState(); m.header.stamp = self.get_clock().now().to_msg()
            m.name = [f"{s}_fr3v2_joint{i}" for i in range(1, 8)]; m.position = Q[s]; pub.publish(m)
rclpy.init(); rclpy.spin(N())
EOF

cat > /tmp/send_chunks.py <<'EOF'
import rclpy, sys, time, threading, numpy as np
sys.path.insert(0, "src")
from pathlib import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String, Float32
from sensor_msgs.msg import JointState
from franka_duo_tele_data.rgb20d_io import RGB20DContract
from franka_duo_tele_data.replay_rgb20d import DatasetEpisodePolicy
from franka_duo_tele_data.joint_servo_client import ChunkPacer, chunk_payload, parse_status
c = RGB20DContract(Path("datasets/franka_duo_lerobot_rgb20d_v1")); p = DatasetEpisodePolicy(c, 0, horizon=32)
count = 60; rclpy.init(); n = rclpy.create_node("chunk_sender")
pub = n.create_publisher(Float32MultiArray, "/franka_duo/joint_servo/action_chunk", 10)
lock = threading.Lock(); st = {"s": None}; tgt = {"left": [], "right": []}; grip = []
def on_status(m):
    with lock: st["s"] = m.data
n.create_subscription(String, "/franka_duo/joint_servo/status", on_status, 10)
for s in ("left", "right"):
    n.create_subscription(JointState, f"/franka_duo/joint_servo/{s}/target", lambda m, s=s: tgt[s].append((time.monotonic(), list(m.position), list(m.velocity))), 10)
n.create_subscription(Float32, "/franka_duo/joint_servo/left/gripper", lambda m: grip.append(m.data), 10)
th = threading.Thread(target=rclpy.spin, args=(n,), daemon=True); th.start()
pacer = ChunkPacer(32, count); t0 = time.monotonic(); sent = 0; s = None; raw = None
while time.monotonic() - t0 < 120:
    time.sleep(0.033)
    with lock: raw = st["s"]
    s = parse_status(raw) if raw else None
    if s and s.fault: print("FAULT", s.fault_reason); break
    start = pacer.next_start(s)
    if start is not None:
        rows = np.stack([c.action_spec.to_link0_action(a) for a in p.actions[start:min(count, start + 32)]])
        data, dims, off = chunk_payload(rows, start); m = Float32MultiArray(data=data)
        m.layout.dim = [MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * 20), MultiArrayDimension(label="action", size=20, stride=20)]
        m.layout.data_offset = off; pub.publish(m); pacer.record(start); sent += 1
        print("sent chunk start", start, "servo step", None if s is None else round(s.step, 2), flush=True)
    if pacer.finished(s): print("FINISHED at step", s.step); break
print("chunks_sent", sent, "final", raw)
for s in ("left", "right"):
    rows = list(tgt[s])
    a = np.array([q for _, q, _ in rows]); vel = np.array([v for _, _, v in rows]); t = np.array([x for x, _, _ in rows])
    if len(a) < 10: print(s, "too few targets", len(a)); continue
    dt = 0.001; v = np.diff(a, axis=0) / dt; acc = np.diff(vel, axis=0) / dt; jerk = np.diff(acc, axis=0) / dt
    rx = np.diff(t)
    print(f"{s}: targets={len(a)} rx_dt_max={rx.max()*1e3:.1f}ms |v|max={np.abs(v).max():.3f} |vel_field|max={np.abs(vel).max():.3f} |a|max={np.abs(acc).max():.3f} |j|p99={np.percentile(np.abs(jerk),99):.2f} |j|max={np.abs(jerk).max():.2f} travel={np.abs(a[-1]-a[0]).max():.4f}rad")
print("gripper msgs", len(grip), grip[:3])
rclpy.shutdown()
EOF

# 1) IK seed for the first action via the PTP tool dry-run from a ready pose
Q0='{"left": [0.0,-0.785,0.0,-2.356,0.0,1.571,0.785], "right": [0.0,-0.785,0.0,-2.356,0.0,1.571,0.785]}'
python3 /tmp/fake_joints.py "$Q0" >/dev/null 2>&1 &
PIDS+=($!)
sleep 1
timeout 60 ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py action_file:=$PWD/outputs/rgb20d_start_action.json action_index:=0 max_joint_delta_rad:=3.0 > /tmp/ptp_dry.log 2>&1
LQ=$(grep -o 'left_arm q=\[[^]]*\]' /tmp/ptp_dry.log | head -1 | sed 's/left_arm q=//')
RQ=$(grep -o 'right_arm q=\[[^]]*\]' /tmp/ptp_dry.log | head -1 | sed 's/right_arm q=//')
echo "IK seed left=$LQ right=$RQ"
cleanup; sleep 1
if [ -z "$LQ" ] || [ -z "$RQ" ]; then echo NO_IK; grep -i 'error\|fatal' /tmp/ptp_dry.log | head; exit 1; fi

# 2) servo dry-run seeded at the first-action IK, 0.1x speed
python3 /tmp/fake_joints.py "{\"left\": $LQ, \"right\": $RQ}" >/dev/null 2>&1 &
PIDS+=($!)
sleep 1
ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:=0.1 max_tracking_error_rad:=10.0 enable_gripper:=true > /tmp/servo.log 2>&1 &
PIDS+=($!)
sleep 6
timeout 150 .venv/bin/python /tmp/send_chunks.py 2>&1 | tail -20
echo '=== SERVO LOG'
grep -v '^\[INFO\] \[launch\]' /tmp/servo.log | grep -i 'chunk\|fault\|error\|drop\|ready' | head -20
