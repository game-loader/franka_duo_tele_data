#!/usr/bin/env bash
# Instrumented right-arm impedance activation: record 1 kHz joint state, robot_state and gello
# targets around the switch, then report whether a reflex occurred and what moved.
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
L=log/live_20260905
imp() { timeout 10 ros2 control list_controllers -c /$1/controller_manager 2>/dev/null | grep impedance | awk '{print $NF}'; }
echo "right impedance before=$(imp right)"
ST=$(timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | grep status_v1 | head -1); echo "$ST"
case "$ST" in *'"started":false,"fault":false'*) ;; *) echo SERVO_NOT_IDLE_ABORT; exit 1;; esac
cat > /tmp/rec_right.py <<'EOF'
import rclpy, sys, time, json, threading
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaRobotState
from rclpy.qos import qos_profile_sensor_data
out = open(sys.argv[1], "w"); dur = float(sys.argv[2])
rclpy.init(); n = rclpy.create_node("rec_right"); lock = threading.Lock()
def w(kind, **k):
    with lock: out.write(json.dumps({"t": time.monotonic(), "k": kind, **k}) + "\n")
n.create_subscription(JointState, "/right/franka_robot_state_broadcaster/measured_joint_states", lambda m: w("js", q=list(m.position), dq=list(m.velocity), tau=list(m.effort)), qos_profile_sensor_data)
n.create_subscription(JointState, "/right/gello/joint_states", lambda m: w("gello", q=list(m.position)), 10)
def rs(m):
    errs = [f for f in dir(m.current_errors) if not f.startswith("_") and getattr(m.current_errors, f) is True]
    w("rs", mode=int(m.robot_mode), errs=errs, tau_d=list(m.tau_j_d) if hasattr(m, "tau_j_d") else None, rate=float(m.control_command_success_rate))
n.create_subscription(FrankaRobotState, "/right/franka_robot_state_broadcaster/robot_state", rs, qos_profile_sensor_data)
th = threading.Thread(target=rclpy.spin, args=(n,), daemon=True); th.start()
w("mark", what="start"); time.sleep(dur); w("mark", what="end"); out.close(); rclpy.shutdown()
EOF
PYTHONPATH=src:$PYTHONPATH .venv/bin/python /tmp/rec_right.py $L/right_activation.jsonl 6 2>/dev/null &
REC=$!
sleep 2
date +%s.%N > $L/right_activation_t0.txt
PF=$(ros2 pkg prefix franka_fr3_arm_controllers)/share/franka_fr3_arm_controllers/config/controllers.yaml
if [ -z "$(imp right)" ]; then
  timeout 40 ros2 run controller_manager spawner joint_impedance_controller -c /right/controller_manager --param-file "$PF" 2>/dev/null | grep -i 'activated\|error' | tail -1
else
  timeout 30 ros2 control switch_controllers -c /right/controller_manager --activate joint_impedance_controller 2>/dev/null | tail -1
fi
wait $REC
echo "right impedance after=$(imp right)"
echo -n "reflex lines: "; grep -c 'reflex' $L/driver_right3.log
grep -i 'reflex\|q_goal\|Waiting for valid' $L/driver_right3.log | tail -3 | cut -c1-200
PYTHONPATH=src:$PYTHONPATH .venv/bin/python - $L/right_activation.jsonl <<'EOF'
import json, sys, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
t0 = rows[0]["t"]
js = [r for r in rows if r["k"] == "js"]; rs = [r for r in rows if r["k"] == "rs"]; ge = [r for r in rows if r["k"] == "gello"]
print("samples js", len(js), "rs", len(rs), "gello", len(ge), "span s", round(rows[-1]["t"]-t0, 2))
if js:
    dq = np.array([r["dq"] for r in js]); tau = np.array([r["tau"] for r in js]); t = np.array([r["t"]-t0 for r in js])
    imax = int(np.abs(dq).max(axis=1).argmax())
    print("max |dq| rad/s", round(float(np.abs(dq).max()), 4), "at t", round(float(t[imax]), 3), "joint", int(np.abs(dq[imax]).argmax()))
    print("max |tau_J|", np.abs(tau).max(axis=0).round(2).tolist())
    gaps = np.diff(t); print("js gap max ms", round(float(gaps.max()*1e3), 2), "count>2ms", int((gaps>0.002).sum()))
modes = []
for r in rs:
    if not modes or modes[-1][1] != r["mode"] or modes[-1][2] != r["errs"]: modes.append((round(r["t"]-t0, 3), r["mode"], r["errs"]))
print("robot_mode/errors transitions", modes[:10])
rates = [r["rate"] for r in rs]; print("success_rate min/max", min(rates) if rates else None, max(rates) if rates else None)
if js and ge:
    g = np.array(ge[-1]["q"]); q = np.array(js[-1]["q"]); print("final gello-measured max rad", round(float(np.abs(g-q).max()), 4))
EOF
