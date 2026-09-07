#!/usr/bin/env bash
# Discriminating test: stop the wrist RealSense cameras on this host, then repeat the
# instrumented right-arm impedance activation. Cameras are sensors only; the ZED on
# 172.16.0.50 is left running (its traffic does not use the robot NIC).
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
L=log/live_20260905
imp() { timeout 10 ros2 control list_controllers -c /$1/controller_manager 2>/dev/null | grep impedance | awk '{print $NF}'; }

echo "=== right must be recovered and idle"
echo "right impedance=$(imp right) hw=$(timeout 15 ros2 control list_hardware_components -c /right/controller_manager 2>/dev/null | grep -i state | awk '{print $NF}')"
ST=$(timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | grep status_v1 | head -1); echo "$ST"
case "$ST" in *'"started":false,"fault":false'*) ;; *) echo SERVO_NOT_IDLE_ABORT; exit 1;; esac

echo "=== stop RealSense launch (by PID)"
RS=$(ps -eo pid,args | awk '/ros2 launch realsense2_camera rs_multi_camera_launch/ && !/awk/ {print $1}')
RSN=$(ps -eo pid,args | awk '/realsense2_camera_node/ && !/awk/ {print $1}')
echo "launch=$RS nodes=$RSN"
for P in $RS; do kill -INT "$P" 2>/dev/null; done; sleep 5
for P in $RSN; do kill "$P" 2>/dev/null; done; sleep 2
echo -n 'remaining realsense procs: '; ps -eo args | grep -c '[r]ealsense2_camera_node'
echo -n 'wrist topic still publishing? '; timeout 4 ros2 topic hz /wrist_camera_left/color/image_raw 2>/dev/null | grep -c average
sleep 5
echo "=== load now"; uptime; ps -eo pcpu,comm --sort=-pcpu | head -5

echo "=== instrumented right activation (no cameras)"
PYTHONPATH=src:$PYTHONPATH .venv/bin/python /tmp/rec_right.py $L/right_activation_nocam.jsonl 6 2>/dev/null &
REC=$!
sleep 2
PF=$(ros2 pkg prefix franka_fr3_arm_controllers)/share/franka_fr3_arm_controllers/config/controllers.yaml
if [ -z "$(imp right)" ]; then
  timeout 40 ros2 run controller_manager spawner joint_impedance_controller -c /right/controller_manager --param-file "$PF" 2>/dev/null | grep -i 'activated\|error' | tail -1
else
  timeout 30 ros2 control switch_controllers -c /right/controller_manager --activate joint_impedance_controller 2>/dev/null | tail -1
fi
wait $REC 2>/dev/null
sleep 2
echo "right impedance after=$(imp right)"
echo -n "reflex lines in right3: "; grep -c 'reflex' $L/driver_right3.log
grep -i 'reflex\|q_goal' $L/driver_right3.log | tail -2 | cut -c1-200
PYTHONPATH=src:$PYTHONPATH .venv/bin/python - $L/right_activation_nocam.jsonl <<'EOF'
import json, sys, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
t0 = rows[0]["t"]
js = [r for r in rows if r["k"] == "js"]; rs = [r for r in rows if r["k"] == "rs"]
print("samples js", len(js), "rs", len(rs), "span s", round(rows[-1]["t"]-t0, 2))
if js:
    dq = np.array([r["dq"] for r in js]); t = np.array([r["t"]-t0 for r in js])
    print("max |dq| rad/s", round(float(np.abs(dq).max()), 4)); gaps = np.diff(t); print("js gap max ms", round(float(gaps.max()*1e3), 2))
modes = []
for r in rs:
    if not modes or modes[-1][1] != r["mode"] or modes[-1][2] != r["errs"]: modes.append((round(r["t"]-t0, 3), r["mode"], r["errs"]))
print("robot_mode/errors transitions", modes[:8])
rates = [r["rate"] for r in rs]; print("success_rate min/max", min(rates) if rates else None, max(rates) if rates else None)
EOF
echo "=== left still fine: impedance=$(imp left) errors=$(timeout 5 ros2 topic echo --once /left/franka_robot_state_broadcaster/robot_state 2>/dev/null | grep -c ': true')"
