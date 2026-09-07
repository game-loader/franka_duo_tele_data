#!/usr/bin/env bash
# Recover right arm, set full collision behavior on both arms (ebim stack), then activate right
# with 1 kHz recording. Stops before inference if the right arm reflexes again.
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
L=log/live_20260905
imp() { timeout 10 ros2 control list_controllers -c /$1/controller_manager 2>/dev/null | grep impedance | awk '{print $NF}'; }
errs() { timeout 5 ros2 topic echo --once /$1/franka_robot_state_broadcaster/robot_state 2>/dev/null | grep -c ': true'; }
mode() { timeout 5 ros2 topic echo --once /$1/franka_robot_state_broadcaster/robot_state --field robot_mode 2>/dev/null | head -1; }

echo "=== 1. restart + recover right driver"
bash /tmp/dev_restart_right_driver.sh 2>&1 | grep -E 'recovery|Goal|mode=|errors=|remaining'

echo "=== 2. deactivate left impedance (collision behavior needs no active control)"
timeout 20 ros2 control switch_controllers -c /left/controller_manager --deactivate joint_impedance_controller 2>/dev/null | tail -1
sleep 1
for S in left right; do echo "$S impedance=$(imp $S) mode=$(mode $S) errors=$(errs $S)"; done

echo "=== 3. set full collision behavior (ebim stack env)"
( source /home/aup/ros2_ws_ebim/source_ebim_stack.sh >/dev/null 2>&1
  for arm in left right; do
    echo "-- $arm"
    timeout 30 ros2 service call /${arm}/service_server/set_full_collision_behavior \
      franka_msgs/srv/SetFullCollisionBehavior \
      '{lower_torque_thresholds_acceleration: [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0],
        upper_torque_thresholds_acceleration: [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0],
        lower_torque_thresholds_nominal: [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0],
        upper_torque_thresholds_nominal: [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0],
        lower_force_thresholds_acceleration: [35.0, 35.0, 35.0, 30.0, 30.0, 30.0],
        upper_force_thresholds_acceleration: [50.0, 50.0, 50.0, 42.0, 42.0, 42.0],
        lower_force_thresholds_nominal: [35.0, 35.0, 35.0, 30.0, 30.0, 30.0],
        upper_force_thresholds_nominal: [50.0, 50.0, 50.0, 42.0, 42.0, 42.0]}' 2>&1 | grep -v 'type hash' | tail -3
  done )

echo "=== 4. servo idle + single publishers"
ST=$(timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | grep status_v1 | head -1); echo "$ST"
case "$ST" in *'"started":false,"fault":false'*) ;; *) echo SERVO_NOT_IDLE_ABORT; exit 1;; esac
for S in left right; do echo -n "$S gello pubs: "; timeout 6 ros2 topic info /$S/gello/joint_states 2>/dev/null | grep -o 'Publisher count: [0-9]*'; done

echo "=== 5. instrumented right activation"
PYTHONPATH=src:$PYTHONPATH .venv/bin/python /tmp/rec_right.py $L/right_activation_collision.jsonl 6 2>/dev/null &
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
echo "right impedance after=$(imp right) mode=$(mode right) errors=$(errs right)"
grep -i 'reflex' $L/driver_right3.log | tail -1 | cut -c1-200
PYTHONPATH=src:$PYTHONPATH .venv/bin/python - $L/right_activation_collision.jsonl <<'EOF'
import json, sys, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; t0 = rows[0]["t"]
js = [r for r in rows if r["k"] == "js"]; rs = [r for r in rows if r["k"] == "rs"]
if js:
    dq = np.array([r["dq"] for r in js]); print("max |dq| rad/s", round(float(np.abs(dq).max()), 4))
modes = []
for r in rs:
    if not modes or modes[-1][1] != r["mode"] or modes[-1][2] != r["errs"]: modes.append((round(r["t"]-t0, 3), r["mode"], r["errs"]))
print("mode/errors transitions", modes[:8])
EOF
[ "$(imp right)" = "active" ] || { echo RIGHT_ACTIVATION_FAILED_STOP; exit 1; }

echo "=== 6. activate left"
timeout 30 ros2 control switch_controllers -c /left/controller_manager --activate joint_impedance_controller 2>/dev/null | tail -1
sleep 3
for S in left right; do echo "$S impedance=$(imp $S) mode=$(mode $S) errors=$(errs $S)"; done
echo RIGHT_AND_LEFT_ACTIVE_OK
