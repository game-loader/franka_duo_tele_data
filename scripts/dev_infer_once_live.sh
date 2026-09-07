#!/usr/bin/env bash
# Restart servo/relay (controllers must be inactive), activate impedance controllers
# inside the idle-follow window, verify, then run one live SmolVLA inference.
set +u
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh >/dev/null 2>&1
source site/install/setup.bash
L=log/live_20260906
status() { timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | grep status_v1 | head -1; }
imp() { timeout 10 ros2 control list_controllers -c /$1/controller_manager 2>/dev/null | grep impedance | awk '{print $NF}'; }
errs() { timeout 5 ros2 topic echo --once /$1/franka_robot_state_broadcaster/robot_state 2>/dev/null | grep -c ': true'; }

echo "=== controllers must be inactive"; for S in left right; do echo "$S impedance=$(imp $S) errors=$(errs $S)"; done
for S in left right; do st="$(imp $S)"; [ "$st" = "inactive" ] || [ -z "$st" ] || { echo ABORT_${S}_NOT_INACTIVE; exit 1; }; done

echo "=== stop servo/relay by PID"
OLD=$(ps -eo pid=,args= | awk '/policy_chunk_joint_servo|gello_target_relay|joint_servo\.launch|gello_target_relay\.launch/ && !/awk/ {print $1}')
echo "$OLD" | tr '\n' ' '; echo
for P in $OLD; do kill "$P" 2>/dev/null; done; sleep 4; for P in $OLD; do kill -9 "$P" 2>/dev/null; done; sleep 1
echo -n 'remaining: '; ps -eo args= | grep -c '[p]olicy_chunk_joint_servo\|[g]ello_target_relay'

echo "=== start servo + relay"
nohup setsid ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:=0.1 enable_gripper:=true idle_follow_timeout_s:=20.0 > $L/servo6.log 2>&1 < /dev/null &
sleep 8
nohup setsid ros2 launch franka_duo_joint_servo gello_target_relay.launch.py enable_robot:=true enable_gripper:=true > $L/relay5.log 2>&1 < /dev/null &
sleep 3
for T in /left/gello/joint_states /right/gello/joint_states /franka_duo/joint_servo/status; do echo -n "$T: "; timeout 6 ros2 topic info $T 2>/dev/null | grep -i publisher | tr '\n' ' '; echo; done
ST=$(status); echo "$ST"
case "$ST" in *'"started":false,"fault":false'*'"idle_latched":false'*) ;; *) echo SERVO_NOT_IDLE_FOLLOWING_ABORT; exit 1;; esac
N=$(timeout 6 ros2 topic info /left/gello/joint_states 2>/dev/null | grep -o 'Publisher count: [0-9]*' | grep -o '[0-9]*'); [ "$N" = "1" ] || { echo GELLO_PUBLISHERS_${N}_ABORT; exit 1; }
echo "=== gello vs measured"; for S in left right; do timeout 5 ros2 topic echo --once /$S/gello/joint_states --field position 2>/dev/null | head -1 | cut -c1-110; timeout 5 ros2 topic echo --once /$S/franka_robot_state_broadcaster/measured_joint_states --field position 2>/dev/null | head -1 | cut -c1-110; done

echo "=== activate impedance (inside follow window)"
PF=$(ros2 pkg prefix franka_fr3_arm_controllers)/share/franka_fr3_arm_controllers/config/controllers.yaml
for S in left right; do
  if [ -z "$(imp $S)" ]; then
    timeout 40 ros2 run controller_manager spawner joint_impedance_controller -c /$S/controller_manager --param-file "$PF" 2>/dev/null | grep -i 'activated\|error' | tail -1
  else
    timeout 30 ros2 control switch_controllers -c /$S/controller_manager --activate joint_impedance_controller 2>/dev/null | tail -1
  fi
done
sleep 3
for S in left right; do echo "$S impedance=$(imp $S) errors=$(errs $S)"; done
for S in left right; do [ "$(imp $S)" = "active" ] || { echo ABORT_${S}_NOT_ACTIVE; grep -i 'reflex' $L/driver_${S}*.log | tail -1 | cut -c1-200; exit 1; }; done
echo "=== wait for idle latch"; sleep 12; status

echo "=== INFER ONCE + EXECUTE"
timeout 60 ros2 topic echo /franka_duo/joint_servo/status --field data > $L/status_infer.log 2>/dev/null &
timeout 60 ros2 topic echo /left/gripper/joint_states --field position > $L/grip_infer_left.log 2>/dev/null &
timeout 60 ros2 topic echo /right/gripper/joint_states --field position > $L/grip_infer_right.log 2>/dev/null &
PYTHONPATH=src:$PYTHONPATH timeout 150 .venv/bin/python -m franka_duo_tele_data.smolvla_once --dataset datasets/franka_duo_lerobot_rgb20d_v1 --output outputs/smolvla_once_live.json --publish --enable-robot 2>&1 | grep -v 'rmw_cyclonedds' | grep -v '^  \|Traceback' | tail -12 | cut -c1-700
echo INFER_EXIT=${PIPESTATUS[0]}
echo "=== servo log"; grep -i 'chunk [0-9]\|fault\|drop\|IK\|latched' $L/servo6.log | tail -5 | cut -c1-180
echo -n "max servo tracking: "; grep -o 'tracking_error_rad":[0-9.e-]*' $L/status_infer.log | cut -d: -f2 | sort -g | tail -1
for S in left right; do echo -n "$S gripper pos uniq: "; grep -o '\[[0-9.e-]*\]' $L/grip_infer_$S.log | awk '{printf "%.2f\n", substr($0,2)}' | uniq | tr '\n' ' '; echo; done
echo "=== final"; for S in left right; do echo "$S impedance=$(imp $S) errors=$(errs $S) mode=$(timeout 5 ros2 topic echo --once /$S/franka_robot_state_broadcaster/robot_state --field robot_mode 2>/dev/null | head -1)"; grep -i reflex $L/driver_${S}3.log $L/driver_${S}.log 2>/dev/null | tail -1 | cut -c1-160; done
status
