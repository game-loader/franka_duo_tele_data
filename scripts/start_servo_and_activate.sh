#!/usr/bin/env bash
# One-shot: restart the joint servo + gello relay (by exact PID) so the servo follows the
# current pose, then activate joint_impedance_controller on both arms inside the servo's
# idle-follow window and verify no reflex. Run this after scripts/ptp_home.sh and before
# scripts/run_smolvla_loop.sh.
#
# Refuses to run while an impedance controller is active (restarting the servo under an
# active controller would let the driver shut itself down). Deactivate first, or run
# ptp_home.sh, which already deactivates both.
#
# Usage: bash scripts/start_servo_and_activate.sh [PLAYBACK_SPEED] [--reconfigure]
# Default speed 0.3 gives 9 Hz actions. --reconfigure may deactivate an idle,
# healthy controller pair before restart, only when no policy publisher exists.
set -euo pipefail
SPEED="${1:-0.3}"
RECONFIGURE="${2:-}"
[[ -z "$RECONFIGURE" || "$RECONFIGURE" = "--reconfigure" ]] || { echo "Unknown option: $RECONFIGURE" >&2; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
L="$PWD/log/servo_start"; mkdir -p "$L" log/ptp_home
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_LOG="$L/run_$STAMP.log"
exec 9>log/ptp_home/.lock
flock -n 9 || { echo "ABORT: another arm startup/PTP operation is running" >&2; exit 1; }
exec > >(tee -a "$RUN_LOG" 9>&-) 2>&1
trap 'rc=$?; echo "ABORT: line $LINENO failed (exit $rc); see $RUN_LOG" >&2; exit "$rc"' ERR
echo "Servo startup log: $RUN_LOG"
[[ "$SPEED" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
  awk -v speed="$SPEED" 'BEGIN { exit !(speed > 0 && speed <= 1) }' || { echo "ABORT: speed must be in (0, 1]"; exit 1; }
set +u
source ~/tmr_env.sh
source site/install/setup.bash
set -u
controllers() {
  local result
  result=$(timeout 10 ros2 control list_controllers -c "/$1/controller_manager" 2>>"$RUN_LOG") || {
    echo "ABORT: $1 controller manager unavailable; run scripts/ptp_home.sh first" >&2; return 1;
  }
  printf '%s\n' "$result" | sed $'s/\033\\[[0-9;]*m//g'
}
imp() {
  local result
  result=$(controllers "$1") || return 1
  printf '%s\n' "$result" | awk '$1 == "joint_impedance_controller" {state=$NF} END {print state == "" ? "not_loaded" : state}'
}
errs() {
  local result
  result=$(timeout 5 ros2 topic echo --once "/$1/franka_robot_state_broadcaster/robot_state" \
    --field current_errors --qos-reliability best_effort 2>>"$RUN_LOG") || {
    echo "ABORT: $1 robot_state unavailable" >&2; return 1;
  }
  printf '%s\n' "$result" | awk '/: (true|false)$/ {fields++} /: true$/ {errors++} END {if (!fields) exit 1; print errors+0}'
}
mode() { timeout 5 ros2 topic echo --once "/$1/franka_robot_state_broadcaster/robot_state" --field robot_mode --qos-reliability best_effort 2>>"$RUN_LOG" | awk '/^[0-9]+$/ {print; found=1} END {exit !found}'; }
status() { timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>>"$RUN_LOG" | awk '/status_v1/ {print; found=1} END {exit !found}'; }
pubs() { timeout 6 ros2 topic info "$1" 2>>"$RUN_LOG" | awk '/Publisher count:/ {print $3; found=1} END {exit !found}'; }
servo_pids() {
  ps -eo pid=,args= | awk '
    $2 ~ /(^|\/)(policy_chunk_joint_servo|gello_target_relay)$/ {print $1; next}
    $2 ~ /(^|\/)(python[0-9.]*|ros2)$/ {
      for (i=3; i<NF-1; i++)
        if ($i == "launch" && $(i+1) == "franka_duo_joint_servo" &&
            ($(i+2) == "joint_servo.launch.py" || $(i+2) == "gello_target_relay.launch.py")) {print $1; break}
    }'
}

if [ "$RECONFIGURE" = "--reconfigure" ]; then
  left_state=$(imp left); right_state=$(imp right)
  if [ "$left_state" = active ] || [ "$right_state" = active ]; then
    [ "$(pubs /franka_duo/joint_servo/action_chunk)" = "0" ] || {
      echo "ABORT: stop the running policy publisher before reconfiguring servo"; exit 1;
    }
    echo "Waiting for the accepted plan to finish before changing servo configuration"
    deadline=$((SECONDS + 30))
    while true; do
      ST=$(status)
      phase=$(python3 -c 'import json,sys; s=json.loads(sys.argv[1]); print("fault" if s["fault"] else "holding" if s["holding"] else "moving")' "$ST")
      [ "$phase" != fault ] || { echo "ABORT: servo fault must be inspected before restart"; exit 1; }
      [ "$phase" = holding ] && break
      (( SECONDS < deadline )) || { echo "ABORT: current plan did not finish in time"; exit 1; }
      sleep 0.2
    done
    for S in left right; do
      if [ "$(imp "$S")" = active ]; then
        timeout 20 ros2 control switch_controllers -c "/$S/controller_manager" --deactivate joint_impedance_controller
      fi
    done
    sleep 1
  fi
fi

echo "=== 1. preconditions"
for S in left right; do
  st="$(imp $S)"; e="$(errs $S)"; echo "$S impedance=${st:-not_loaded} mode=$(mode $S) errors=$e"
  [ "$st" = "active" ] && { echo "ABORT: $S impedance controller is active; deactivate it first"; exit 1; }
  [ "$e" = "0" ] || { echo "ABORT: $S has robot errors; run error_recovery first"; exit 1; }
  [ "$(mode $S)" = "1" ] || { echo "ABORT: $S robot must be in idle mode before activation"; exit 1; }
  other=$(controllers "$S" | awk '$NF == "active" && $1 != "joint_state_broadcaster" && $1 != "franka_robot_state_broadcaster" {print $1}')
  [ -z "$other" ] || { echo "ABORT: $S has active command controllers: $other"; exit 1; }
done

echo "=== 1b. load/configure impedance controllers without activating"
PF=$(ros2 pkg prefix franka_fr3_arm_controllers)/share/franka_fr3_arm_controllers/config/controllers.yaml
for S in left right; do
  st=$(imp "$S")
  if [ "$st" = "not_loaded" ]; then
    timeout 40 ros2 run controller_manager spawner joint_impedance_controller -c "/$S/controller_manager" --param-file "$PF" --inactive
  elif [ "$st" = "unconfigured" ]; then
    timeout 20 ros2 control set_controller_state joint_impedance_controller inactive -c "/$S/controller_manager"
  fi
  [ "$(imp "$S")" = "inactive" ] || { echo "ABORT: $S impedance controller is not inactive"; exit 1; }
done

echo "=== 2. stop existing servo/relay by PID"
OLD=$(servo_pids)
if [ -n "$OLD" ]; then
  for P in $OLD; do kill "$P" 2>/dev/null || true; done
  sleep 4
  for P in $(servo_pids); do kill -9 "$P" 2>/dev/null || true; done
  sleep 1
fi
[ -z "$(servo_pids)" ] || { echo "ABORT: servo/relay processes still alive"; exit 1; }

echo "=== 3. start servo (speed $SPEED, grippers on) + relay"
nohup setsid ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:="$SPEED" commit_lead_steps:=0 blend_steps:=4 enable_gripper:=true idle_follow_timeout_s:=60.0 > "$L/servo_$STAMP.log" 2>&1 < /dev/null 9>&- &
sleep 9
nohup setsid ros2 launch franka_duo_joint_servo gello_target_relay.launch.py enable_robot:=true enable_gripper:=true > "$L/relay_$STAMP.log" 2>&1 < /dev/null 9>&- &
sleep 4
grep -q 'Ready:' "$L/servo_$STAMP.log" || { echo "ABORT: servo did not report Ready; see $L/servo_$STAMP.log"; exit 1; }
for T in /left/gello/joint_states /right/gello/joint_states /franka_duo/joint_servo/status /franka_duo/joint_servo/action_chunk; do
  n="$(pubs $T)"; echo "$T publishers=$n"
  case "$T" in */action_chunk) [ "$n" = "0" ] || { echo "ABORT: stop action-chunk publishers before activation"; exit 1; };; *) [ "$n" = "1" ] || { echo "ABORT: expected exactly one publisher on $T"; exit 1; };; esac
done
ST=$(status); echo "$ST"
case "$ST" in *'"started":false,"fault":false'*'"idle_latched":false'*) ;; *) echo "ABORT: servo not idle-following"; exit 1;; esac
echo "Checking live relay targets against measured joints before activation"
python3 - 2>>"$RUN_LOG" <<'PY'
import json
import math
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

rclpy.init()
node = rclpy.create_node("servo_activation_alignment_check")
latest = {}

def store(key, message):
    latest[key] = (message, time.monotonic())

try:
    for side in ("left", "right"):
        for kind, topic in (("target", "gello/joint_states"),
                            ("measured", "franka_robot_state_broadcaster/measured_joint_states")):
            node.create_subscription(JointState, f"/{side}/{topic}",
                lambda msg, key=(side, kind): store(key, msg), qos_profile_sensor_data)
    deadline = time.monotonic() + 5
    while len(latest) < 4 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if len(latest) != 4:
        raise RuntimeError("live joint targets/measured states missing")
    report = {}
    for side in ("left", "right"):
        values = {}
        for kind in ("target", "measured"):
            msg, received = latest[side, kind]
            age = (node.get_clock().now().nanoseconds - msg.header.stamp.sec * 10**9
                   - msg.header.stamp.nanosec) / 1e9
            if time.monotonic() - received > 0.2 or not -0.1 <= age <= 0.2:
                raise RuntimeError(f"{side} {kind} joints are stale")
            positions = dict(zip(msg.name, msg.position))
            values[kind] = [positions[f"{side}_fr3v2_joint{i}"] for i in range(1, 8)]
            if not all(math.isfinite(v) for v in values[kind]):
                raise RuntimeError(f"{side} {kind} joints contain invalid values")
        error = max(abs(a - b) for a, b in zip(values["target"], values["measured"]))
        if error > 0.003:
            raise RuntimeError(f"{side} relay target differs from measured joints by {error:.6f} rad")
        report[side] = {"max_target_error_rad": error}
    print(json.dumps(report), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
PY

echo "=== 4. activate impedance controllers"
for S in left right; do
  ST=$(status)
  case "$ST" in *'"started":false,"fault":false'*'"idle_latched":false'*) ;; *) echo "ABORT: servo left the idle-follow window before $S activation"; exit 1;; esac
  timeout 30 ros2 control switch_controllers -c "/$S/controller_manager" --activate joint_impedance_controller
  sleep 2
done
sleep 2
OK=1
for S in left right; do
  st="$(imp $S)"; echo "$S impedance=$st mode=$(mode $S) errors=$(errs $S)"
  [ "$st" = "active" ] && [ "$(mode $S)" = "2" ] && [ "$(errs $S)" = "0" ] || OK=0
done
if [ "$OK" != "1" ]; then
  echo "ACTIVATION FAILED. Recent driver reflex lines:"
  grep -h -i 'reflex' /home/aup/.ros/log/*/ros2_control_node*.log 2>/dev/null | tail -2 | cut -c1-200 || true
  echo "Inspect the current robot error and driver log before recovery. Servo/relay are left running to preserve target output."
  exit 1
fi
echo "=== 5. waiting for servo idle latch"
deadline=$((SECONDS + 65))
while true; do
  ST=$(status)
  case "$ST" in
    *'"started":false,"fault":false'*'"idle_latched":true'*) echo "$ST"; break;;
    *'"started":false,"fault":false'*) ;;
    *) echo "ABORT: unexpected servo state: $ST"; exit 1;;
  esac
  (( SECONDS < deadline )) || { echo "ABORT: servo idle latch timed out"; exit 1; }
  sleep 2
done
for S in left right; do
  [ "$(imp "$S")" = "active" ] && [ "$(mode "$S")" = "2" ] && [ "$(errs "$S")" = "0" ] || {
    echo "ABORT: $S did not remain active/error-free through idle latch"; exit 1;
  }
done
echo "READY: both impedance controllers active, servo latched. Run: bash scripts/run_smolvla_loop.sh"
