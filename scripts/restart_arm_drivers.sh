#!/usr/bin/env bash
# Restart both arm drivers (broadcasters only, no command controller) and restore
# current_pose publishing. Use this when reconnect_arm_state.sh fails with a
# libfranka "Net Exception": the driver's libfranka session is stale after a
# Desk Programming/Execution switch or a robot controller reboot, and only a
# driver restart recreates it. No robot motion is commanded.
#
# Usage: bash scripts/restart_arm_drivers.sh
# Refuses if a joint_impedance_controller is active or a gello target publisher exists.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p log/drivers
STAMP="$(date +%Y%m%d_%H%M%S)"
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}" >/dev/null 2>&1
[ -f site/install/setup.bash ] && source site/install/setup.bash
set -u
strip() { sed $'s/\033\[[0-9;]*m//g'; }
declare -A IP=([left]=172.16.16.12 [right]=172.16.16.11)

echo "=== 1. safety checks"
for S in left right; do
  st=$(timeout 8 ros2 control list_controllers -c "/$S/controller_manager" 2>/dev/null | strip | awk '$1 == "joint_impedance_controller" {print $NF}')
  [ "$st" = active ] && { echo "ABORT: $S joint_impedance_controller is active; deactivate it first"; exit 1; }
  n=$(timeout 6 ros2 topic info "/$S/gello/joint_states" 2>/dev/null | awk '/Publisher count:/ {print $3}')
  [ "${n:-0}" = 0 ] || { echo "ABORT: /$S/gello/joint_states has $n publisher(s); stop servo/relay first"; exit 1; }
done
for ip in "${IP[@]}"; do
  timeout 3 bash -c "echo > /dev/tcp/$ip/1337" 2>/dev/null || { echo "ABORT: $ip:1337 unreachable; robot off, FCI disabled, or Desk not in Execution"; exit 1; }
done

echo "=== 2. stop old arm drivers by exact PID (grippers untouched)"
arm_pids() {
  ps -eo pid=,args= | awk '
    $2 ~ /ros2_control_node$/ && / __ns:=\/(left|right) / {print $1; next}
    $2 ~ /(^|\/)(python[0-9.]*|ros2)$/ && / launch / && / franka_fr3_arm_controllers / && / franka(_fr3_arm_controllers)?\.launch\.py/ {print $1}'
}
OLD=$(arm_pids)
if [ -n "$OLD" ]; then
  echo "stopping: $(echo $OLD | tr '\n' ' ')"
  for P in $OLD; do kill -INT "$P" 2>/dev/null || true; done
  for _ in $(seq 1 20); do [ -z "$(arm_pids)" ] && break; sleep 0.5; done
  for P in $(arm_pids); do kill -9 "$P" 2>/dev/null || true; done
  sleep 1
fi
[ -z "$(arm_pids)" ] || { echo "ABORT: old driver processes still alive: $(arm_pids | tr '\n' ' ')"; exit 1; }
screen -S arms -X quit 2>/dev/null || true

echo "=== 3. start drivers (broadcasters only)"
for S in left right; do
  LOG="log/drivers/${S}_$STAMP.log"
  nohup setsid ros2 launch franka_fr3_arm_controllers franka.launch.py \
    arm_id:=fr3v2 arm_prefix:="$S" namespace:="$S" robot_ip:="${IP[$S]}" \
    load_gripper:=false joint_sources:=joint_states > "$LOG" 2>&1 < /dev/null &
  echo "$S driver -> $LOG"
done
for S in left right; do
  ok=0
  for _ in $(seq 1 60); do
    timeout 5 ros2 control list_controllers -c "/$S/controller_manager" >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  [ $ok = 1 ] || { echo "ABORT: /$S/controller_manager did not come up; see log/drivers/${S}_$STAMP.log"; grep -iE 'error|exception|refused' "log/drivers/${S}_$STAMP.log" | tail -3 | cut -c1-200; exit 1; }
done
sleep 2

echo "=== 4. restore state broadcasters"
bash scripts/reconnect_arm_state.sh --no-restart
rc=$?
for S in left right; do
  if grep -q 'Net Exception' "log/drivers/${S}_$STAMP.log"; then
    echo "$S: libfranka Net Exception right after a fresh start. Check Desk: Execution mode, FCI active, joints unlocked, robot light blue, no task running."
    rc=1
  fi
done
exit $rc
