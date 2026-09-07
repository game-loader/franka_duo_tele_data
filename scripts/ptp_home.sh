#!/usr/bin/env bash
# Start missing arm drivers, deactivate impedance control, and PTP both arms to
# the saved home poses in configs/ptp_home_target.json. This command moves the
# robot; driver bring-up alone only
# loads state broadcasters. Existing drivers are reused, never killed/restarted.
# Usage: bash scripts/ptp_home.sh [MAX_JOINT_VELOCITY_RAD_S]   (default 0.05)
# PTP_DRIVER_TIMEOUT_S optionally changes the 60-second driver readiness wait.
set -euo pipefail
VEL="${1:-0.05}"
DRIVER_TIMEOUT="${PTP_DRIVER_TIMEOUT_S:-60}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p log/ptp_home outputs
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_LOG="$PWD/log/ptp_home/run_$STAMP.log"
LOG="$PWD/log/ptp_home/ptp_$STAMP.log"
TARGET=configs/ptp_home_target.json

# Do not let two copies race to start drivers or send PTP goals. Background
# drivers close fd 9 so they do not retain this lock after the script exits.
exec 9>log/ptp_home/.lock
flock -n 9 || { echo "ABORT: another ptp_home.sh is running" >&2; exit 1; }
exec > >(tee -a "$RUN_LOG" 9>&-) 2>&1
trap 'rc=$?; echo "ABORT: line $LINENO failed (exit $rc); see $RUN_LOG" >&2; exit "$rc"' ERR
die() { echo "ABORT: $*; see $RUN_LOG" >&2; exit 1; }
echo "PTP home run log: $RUN_LOG"
[[ "$VEL" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
  awk -v velocity="$VEL" 'BEGIN { exit !(velocity > 0) }' || die "velocity must be positive"
[[ "$DRIVER_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "PTP_DRIVER_TIMEOUT_S must be a positive integer"
[[ -f "$TARGET" ]] || die "saved home target missing: $TARGET; restore it before homing"
echo "Home target: $PWD/$TARGET"
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}"
source site/install/setup.bash
set -u

controllers() {
  local result
  if ! result=$(timeout 5 ros2 control list_controllers -c "/$1/controller_manager" 2>>"$RUN_LOG"); then
    return 1
  fi
  # ros2controlcli can color the state even when its stdout is captured.
  printf '%s\n' "$result" | sed $'s/\033\\[[0-9;]*m//g'
}
controller_state() { awk -v name="$1" '$1 == name { print $NF }'; }
driver_process_exists() {
  # Match arm namespaces exactly: /left/gripper is a different driver.
  ps -eo args= | awk -v side="$1" -v ip="$2" '
    {
      control = ($1 ~ /(^|\/)ros2_control_node$/)
      launch = (($1 ~ /(^|\/)ros2$/ && $2 == "launch") ||
                ($1 ~ /(^|\/)python[0-9.]*$/ && $2 ~ /(^|\/)ros2$/ && $3 == "launch"))
      for (i = 1; i <= NF; i++) {
        if (control && $i == "__ns:=/" side) found = 1
        if (launch && ($i == "namespace:=" side || $i == "robot_ip:=" ip)) found = 1
      }
    }
    END { exit !found }'
}
ready_controllers() {
  local result
  result=$(controllers "$1") || return 1
  [[ "$(printf '%s\n' "$result" | controller_state joint_state_broadcaster)" == active &&
     "$(printf '%s\n' "$result" | controller_state franka_robot_state_broadcaster)" == active ]]
}

echo "=== 0. ensure both arm drivers are running"
SIDES=(left right)
IPS=(172.16.16.12 172.16.16.11)
DRIVER_PIDS=("" "")
for i in 0 1; do
  side="${SIDES[$i]}"
  if controllers "$side" >/dev/null; then
    echo "$side: reusing /$side/controller_manager"
  elif driver_process_exists "$side" "${IPS[$i]}"; then
    echo "$side: driver process already exists; waiting for its controller manager"
  else
    driver_log="$PWD/log/ptp_home/driver_${side}_$STAMP.log"
    echo "$side: starting driver (${IPS[$i]}); log: $driver_log"
    nohup setsid ros2 launch franka_fr3_arm_controllers franka.launch.py \
      arm_id:=fr3v2 arm_prefix:="$side" namespace:="$side" robot_ip:="${IPS[$i]}" \
      load_gripper:=false use_fake_hardware:=false joint_sources:=joint_states \
      >"$driver_log" 2>&1 </dev/null 9>&- &
    DRIVER_PIDS[$i]=$!
  fi
done
for i in 0 1; do
  side="${SIDES[$i]}"
  deadline=$((SECONDS + DRIVER_TIMEOUT))
  until ready_controllers "$side"; do
    if [[ -n "${DRIVER_PIDS[$i]}" ]] && ! kill -0 "${DRIVER_PIDS[$i]}" 2>/dev/null; then
      die "$side driver exited; inspect log/ptp_home/driver_${side}_$STAMP.log"
    fi
    (( SECONDS < deadline )) || die "$side controller manager/state broadcasters not ready after ${DRIVER_TIMEOUT}s (driver logs: log/ptp_home/)"
    sleep 1
  done
  echo "$side: controller manager and state broadcasters ready"
done

echo "=== 1. deactivate impedance controllers"
for side in left right; do
  state_list=$(controllers "$side") || die "$side controller manager unavailable"
  state=$(printf '%s\n' "$state_list" | controller_state joint_impedance_controller)
  if [[ "$state" == active ]]; then
    timeout 20 ros2 control switch_controllers -c "/$side/controller_manager" \
      --deactivate joint_impedance_controller || die "could not deactivate $side impedance controller"
  fi
done

echo "=== 2. verify live feedback and PTP servers"
for side in left right; do
  state_list=$(controllers "$side") || die "$side controller manager unavailable"
  active_commands=$(printf '%s\n' "$state_list" | awk '
    $NF == "active" && $1 != "joint_state_broadcaster" && $1 != "franka_robot_state_broadcaster" { print $1 }')
  [[ -z "$active_commands" ]] || die "$side still has active controllers: $active_commands"
  # Missing feedback must never be interpreted as zero errors. Inspect current
  # errors only; last_motion_errors may describe an already recovered fault.
  errors=$(timeout 5 ros2 topic echo --once "/$side/franka_robot_state_broadcaster/robot_state" \
    --field current_errors --qos-reliability best_effort 2>>"$RUN_LOG") || die "$side robot_state unavailable"
  printf '%s\n' "$errors" | grep -Eq '^[[:space:]]*[a-zA-Z0-9_]+: (true|false)$' || die "$side current_errors could not be read"
  if printf '%s\n' "$errors" | grep -Eq ': true$'; then
    printf '%s\n' "$errors" | grep ': true$'
    die "$side has current robot errors; recovery is required before PTP"
  fi
  for topic in measured_joint_states current_pose; do
    timeout 5 ros2 topic echo --once "/$side/franka_robot_state_broadcaster/$topic" \
      --field header --qos-reliability best_effort >>"$RUN_LOG" 2>&1 || die "$side $topic unavailable"
  done
  deadline=$((SECONDS + DRIVER_TIMEOUT))
  while true; do
    action_info=$(timeout 5 ros2 action info "/$side/action_server/ptp_motion" 2>>"$RUN_LOG") || action_info=""
    if printf '%s\n' "$action_info" | grep -Eq '^Action servers: 1[[:space:]]*$'; then
      break
    fi
    (( SECONDS < deadline )) || die "$side requires exactly one PTP action server; readiness timed out"
    sleep 1
  done
  echo "$side: live state received, no current robot errors, PTP server ready"
done

echo "=== 3. PTP to saved home poses (max joint velocity $VEL rad/s)"
ptp_rc=0
timeout 240 ros2 launch franka_duo_ptp_step duo_ptp_episode.launch.py \
  action_file:="$PWD/$TARGET" action_index:=0 \
  max_joint_velocity:="$VEL" max_joint_delta_rad:=2.5 wait_timeout_s:=200.0 \
  log_file:="$PWD/outputs/ptp_home_last.csv" execute:=true confirm:=true >"$LOG" 2>&1 || ptp_rc=$?
grep -i 'delta=\|PTP results\|error\|fatal\|reflex' "$LOG" | grep -v launch | tail -8 || true
[[ "$ptp_rc" == 0 ]] || die "PTP failed (exit $ptp_rc); inspect $LOG"
grep -q 'left_code=4 left_status=2 right_code=4 right_status=2' "$LOG" || die "PTP did not complete for both arms; inspect $LOG"
echo "PTP OK: both arms TARGET_REACHED"
echo "Arm drivers remain running; command controllers are inactive."
echo "To resume inference, run: bash scripts/run_smolvla_loop.sh"
