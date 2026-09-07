#!/usr/bin/env bash
# Reconnect both arm drivers to FCI after switching Desk back to Execution mode,
# and restore the read-only state broadcasters so current_pose publishes again.
# Does NOT activate any command controller (impedance stays inactive), so the
# running ZED PnP web calibration can continue capturing.
#
# If activation fails with a stale libfranka session (Net Exception), the driver
# processes are restarted via scripts/restart_arm_drivers.sh unless --no-restart.
#
# Usage: bash scripts/reconnect_arm_state.sh [left|right|both] [--no-restart]
set -uo pipefail
WHICH=both; NO_RESTART=0
for a in "$@"; do
  case "$a" in left|right|both) WHICH=$a;; --no-restart) NO_RESTART=1;; *) echo "usage: $0 [left|right|both] [--no-restart]" >&2; exit 2;; esac
done
case "$WHICH" in left) SIDES="left";; right) SIDES="right";; both) SIDES="left right";; esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}" >/dev/null 2>&1
[ -f site/install/setup.bash ] && source site/install/setup.bash
set -u
strip() { sed $'s/\033\[[0-9;]*m//g'; }
hw_state() { timeout 8 ros2 control list_hardware_components -c "/$1/controller_manager" 2>/dev/null | strip | grep -o 'label=[a-z]*' | head -1; }
ctrl_state() { timeout 8 ros2 control list_controllers -c "/$1/controller_manager" 2>/dev/null | strip | awk -v c="$2" '$1 == c {print $NF}'; }
rc=0; stale=0
for S in $SIDES; do
  echo "=== $S"
  if ! timeout 8 ros2 control list_controllers -c "/$S/controller_manager" >/dev/null 2>&1; then
    echo "$S: controller_manager not reachable; driver not running"; rc=1; stale=1; continue
  fi
  imp=$(ctrl_state "$S" joint_impedance_controller)
  [ "$imp" = active ] && { echo "$S: joint_impedance_controller is ACTIVE; refusing to touch hardware state"; rc=1; continue; }
  st=$(hw_state "$S"); echo "hardware before: ${st:-unknown}"
  if [ "$st" != "label=active" ]; then
    timeout 40 ros2 control set_hardware_component_state "${S}_FrankaHardwareInterface" active -c "/$S/controller_manager" 2>&1 | tail -1
  fi
  st=$(hw_state "$S"); echo "hardware after:  ${st:-unknown}"
  if [ "$st" != "label=active" ]; then
    echo "$S: hardware did not become active (stale libfranka session or robot not ready)"; rc=1; stale=1; continue
  fi
  need=""
  for C in joint_state_broadcaster franka_robot_state_broadcaster; do
    [ "$(ctrl_state "$S" "$C")" = active ] || need="$need $C"
  done
  if [ -n "$need" ]; then
    # shellcheck disable=SC2086
    timeout 40 ros2 control switch_controllers -c "/$S/controller_manager" --activate $need 2>&1 | tail -1
  fi
  hz=$(timeout 6 ros2 topic hz "/$S/franka_robot_state_broadcaster/current_pose" 2>&1 | grep -m1 -oE 'average rate: [0-9.]+' || true)
  mode=$(timeout 5 ros2 topic echo --once "/$S/franka_robot_state_broadcaster/robot_state" --field robot_mode --qos-reliability best_effort 2>/dev/null | head -1)
  errs=$(timeout 5 ros2 topic echo --once "/$S/franka_robot_state_broadcaster/robot_state" --field current_errors --qos-reliability best_effort 2>/dev/null | grep -c ': true')
  echo "current_pose ${hz:-NOT PUBLISHING}; robot_mode=${mode:-?} (1=idle); errors=${errs:-?}"
  [ -n "$hz" ] || { rc=1; stale=1; }
done
if [ $rc -ne 0 ] && [ $stale = 1 ] && [ $NO_RESTART = 0 ]; then
  echo "=== activation failed; restarting arm drivers to rebuild the libfranka session"
  exec bash scripts/restart_arm_drivers.sh
fi
[ $rc -eq 0 ] && echo "READY: web calibration can capture again" || echo "NOT READY (see above)"
exit $rc
