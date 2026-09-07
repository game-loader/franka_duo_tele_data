#!/usr/bin/env bash
# Start-to-table cup/bowl mission: stow arms, raise spine, drive to the table,
# then per object lower the spine to the calibrated grasp height, clear the
# camera view, detect with YOLO, pose up and grasp.
#
# Every arm motion goes through the joint servo into the site
# joint_impedance_controller. PTP is not used anywhere in this path.
#
# Usage: [EXECUTE=1] bash scripts/run_table_mission.sh [SPEED] [extra args...]
#   e.g.  bash scripts/run_table_mission.sh                 # print the strategy only
#         EXECUTE=1 bash scripts/run_table_mission.sh 0.1   # drives and grasps
#
# The base host must already have its navigation stack up
# (scripts/19_ensure_navigation_stack.sh and 17_control_mode.sh mission there).
set -euo pipefail
SPEED="${1:-0.1}"
shift $(( $# > 0 ? 1 : 0 )) || true
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p outputs log/table_mission
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}" >/dev/null 2>&1
source site/install/setup.bash
set -u
# Append, never replace: overwriting PYTHONPATH hides the host's ROS packages.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "${EXECUTE:-0}" = "1" ]; then
  echo "EXECUTING table mission at speed $SPEED"
  exec .venv/bin/python -m franka_duo_tele_data.table_mission \
    --speed "$SPEED" --execute "$@" 2>&1 | grep --line-buffered -v rmw_cyclonedds
fi
echo "DRY-RUN (set EXECUTE=1 to drive and grasp)"
exec .venv/bin/python -m franka_duo_tele_data.table_mission \
  --speed "$SPEED" "$@" 2>&1 | grep --line-buffered -v rmw_cyclonedds
