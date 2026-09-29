#!/usr/bin/env bash
# Current-pose follower startup and recovery; host/site drivers remain site-owned.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      cat <<'HELP'
Usage:
  bash scripts/labs_robot_control.sh --status
  bash scripts/labs_robot_control.sh --start --publish --enable-robot
  bash scripts/labs_robot_control.sh --start --restart-relay --publish --enable-robot
  bash scripts/labs_robot_control.sh --recover --publish --enable-robot

--status   Read driver/controller/relay/feedback status (default).
--start    Start existing site driver containers if stopped; activate followers
           through a stationary current-pose relay hold. Already ready is a no-op.
--recover  Deactivate followers, stop old relay, request both Franka hardware
           error-recovery actions, then re-establish current-pose control.
--restart-relay  With --start, reload relay code/task starts while re-establishing
                 current-pose hold; does not perform hardware error recovery.

Stop inference/replay scripts before start/recover. The script stops the site's
controller-coordinator to prevent competing controller switches. Existing
foreign target publishers are reported, never killed. Recovery does not return
to an episode pose, open the grippers, run inference or replay old actions.
Physical E-stop/FCI/manual-recovery conditions must be cleared on the robot.
--start/--recover require BOTH --publish --enable-robot.
Logs: outputs/labs_control_service/latest.json and outputs/labs_relay/relay.log.
HELP
      exit 0 ;;
  esac
done
set +u
source "${LABS_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f site/install/setup.bash ]]; then source site/install/setup.bash; fi
set -u
export ROS_DOMAIN_ID="${LABS_ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-/home/agile/work/labs/deployments/example_station/cyclonedds.xml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${LABS_PYTHON:-.venv/bin/python}" -m franka_duo_tele_data.labs_control_service "$@"
