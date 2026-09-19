#!/usr/bin/env bash
# Replay recorded actions, with per-row dataset references and continuous tracking.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/labs_replay_episode.sh --episode 0 [options]
Default: full offline IK and continuous trajectory validation at 9 Hz.
Add --publish --enable-robot to return to the episode start and replay all
recorded actions through the running site relay. Grippers follow the recording.
Return uses the same rate/30 time scale; interpolation stays at 100 Hz.
Options: --episode N, --rate HZ, --dataset PATH, --output PATH, --ik PATH.
The enabled relay must have the same dataset and --start-episode configured.
No model connection is made. Accepted episodes finish even if the client exits.
HELP
      exit 0 ;;
  esac
done
set +u
source "${LABS_ROS_SETUP:-/opt/ros/humble/setup.bash}"
source site/install/setup.bash
set -u
export ROS_DOMAIN_ID="${LABS_ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-/home/agile/work/labs/deployments/example_station/cyclonedds.xml}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${LABS_PYTHON:-.venv/bin/python}" -m franka_duo_tele_data.labs_replay \
  --dataset "${LABS_DATASET:-/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916}" \
  --ik site/install/labs_fr3_kinematics/lib/labs_fr3_kinematics/labs_fr3_ik "$@"
