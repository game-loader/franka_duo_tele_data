#!/usr/bin/env bash
# Host-managed ROS 2; default operation never publishes robot commands.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set +u
source "${LABS_ROS_SETUP:-/opt/ros/humble/setup.bash}"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-${LABS_DDS_CONFIG:-/home/agile/work/labs/deployments/example_station/cyclonedds.xml}}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${LABS_PYTHON:-.venv/bin/python}" -m franka_duo_tele_data.labs_client "$@"
