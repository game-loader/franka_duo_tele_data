#!/usr/bin/env bash
set -eo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/adapter.pid"
log_file="${root_dir}/adapter.log"

if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "adapter already running (PID $(cat "${pid_file}"))"
  exit 0
fi

source /opt/ros/humble/setup.bash
source "${HOME}/ros2_ws/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${root_dir}/install/tmr_local_navigation:${AMENT_PREFIX_PATH}"
export PYTHONPATH="${root_dir}/install/tmr_local_navigation/lib/python3.10/dist-packages:${root_dir}/install/tmr_local_navigation/lib/python3.10/site-packages:${PYTHONPATH:-}"

setsid ros2 launch tmr_local_navigation navigation_adapter.launch.py \
  >"${log_file}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" >"${pid_file}"
echo "adapter started (PID ${pid}); log: ${log_file}"
