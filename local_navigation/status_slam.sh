#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/slam.pid"
if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "SLAM running (PID $(cat "${pid_file}"))"
  pgrep -af "dual_lidar_slam.launch.py|dual_laser_merger|async_slam_toolbox_node" || true
else
  echo "SLAM not running"
  exit 1
fi
