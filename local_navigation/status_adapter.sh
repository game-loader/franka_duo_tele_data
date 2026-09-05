#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/adapter.pid"

if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "adapter running (PID $(cat "${pid_file}"))"
  pgrep -af "navigation_adapter.launch.py|odom_frame_adapter|static_transform_publisher" || true
else
  echo "adapter not running"
  exit 1
fi
