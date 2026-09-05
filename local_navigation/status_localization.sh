#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/localization.pid"
if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "localization running (PID $(cat "${pid_file}"))"
  pgrep -af "localization.launch.py|dual_laser_merger|map_server|amcl" || true
else
  echo "localization not running"; exit 1
fi
