#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/slam.pid"
if [[ ! -f "${pid_file}" ]]; then
  echo "SLAM is not managed by this directory"
  exit 0
fi
pid="$(cat "${pid_file}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill -TERM -- "-${pid}"
  echo "SLAM process group ${pid} stopped"
fi
mv "${pid_file}" "${pid_file}.stopped"
